"""Joint Point-MAE + VICReg + DEC mitochondrial clustering experiment."""
from __future__ import annotations
import argparse, csv, gc, json, math, time
from pathlib import Path
from typing import Any
import numpy as np
import torch, yaml, zarr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from pointmae import PointCloudCache, PointMAE, augmented_point_view, model_config_from_yaml, seed_everything, split_by_neuron

class Indexed(Dataset):
    def __init__(self, base, indices): self.base, self.indices = base, np.asarray(indices, dtype=np.int64)
    def __len__(self): return len(self.indices)
    def __getitem__(self, pos):
        i = int(self.indices[pos]); item = self.base[i]
        item["cache_index"], item["subset_position"] = i, int(pos)
        return item

class VICRegProjector(nn.Module):
    def __init__(self, d, hidden, out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Linear(hidden, out))
    def forward(self, x): return self.net(x)

class DECHead(nn.Module):
    def __init__(self, centers, alpha=1.0):
        super().__init__()
        self.centers = nn.Parameter(torch.as_tensor(centers, dtype=torch.float32))
        self.alpha = float(alpha)
    def forward(self, h):
        d2 = torch.cdist(h.float(), self.centers.float()).square()
        q = (1 + d2 / self.alpha).pow(-(self.alpha + 1) / 2)
        return q / q.sum(1, keepdim=True).clamp_min(1e-12)

def loader(base, indices, bs, workers, device, shuffle=False, seed=42, drop=False):
    return DataLoader(Indexed(base, indices), batch_size=bs, shuffle=shuffle, drop_last=drop,
                      num_workers=workers, pin_memory=device.type == "cuda",
                      generator=torch.Generator().manual_seed(seed),
                      persistent_workers=workers > 0)

def amp_ctx(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=enabled and device.type == "cuda")

def encoder_embedding(model, xyz, attrs):
    # Required downstream embedding: mean-pooled encoder tokens, before model.projection.
    centers, gx, ga = model.group(xyz, attrs)
    return model.encoder(model._tokens(centers, gx, ga)).mean(dim=1)

@torch.no_grad()
def extract(model, data_loader, device, amp):
    model.eval(); blocks = []
    for b in data_loader:
        with amp_ctx(device, amp):
            h = encoder_embedding(model, b["xyz"].to(device, non_blocking=True), b["attrs"].to(device, non_blocking=True))
        blocks.append(h.float().cpu().numpy())
    return np.concatenate(blocks)

def vicreg(z1, z2, c):
    z1, z2 = z1.float(), z2.float()
    inv = F.mse_loss(z1, z2)
    def var(z): return F.relu(float(c["gamma"]) - torch.sqrt(z.var(0, unbiased=True) + float(c["eps"]))).mean()
    def cov(z):
        z = z - z.mean(0); m = z.T @ z / max(len(z)-1, 1)
        return (m.square().sum() - m.diagonal().square().sum()) / m.shape[0]
    v, cv = var(z1)+var(z2), cov(z1)+cov(z2)
    total = float(c["invariance_weight"])*inv + float(c["variance_weight"])*v + float(c["covariance_weight"])*cv
    return total, {"vic_invariance": inv, "vic_variance": v, "vic_covariance": cv}

def target_dist(q):
    w = q.square() / q.sum(0, keepdim=True).clamp_min(1e-12)
    return w / w.sum(1, keepdim=True).clamp_min(1e-12)

def refresh_targets(model, dec, base, train_idx, cfg, device, amp):
    l = loader(base, train_idx, cfg["train"]["batch_size"], cfg["train"]["num_workers"], device)
    h = extract(model, l, device, amp)
    with torch.no_grad():
        q = dec(torch.as_tensor(h, device=device))
    return target_dist(q).detach(), q.detach().sum(0).cpu().numpy()

def eval_mae(model, l, device, amp):
    model.eval(); s=n=0
    with torch.no_grad():
        for b in l:
            xyz,attrs=b["xyz"].to(device),b["attrs"].to(device)
            with amp_ctx(device,amp): loss=model(xyz,attrs,return_loss=True)["loss"]
            s += float(loss.float())*len(xyz); n += len(xyz)
    return s/max(n,1)

def train_one(mode, k, cfg, mcfg, base, train_idx, val_idx, init_state, args, device, out):
    seed_everything(cfg["seed"])
    model=PointMAE(mcfg,cfg["data"]["feature_names"]).to(device); model.load_state_dict(init_state,strict=True)
    vc=cfg["objective"]["vicreg"]; dc=cfg["objective"]["cluster"]
    vh=VICRegProjector(mcfg["encoder_dim"],vc["hidden_dim"],vc["projection_dim"]).to(device) if mode in ("B","D") else None
    dec=None
    if mode in ("C","D"):
        h0=extract(model,loader(base,train_idx,cfg["train"]["batch_size"],cfg["train"]["num_workers"],device),device,cfg["train"]["amp"])
        km=KMeans(n_clusters=k,n_init=dc["kmeans_n_init"],random_state=cfg["seed"]).fit(h0)
        dec=DECHead(km.cluster_centers_,dc["alpha"]).to(device)
    params=list(model.parameters())+(list(vh.parameters()) if vh is not None else [])+(list(dec.parameters()) if dec is not None else [])
    opt=torch.optim.AdamW(params,lr=cfg["train"]["lr"],weight_decay=cfg["train"]["weight_decay"])
    tr=loader(base,train_idx,cfg["train"]["batch_size"],cfg["train"]["num_workers"],device,True,cfg["seed"]+1000,True)
    vl=loader(base,val_idx,cfg["train"]["batch_size"],cfg["train"]["num_workers"],device)
    amp=cfg["train"]["amp"] and device.type=="cuda"; ptarget=None; history=[]
    for ep in range(args.epochs or cfg["train"]["epochs"]):
        if dec is not None and (ptarget is None or ep % dc["update_every_epochs"] == 0):
            ptarget,mass=refresh_targets(model,dec,base,train_idx,cfg,device,amp)
            print(json.dumps({"mode":mode,"k":k,"epoch":ep,"target_mass":mass.tolist()}),flush=True)
        model.train()
        if vh: vh.train()
        totals={}; seen=0; start=time.time()
        for step,b in enumerate(tr):
            x,a=b["xyz"].to(device,non_blocking=True),b["attrs"].to(device,non_blocking=True)
            pos=b["subset_position"].to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
            with amp_ctx(device,amp):
                terms=model(x,a,return_loss=True); mae=terms["loss"].float(); zero=mae.detach()*0
                vloss,vt=zero,{}
                if vh:
                    x1,a1=augmented_point_view(x,a,model.attr_names,vc["keep_ratio"])
                    x2,a2=augmented_point_view(x,a,model.attr_names,vc["keep_ratio"])
                    h1=encoder_embedding(model,x1,a1); h2=encoder_embedding(model,x2,a2)
                    z=vh(torch.cat((h1,h2),0)); z1,z2=z.chunk(2,0)
                    vloss,vt=vicreg(z1,z2,vc)
                closs=zero
                if dec:
                    h=encoder_embedding(model,x,a); q=dec(h)
                    closs=F.kl_div(q.clamp_min(1e-8).log(),ptarget[pos],reduction="batchmean")
                warm=min(1.0,(ep+1)/max(float(dc["warmup_epochs"]),1.0)) if dec else 0.0
                total=mae+(vc["lambda"]*vloss if vh else 0)+(dc["lambda"]*warm*closs if dec else 0)
            total.backward()
            if cfg["train"]["grad_clip"]>0: torch.nn.utils.clip_grad_norm_(params,cfg["train"]["grad_clip"])
            opt.step(); bs=len(x); seen+=bs
            vals={"total":total,"mae":mae,"vicreg_weighted":vc["lambda"]*vloss if vh else zero,
                  "cluster_kl":closs,"cluster_weighted":dc["lambda"]*warm*closs if dec else zero,**vt}
            for name,val in vals.items(): totals[name]=totals.get(name,0)+float(val.detach())*bs
            if args.max_train_steps and step+1>=args.max_train_steps: break
        row={"mode":mode,"k":k,"epoch":ep+1,"train":{n:v/max(seen,1) for n,v in totals.items()},
             "val_mae":eval_mae(model,vl,device,amp),"cluster_warmup":warm if dec else 0,
             "steps":min(len(tr),args.max_train_steps) if args.max_train_steps else len(tr),"seconds":round(time.time()-start,2)}
        history.append(row); print(json.dumps(row),flush=True)
        with (out/"metrics.jsonl").open("a") as f: f.write(json.dumps(row)+"\n")
        state={"mode":mode,"k":k,"epoch":ep+1,"model":model.state_dict(),
               "vicreg_head":vh.state_dict() if vh else None,"dec_head":dec.state_dict() if dec else None,
               "config":cfg,"model_config":mcfg,"init_checkpoint":args.init_checkpoint}
        tmp=out/"last.tmp.pt"; torch.save(state,tmp); tmp.replace(out/"last.pt")
        if args.max_train_steps: break
    (out/"training_summary.json").write_text(json.dumps(history,indent=2))
    return model,vh,dec

def metadata_for(n, checkpoint):
    d=Path(checkpoint).parent/"diagnostics"; out=[{"cache_index":i} for i in range(n)]
    for p in [d/"umap_coordinates_morphology.csv",d/"leiden_celltype_pc30_24neurons_20261011_assignments.csv"]:
        if p.exists():
            with p.open(encoding="utf-8",newline="") as f:
                for row in csv.DictReader(f):
                    if "cache_index" in row: out[int(row["cache_index"])].update(row)
    return out

def stats(x):
    x=np.asarray(x,dtype=np.float64); s=x.std(0); y=x-x.mean(0); ev=np.clip(np.linalg.eigvalsh(y.T@y/max(len(x)-1,1)),0,None)[::-1]
    p=ev/max(ev.sum(),1e-12); p=p[p>1e-12]
    return {"n":len(x),"dim":x.shape[1],"mean_feature_std":float(s.mean()),"median_feature_std":float(np.median(s)),
            "norm_mean":float(np.linalg.norm(x,axis=1).mean()),"effective_rank":float(np.exp(-(p*np.log(p)).sum())),
            "pca_top5":float(np.sort(ev)[::-1][:5].sum()/max(ev.sum(),1e-12)),
            "pca_top20":float(np.sort(ev)[::-1][:20].sum()/max(ev.sum(),1e-12))}

def cluster_stability(xtrain,x,k,seeds=10):
    ls=[]
    for s in range(seeds):
        km=KMeans(n_clusters=k,n_init=1,random_state=s).fit(xtrain); ls.append(km.predict(x))
    a=[]; n=[]
    for i in range(len(ls)):
        for j in range(i+1,len(ls)):
            a.append(adjusted_rand_score(ls[i],ls[j])); n.append(normalized_mutual_info_score(ls[i],ls[j]))
    return {"frozen_embedding_kmeans_seed_ari_mean":float(np.mean(a)),"frozen_embedding_kmeans_seed_ari_sd":float(np.std(a)),
            "frozen_embedding_kmeans_seed_nmi_mean":float(np.mean(n))}

def vals(meta,name):
    out=[]
    for r in meta:
        try: v=float(r[name]); out.append(v if math.isfinite(v) else np.nan)
        except (KeyError,ValueError,TypeError): out.append(np.nan)
    return np.asarray(out)

def analyze(run,x,xtrain,labels,probs,centers,k,kind,meta,ids,neurons,seed):
    d=run/f"k{k:02d}"; d.mkdir(parents=True,exist_ok=True); counts=np.bincount(labels,minlength=k)
    ent=-(probs*np.log(np.clip(probs,1e-12,1))).sum(1); cell=np.array([str(r.get("cell_type","")) for r in meta])
    known=np.array([bool(v) and "manual" not in v.lower() and v.lower()!="nan" for v in cell])
    m={"k":k,"probability_kind":kind,"counts":counts.tolist(),"fractions":(counts/counts.sum()).tolist(),
       "nonempty_clusters":int((counts>0).sum()),"largest_fraction":float(counts.max()/counts.sum()),
       "collapsed":bool((counts==0).any() or counts.max()/counts.sum()>.9),
       "assignment_entropy_mean":float(ent.mean()),"max_probability_mean":float(probs.max(1).mean()),
       **cluster_stability(xtrain,x,k)}
    if known.sum()>1 and len(np.unique(cell[known]))>1:
        m["cell_type_ari_predictions_only"]=float(adjusted_rand_score(cell[known],labels[known]))
        m["cell_type_nmi_predictions_only"]=float(normalized_mutual_info_score(cell[known],labels[known]))
    try:
        xn=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
        m["cosine_silhouette_sample500"]=float(silhouette_score(xn,labels,metric="cosine",sample_size=min(500,len(x)),random_state=seed))
    except Exception as e: m["silhouette_error"]=str(e)
    np.savez_compressed(d/"assignment_probabilities.npz",labels=labels.astype("int32"),probabilities=probs.astype("float32"),
                        centers=centers.astype("float32"),probability_kind=np.asarray(kind))
    mf=["volume_um3","surface_area_um2","bbox_aspect_ratio","sphericity","ellipsoid_volume_agreement"]
    with (d/"assignments.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["cache_index","instance_id","neuron_id","cluster","max_probability","cell_type"]+mf); w.writeheader()
        for i,r in enumerate(meta):
            w.writerow({"cache_index":i,"instance_id":int(ids[i]),"neuron_id":int(neurons[i]),"cluster":int(labels[i]),
                        "max_probability":float(probs[i].max()),"cell_type":r.get("cell_type",""),**{n:r.get(n,"") for n in mf}})
    rows=[]
    for c in range(k):
        mask=labels==c; row={"cluster":c,"n":int(mask.sum()),"fraction":float(mask.mean())}
        for name in mf:
            z=vals(meta,name)[mask]; z=z[np.isfinite(z)]
            if len(z): row.update({name+"_median":float(np.median(z)),name+"_q25":float(np.quantile(z,.25)),name+"_q75":float(np.quantile(z,.75))})
        rows.append(row)
    keys=sorted({a for r in rows for a in r})
    with (d/"cluster_morphology_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    (d/"summary.json").write_text(json.dumps(m,indent=2))
    return m

@torch.no_grad()
def sampling_stability(model,base,indices,cfg,device):
    seed_everything(cfg["seed"]+404); sims=[]; model.eval()
    l=loader(base,indices,cfg["train"]["batch_size"],cfg["train"]["num_workers"],device)
    for b in l:
        x,a=b["xyz"].to(device),b["attrs"].to(device)
        x1,a1=augmented_point_view(x,a,model.attr_names,cfg["objective"]["vicreg"]["keep_ratio"])
        x2,a2=augmented_point_view(x,a,model.attr_names,cfg["objective"]["vicreg"]["keep_ratio"])
        with amp_ctx(device,cfg["train"]["amp"]):
            h1=encoder_embedding(model,x1,a1); h2=encoder_embedding(model,x2,a2)
        sims.extend(F.cosine_similarity(h1.float(),h2.float()).cpu().tolist())
    return {"sampling_stability_n":len(sims),"same_instance_resampling_cosine_mean":float(np.mean(sims)),
            "same_instance_resampling_cosine_median":float(np.median(sims))}

def plot_umap(run,x,labels,meta,k,seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt, umap
    xn=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    pc=PCA(n_components=min(30,xn.shape[1],len(xn)-1),random_state=seed).fit_transform(xn)
    p2=PCA(n_components=2,random_state=seed).fit_transform(xn)
    u=umap.UMAP(n_neighbors=30,min_dist=.2,metric="cosine",random_state=seed).fit_transform(pc/np.maximum(np.linalg.norm(pc,axis=1,keepdims=True),1e-12))
    fig,ax=plt.subplots(2,4,figsize=(19,9)); ax=ax.ravel(); colors=plt.get_cmap("tab20",k)
    for a,co,title in [(ax[0],p2,"PCA (encoder)"),(ax[1],u,"UMAP (PC30, cosine)")]:
        for c in range(k):
            q=labels==c; a.scatter(co[q,0],co[q,1],s=7,alpha=.72,color=colors(c),label=f"C{c} n={q.sum()}")
        a.set_title(title); a.set_xlabel("1"); a.set_ylabel("2")
        if k<=10: a.legend(fontsize=6,markerscale=1.4,frameon=False)
    feats=[("volume_um3","log10 volume",True),("surface_area_um2","log10 area",True),
           ("bbox_aspect_ratio","bbox elongation",False),("sphericity","sphericity",False),
           ("ellipsoid_volume_agreement","ellipsoid agreement",False)]
    for a,(name,title,log) in zip(ax[2:],feats):
        z=vals(meta,name)
        if log: z=np.log10(np.maximum(z,1e-12))
        ok=np.isfinite(z)
        if ok.any():
            lo,hi=np.quantile(z[ok],[.02,.98]); sc=a.scatter(u[ok,0],u[ok,1],c=np.clip(z[ok],lo,hi),s=7,cmap="viridis")
            fig.colorbar(sc,ax=a,fraction=.046,pad=.03)
        a.set_title(title); a.set_xlabel("UMAP-1"); a.set_ylabel("UMAP-2")
    ax[-1].axis("off"); fig.suptitle(f"Encoder embeddings, K={k}: clusters and morphology"); fig.tight_layout()
    fig.savefig(run/f"pca_umap_morphology_k{k:02d}.png",dpi=170); plt.close(fig)
    np.savetxt(run/f"umap_k{k:02d}.csv",u,delimiter=",",header="umap1,umap2",comments="")
    return float(PCA(n_components=min(30,xn.shape[1]),random_state=seed).fit(xn).explained_variance_ratio_.sum())

def plot_3d(run,group,x,labels,centers,k,seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng=np.random.default_rng(seed); fig=plt.figure(figsize=(2.5*k,6)); picks=[]
    for c in range(k):
        members=np.flatnonzero(labels==c)
        if not len(members): continue
        near=int(members[np.argmin(np.linalg.norm(x[members]-centers[c],axis=1))]); rand=int(rng.choice(members))
        for r,(idx,kind) in enumerate(((near,"center-nearest"),(rand,"random"))):
            ax=fig.add_subplot(2,k,r*k+c+1,projection="3d")
            p=np.asarray(group["xyz"][idx]); at=np.asarray(group["attrs"][idx]); surf=at[:,3]>.5
            ax.scatter(p[:,2],p[:,1],p[:,0],c=np.where(surf,"#d95f02","#1b9e77"),s=1,alpha=.65)
            ax.set_title(f"C{c} {kind}\nidx {idx}",fontsize=7); ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([])
            ax.set_box_aspect((1,1,1)); picks.append({"cluster":c,"kind":kind,"cache_index":idx})
    fig.tight_layout(); fig.savefig(run/f"representative_pointclouds_k{k:02d}.png",dpi=160); plt.close(fig)
    with (run/f"representatives_k{k:02d}.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["cluster","kind","cache_index"]);w.writeheader();w.writerows(picks)

def evaluate(mode,k,run,model,dec,base,group,train_idx,cfg,meta,device):
    allidx=np.arange(len(base)); x=extract(model,loader(base,allidx,cfg["train"]["batch_size"],cfg["train"]["num_workers"],device),device,cfg["train"]["amp"])
    gtrain=extract(model,loader(base,train_idx,cfg["train"]["batch_size"],cfg["train"]["num_workers"],device),device,cfg["train"]["amp"])
    ids=np.asarray(group["instance_id"][:]); neurons=np.asarray(group["neuron_id"][:])
    np.savez_compressed(run/"embeddings_encoder.npz",embedding=x.astype("float32"),instance_id=ids,neuron_id=neurons,cache_index=allidx)
    ks=cfg["objective"]["cluster"]["k_values"] if k is None else [k]; out={}
    for kk in ks:
        if dec:
            q=dec(torch.as_tensor(x,device=device)).detach().cpu().numpy(); labels=q.argmax(1); centers=dec.centers.detach().cpu().numpy(); kind="DEC Student-t probabilities"
        else:
            km=KMeans(n_clusters=kk,n_init=cfg["objective"]["cluster"]["kmeans_n_init"],random_state=cfg["seed"]).fit(gtrain)
            labels=km.predict(x); centers=km.cluster_centers_; q=np.eye(kk,dtype="float32")[labels];kind="KMeans hard one-hot"
        m=analyze(run,x,gtrain,labels,q,centers,kk,kind,meta,ids,neurons,cfg["seed"])
        if kk==cfg["evaluation"]["representative_k"]:
            m["pca_top30"]=plot_umap(run,x,labels,meta,kk,cfg["seed"])
            plot_3d(run,group,x,labels,centers,kk,cfg["seed"])
            sample_n=min(cfg["evaluation"]["stability_samples"],len(base))
            si=np.random.default_rng(cfg["seed"]).choice(len(base),sample_n,replace=False)
            m.update(sampling_stability(model,base,si,cfg,device))
        out[str(kk)]=m; print(json.dumps({"evaluation":mode,"k":kk,**m}),flush=True)
    (run/"evaluation_summary.json").write_text(json.dumps(out,indent=2))
    return [{"mode":mode,"k":int(kk),**m,**stats(x)} for kk,m in out.items()]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True);p.add_argument("--cache",required=True);p.add_argument("--init-checkpoint",required=True);p.add_argument("--output",required=True)
    p.add_argument("--epochs",type=int);p.add_argument("--k-values",nargs="+",type=int);p.add_argument("--methods",nargs="+",choices=["A","B","C","D"],default=["A","B","C","D"])
    p.add_argument("--max-train-steps",type=int,default=0);p.add_argument("--skip-analysis",action="store_true");p.add_argument("--device",default="cuda:0");p.add_argument("--seed",type=int)
    a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text())
    if a.seed is not None:cfg["seed"]=a.seed
    if a.k_values:cfg["objective"]["cluster"]["k_values"]=a.k_values
    seed_everything(cfg["seed"]);device=torch.device(a.device if not a.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    if device.type=="cuda":torch.cuda.set_device(device)
    init=torch.load(a.init_checkpoint,map_location="cpu",weights_only=False)
    mcfg=model_config_from_yaml(init["config"]);state=init["model"]
    base=PointCloudCache(a.cache,feature_names=cfg["data"]["feature_names"]);group=zarr.open_group(a.cache,mode="r")
    neurons=np.asarray(group["neuron_id"][:]);split=split_by_neuron(neurons,cfg["split_seed"],cfg["data"]["val_fraction"],cfg["data"]["test_fraction"])
    tr,va=split["train"],split["val"]
    if not len(tr) or not len(va) or set(neurons[tr])&set(neurons[va]):raise ValueError("Invalid neuron-level train/validation split")
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);meta=metadata_for(len(base),a.init_checkpoint)
    (out/"experiment_config.json").write_text(json.dumps({"config":cfg,"cache":a.cache,"init_checkpoint":a.init_checkpoint,
        "n_total":len(base),"n_neurons":len(np.unique(neurons)),"split_sizes":{k:len(v) for k,v in split.items()},
        "split_mode":"neuron-level; only train split contributes clustering loss","pretrained_epoch":init.get("epoch")},indent=2))
    print(json.dumps({"device":str(device),"n":len(base),"neurons":len(np.unique(neurons)),"splits":{k:len(v) for k,v in split.items()},
        "pretrained_epoch":init.get("epoch"),"methods":a.methods,"k":cfg["objective"]["cluster"]["k_values"]}),flush=True)
    summary=[]
    for mode in a.methods:
        ks=[None] if mode in ("A","B") else cfg["objective"]["cluster"]["k_values"]
        for k in ks:
            name={"A":"A_mae","B":"B_vicreg","C":"C_dec","D":"D_vicreg_dec"}[mode]+(f"_k{k:02d}" if k else "")
            rd=out/name;rd.mkdir(parents=True,exist_ok=True);print(json.dumps({"start":name}),flush=True)
            model,vh,dec=train_one(mode,k,cfg,mcfg,base,tr,va,state,a,device,rd)
            if not a.skip_analysis:
                summary.extend(evaluate(mode,k,rd,model,dec,base,group,tr,cfg,meta,device))
                (out/"comparison_summary.json").write_text(json.dumps(summary,indent=2))
            del model,vh,dec;gc.collect()
            if device.type=="cuda":torch.cuda.empty_cache()
    if not a.skip_analysis and summary:
        keys=sorted({key for row in summary for key in row})
        with (out/"comparison_summary.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(summary)
    print(json.dumps({"finished":not a.skip_analysis,"output":str(out)}),flush=True)

if __name__=="__main__":main()
