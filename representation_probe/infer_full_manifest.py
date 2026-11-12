"""Infer Fusion50 L12 embeddings for the frozen 16,428-instance manifest."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np,torch
from scipy import ndimage as ndi
from common import X,F,CACHE,sha
from extract import Hooks,batch,load
DATASET=Path("/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/neuron_mito_dataset")
SOURCE_MANIFEST=DATASET/"handcraft_features/paper_original_v1/instance_manifest.csv"
OUT=DATASET/"embeddings/fusion50_all16428_20261112"
POINT_SOURCE=F/"point_cache/pointclouds.npy"
OLD_COHORT=X/"cohort.json"
CHECKPOINT=X/"checkpoints/refine_epoch050.pt"
OLD_FUSED=CACHE/"fused_12.npy"
NPOINTS,BATCH,CHUNK,SEED=2048,8,64,20261105

def fill_small_slice_holes(mask,limit=200):
    out=np.asarray(mask,dtype=bool).copy()
    for z,plane in enumerate(out):
        holes=ndi.binary_fill_holes(plane)&~plane
        if not holes.any():continue
        labels,_=ndi.label(holes,structure=ndi.generate_binary_structure(2,1))
        sizes=np.bincount(labels.ravel());ids=np.flatnonzero((sizes>0)&(sizes<limit));ids=ids[ids>0]
        if len(ids):out[z]|=np.isin(labels,ids)
    return out

def save_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding="utf-8");tmp.replace(path)

def key(source,body,instance):return (str(source),str(body),int(instance))

def main():
    global SOURCE_MANIFEST, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--sample-index-offset", type=int, default=0)
    args = parser.parse_args()
    frozen = args.manifest == SOURCE_MANIFEST
    SOURCE_MANIFEST, OUT = args.manifest, args.output
    if args.sample_index_offset < 0:
        parser.error("--sample-index-offset must be nonnegative")
    if OUT.exists():raise FileExistsError(f"Refusing to overwrite existing run: {OUT}")
    OUT.mkdir(parents=True);chunks=OUT/"chunks";masks_out=OUT/"filled_masks";chunks.mkdir();masks_out.mkdir()
    with SOURCE_MANIFEST.open(newline="",encoding="utf-8-sig") as f:source_rows=list(csv.DictReader(f))
    if not source_rows or len({row["record_id"] for row in source_rows}) != len(source_rows):
        raise ValueError("Input manifest must contain unique records")
    if frozen: assert len(source_rows)==16428
    manifest_hash=sha(SOURCE_MANIFEST);old_rows=json.loads(OLD_COHORT.read_text())
    old_by_key={key(r["key"][0],r["key"][1],r["key"][2]):r for r in old_rows};assert len(old_by_key)==10678
    old_points=np.load(POINT_SOURCE,mmap_mode="r",allow_pickle=False);assert old_points.shape==(10678,2,2048,3)
    point_path=OUT/"pointclouds.npy"
    points=np.lib.format.open_memmap(point_path,mode="w+",dtype=np.float32,shape=(len(source_rows),2,NPOINTS,3))
    inference_rows=[];old_cache_rows=np.full(len(source_rows),-1,dtype=np.int64);old_matches=0
    for index,row in enumerate(source_rows):
        identity=key(row["source_run_id"],row["body_id"],row["global_instance"]);old=old_by_key.get(identity)
        if old is not None:
            old_matches+=1;points[index]=old_points[int(old["point_row"])]
            old_cache_rows[index]=int(old["cohort_index"])
            inference_rows.append({"key":list(identity),"cohort_index":int(old["cohort_index"]),"point_row":index,
                "normalization_radius_voxels":float(old["normalization_radius_voxels"]),
                "em_path":old["em_path"],"mask_path":old["mask_path"]})
        else:
            em_path,original_mask_path=DATASET/row["em_path"],DATASET/row["mask_path"]
            assert em_path.is_file() and original_mask_path.is_file(),row["record_id"]
            mask=np.asarray(np.load(original_mask_path,mmap_mode="r",allow_pickle=False)>0)
            filled=fill_small_slice_holes(mask);filled_path=masks_out/f"{row['record_id']}.npy"
            np.save(filled_path,filled.astype(np.uint8),allow_pickle=False)
            surface=filled&~ndi.binary_erosion(filled,structure=ndi.generate_binary_structure(3,1),border_value=0)
            zyx=np.argwhere(surface)
            if not len(zyx):raise ValueError(f"Empty 6-neighbor surface: {row['record_id']}")
            xyz=zyx[:,::-1].astype(np.float32);rng=np.random.default_rng(SEED+(args.sample_index_offset+index)*2);radius=0.0
            for view in range(2):
                chosen=rng.choice(len(xyz),NPOINTS,replace=len(xyz)<NPOINTS)
                sample=xyz[chosen].copy();sample-=sample.mean(axis=0,keepdims=True)
                radius=max(float(np.linalg.norm(sample,axis=1).max()),1e-8);points[index,view]=sample/radius
            points.flush()
            inference_rows.append({"key":list(identity),"cohort_index":index,"point_row":index,
                "normalization_radius_voxels":radius,"em_path":str(em_path),"mask_path":str(filled_path)})
        if (index+1)%100==0:
            save_json(OUT/"preprocess_progress.json",{"done":index+1,"total":len(source_rows),"old_matches":old_matches})
            print(json.dumps({"phase":"preprocess","done":index+1,"total":len(source_rows)}),flush=True)
    points.flush();del points,old_points
    if frozen: assert old_matches==10678 and len(inference_rows)==16428
    with (OUT/"instance_map.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["embedding_row","record_id","source_run_id","body_id","global_instance","cell_type","model_cohort_index","old_embedding_row"])
        for i,row in enumerate(source_rows):
            w.writerow([i,row["record_id"],row["source_run_id"],row["body_id"],row["global_instance"],row["cell_type"],
                inference_rows[i]["cohort_index"],"" if old_cache_rows[i]<0 else int(old_cache_rows[i])])
    save_json(OUT/"input_manifest.json",{"instances":len(inference_rows),"old_cache_matches":old_matches,
        "new_instances":len(inference_rows)-old_matches,"source_manifest":str(SOURCE_MANIFEST),
        "source_manifest_sha256":manifest_hash,"pointclouds":str(point_path),
        "sample_index_offset":args.sample_index_offset,"point_sampling_new":"uniform 6-neighbor voxel surface; 2 deterministic views; replacement only below 2048 points",
        "mask_fill_new":"per native XY slice; 4-connected enclosed holes with area <200 px filled",
        "model_checkpoint":str(CHECKPOINT),"model_checkpoint_sha256":sha(CHECKPOINT),
        "old_pointclouds_reused_from":str(POINT_SOURCE),"inference_code_sha256":sha(Path(__file__))})
    pointdata=np.load(point_path,mmap_mode="r",allow_pickle=False);device=torch.device("cuda:0")
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(20261111);torch.cuda.manual_seed_all(20261111)
    train_module,model,grouping,checkpoint=load("fusion50",device);hooks=Hooks(model)
    embedding=np.lib.format.open_memmap(OUT/"fused_12.npy",mode="w+",dtype=np.float32,shape=(len(inference_rows),384))
    started=time.time()
    for start in range(0,len(inference_rows),CHUNK):
        end=min(start+CHUNK,len(inference_rows));chunk_path=chunks/f"{start:06d}.npy"
        if chunk_path.exists():
            cached=np.load(chunk_path,mmap_mode="r",allow_pickle=False)
            assert cached.shape==(end-start,384) and np.isfinite(cached).all();embedding[start:end]=cached;continue
        values=[]
        with torch.inference_mode():
            for pos in range(start,end,BATCH):
                stop=min(pos+BATCH,end)
                _,result,_=batch(train_module,model,grouping,inference_rows[pos:stop],pointdata,device,"fusion50",hooks)
                value=result[2][12][:,0].float().cpu().numpy()
                assert value.shape==(stop-pos,384) and np.isfinite(value).all();values.append(value)
        arr=np.concatenate(values);tmp=chunk_path.with_suffix(".tmp.npy");np.save(tmp,arr,allow_pickle=False);tmp.replace(chunk_path)
        embedding[start:end]=arr;embedding.flush()
        save_json(OUT/"progress.json",{"done":end,"total":len(inference_rows),"elapsed_seconds":time.time()-started})
        print(json.dumps({"phase":"inference","done":end,"total":len(inference_rows),"elapsed_seconds":round(time.time()-started,1)}),flush=True)
    embedding.flush();del embedding
    all_embeddings=np.load(OUT/"fused_12.npy",mmap_mode="r",allow_pickle=False);old_cache=np.load(OLD_FUSED,mmap_mode="r",allow_pickle=False)
    assert old_cache.shape==(10678,384)
    deltas=np.asarray([all_embeddings[i]-old_cache[j] for i,j in enumerate(old_cache_rows) if j>=0],dtype=np.float64)
    max_abs=float(np.max(np.abs(deltas))) if len(deltas) else 0.0
    relative_l2=float(np.linalg.norm(deltas)/max(np.linalg.norm(old_cache),1e-12)) if len(deltas) else 0.0
    assert max_abs<1e-4 and relative_l2<1e-6,(max_abs,relative_l2)
    save_json(OUT/"complete.json",{"status":"complete","instances":len(inference_rows),"dimension":384,
        "embedding":str(OUT/"fused_12.npy"),"instance_map":str(OUT/"instance_map.csv"),
        "checkpoint":str(checkpoint),"checkpoint_sha256":sha(checkpoint),"source_manifest_sha256":manifest_hash,
        "old_instances_verified":len(deltas),"old_embedding_max_abs_delta":max_abs,
        "old_embedding_relative_l2_delta":relative_l2,"elapsed_seconds":time.time()-started})
    print(json.dumps(json.loads((OUT/"complete.json").read_text()),ensure_ascii=False),flush=True)

if __name__=="__main__":main()
