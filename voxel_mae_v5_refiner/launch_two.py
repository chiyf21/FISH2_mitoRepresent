"""Two isolated three-GPU experiments, pressure probes, immutable source snapshot."""
import argparse,concurrent.futures,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--root',required=True)
    a=p.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
    lock=root/'launcher.pid'
    if lock.exists():
        try:os.kill(int(lock.read_text()),0)
        except ProcessLookupError:pass
        else:raise RuntimeError('Launcher already running')
    lock.write_text(str(os.getpid()));source=Path(__file__).resolve().parents[1]
    snapshot=root/'code_snapshot'
    if snapshot.exists():raise FileExistsError(snapshot)
    for package in ['mito_voxel_mae_v2','mito_voxel_mae_v3','mito_voxel_mae_v4','mito_voxel_mae_v5']:
        shutil.copytree(source/package,snapshot/package,ignore=shutil.ignore_patterns('__pycache__'))
    def event(out,**row):
        row['time_unix']=time.time()
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(dict(experiment=out.name,**row)),flush=True)
    def experiment(name,objective,gpus,port):
        out=root/name;out.mkdir();env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpus,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(snapshot),NCCL_DEBUG='WARN')
        def command(module,extra):
            return [sys.executable,'-u','-m','torch.distributed.run','--nproc_per_node=3',f'--master_port={port}','--module',module]+extra
        def probe(kind,batch,tile,micro):
            dest=out/f'pressure_{kind}_{batch}.json';log=out/f'pressure_{kind}_{batch}.log'
            cmd=command('mito_voxel_mae_v5.pressure',['--cache',a.cache,'--out',str(dest),'--objective',objective,
                '--kind',kind,'--batch',str(batch),'--tile-batch',str(tile),'--slice-microbatch',str(micro),'--memory-fraction','.72'])
            event(out,stage='pressure_start',kind=kind,batch=batch,tile_batch=tile,slice_microbatch=micro)
            with log.open('w') as f:
                proc=subprocess.Popen(cmd,cwd=snapshot,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                (out/'active_process.pid').write_text(str(proc.pid))
                try:code=proc.wait(timeout=900)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGTERM);proc.wait();raise RuntimeError(f'Pressure timeout: {log}')
            if code==0 and dest.exists():
                result=json.loads(dest.read_text());event(out,stage='pressure_pass',**result);return True
            content=log.read_text()
            if 'out of memory' in content.lower() or 'OutOfMemoryError' in content:
                event(out,stage='pressure_oom',kind=kind,batch=batch);return False
            raise RuntimeError(f'Pressure failed unexpectedly: {log}')
        tile_good=0
        for tiles in [4,8,16,32,64]:
            if not probe('tiles',tiles,tiles,tiles*22):break
            tile_good=tiles
        if not tile_good:raise RuntimeError('Smallest tile pressure probe failed')
        tile=max(1,int(tile_good*.75));micro=tile*22
        batch_good=0
        for batch in [4,8,16,32,64,128,256,512]:
            if not probe('instances',batch,tile,micro):break
            batch_good=batch
        if not batch_good:raise RuntimeError('Smallest instance batch probe failed')
        # Highest passing representative batch; workload buckets shrink longer/wider inputs.
        max_batch=batch_good;budget=max_batch*16
        tuning=dict(gpus=gpus,world_size=3,tile_batch=tile,slice_microbatch=micro,max_batch_per_gpu=max_batch,
                    budget=budget,memory_fraction=.72,largest_passing_tile_probe=tile_good,
                    caveat='Measured median-instance maximum, bucket-dependent; not a universal maximum for arbitrary shapes')
        (out/'tuning.json').write_text(json.dumps(tuning,indent=2));event(out,stage='tuned',**tuning)
        while not (Path(a.cache)/'audit.json').exists():time.sleep(10)
        audit=json.loads((Path(a.cache)/'audit.json').read_text())
        assert audit['instances']==19947 and audit['errors']==0 and audit['mask_and_foreground_em_consistency']
        for attempt in range(8):
            args=['--cache',a.cache,'--out',str(out),'--objective',objective,'--epochs','15','--queries','1000',
                  '--views','4','--workers','2','--effective-batch','512','--save-every','5','--seed','42',
                  '--memory-fraction','.72','--budget',str(budget),'--max-batch',str(max_batch),
                  '--tile-batch',str(tile),'--slice-microbatch',str(micro)]
            if objective=='lowfreq':args+=['--lf-with-occ']
            checkpoints=[out/'last.pt',out/'recovery.pt'];available=[x for x in checkpoints if x.exists()]
            if available:args+=['--resume',str(max(available,key=lambda x:x.stat().st_mtime))]
            cmd=command('mito_voxel_mae_v5.train',args)
            (out/f'command_attempt{attempt}.json').write_text(json.dumps(dict(command=cmd,gpus=gpus),indent=2))
            event(out,stage='training_start',attempt=attempt,max_batch=max_batch,budget=budget,tile_batch=tile)
            log=out/f'train_attempt{attempt}.log'
            with log.open('w') as f:
                proc=subprocess.Popen(cmd,cwd=snapshot,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                (out/'active_process.pid').write_text(str(proc.pid));code=proc.wait()
            if code==0:event(out,stage='completed');return
            content=log.read_text()
            if 'out of memory' not in content.lower():raise RuntimeError(f'Training failed: {log}')
            budget=max(16,int(budget*.65));max_batch=max(1,int(max_batch*.65));tile=max(1,tile//2);micro=min(micro,tile*22)
            event(out,stage='oom_backoff',attempt=attempt,budget=budget,max_batch=max_batch,tile_batch=tile)
        raise RuntimeError(f'Exhausted OOM retries: {out}')
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        jobs=[pool.submit(experiment,'expA_occ_intensity','voxel','0,1,2',29551),
              pool.submit(experiment,'expB_occ_lowfreq','lowfreq','3,4,5',29552)]
        failures=[]
        for job in concurrent.futures.as_completed(jobs):
            try:job.result()
            except Exception as exc:failures.append(str(exc));print(str(exc),flush=True)
        if failures:
            (root/'failed.json').write_text(json.dumps(failures));raise RuntimeError(str(failures))
    (root/'completed.json').write_text(json.dumps(dict(completed=True,time_unix=time.time())))

if __name__=='__main__':main()
