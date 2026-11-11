"""Migrate completed owned two-worker caches to four single-thread workers."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
from common import save
def main(out):
    out=Path(out);info=json.loads((out/'final_process.json').read_text());pid=info['pid'];proc=Path('/proc')/str(pid)
    assert not (out/'final/complete.json').exists(),'do not reschedule completed evaluation'
    children=[int(p) for p in (proc/'task'/str(pid)/'children').read_text().split()]
    for p in children+[pid]:
        q=Path('/proc')/str(p)
        if q.exists():
            cmd=(q/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            assert str(out) in cmd and ('evaluate.py' in cmd or 'finish.py' in cmd)
            os.kill(p,signal.SIGTERM)
    time.sleep(.5)
    archive=out/'two_worker_checkpoint';archive.mkdir(exist_ok=False)
    for name in ['final_part0','final_part1','final_part0.log','final_part1.log','final_pipeline.log','final_process.json']:
        p=out/name
        if p.exists():p.rename(archive/name)
    registry=json.loads((out/'feature_registry.json').read_text());assignment={q['name']:i%4 for i,q in enumerate(registry)}
    for w in range(4):
        folder=out/f'final_part{w}';folder.mkdir();(folder/'predictions').mkdir()
        for name in ['folds.json','task_coverage.json']:os.link(archive/'final_part0'/name,folder/name)
    copied=0
    for source in [archive/'final_part0',archive/'final_part1']:
        for p in source.glob('*__*.json'):
            q=json.loads(p.read_text());name=q['feature']
            workers=range(4) if name in ['dummy','simple_morphology'] else [assignment[name]]
            pred=source/'predictions'/(p.stem+'.npz');assert pred.exists()
            for w in workers:
                folder=out/f'final_part{w}';target=folder/p.name
                if not target.exists():
                    os.link(p,target);os.link(pred,folder/'predictions'/pred.name);copied+=1
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    cmd=['nice','-n','10',__import__('sys').executable,'-u',str(Path(__file__).parent/'finish.py'),'--output',str(out),'--workers','4']
    with (out/'final_pipeline.log').open('x') as log:
        p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(out/'final_process.json',dict(pid=p.pid,command=cmd))
    save(out/'parallel_resume.json',dict(from_workers=2,to_workers=4,blas_threads_each=1,completed_result_files_reused=copied,
       archive=str(archive),algorithm_protocol_changed=False,stopped_only_owned_pids=children+[pid]))
    print('reused result files',copied,'new controller',p.pid)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
