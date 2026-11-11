"""Run disjoint feature shards with a four-thread CPU budget, then merge and verify."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from common import *
def main(out,workers):
    out=Path(out);assert workers in [1,2]
    assert (out/'morphology.csv').exists()
    for stage in ['fusion50','pointmae100','nna0','nna20','em_base']:
        assert (out/'extraction'/stage/'complete.json').exists(),stage
    registry=json.loads((out/'feature_registry.json').read_text())
    # Resolve immutable weight identity once; all records retain the original checkpoint paths.
    cp_hashes={}
    for q in registry:
        p=q['checkpoint']
        if p not in cp_hashes:cp_hashes[p]=sha(p)
        q['checkpoint_sha256']=cp_hashes[p]
        q['mask_protocol']='training cohort mapping: 9870 masks filled per XY slice, holes<200px, plus808 unchanged originals; see instances.csv'
    save(out/'feature_registry.json',registry)
    jobs=[];started=time.time()
    for w in range(workers):
        names=[q['name'] for q in registry[w::workers]]
        phase=f'final_part{w}'
        cmd=[sys.executable,'-u',str(Path(__file__).parent/'evaluate.py'),'--output',str(out),'--phase',phase,'--features',','.join(names)]
        env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        log=(out/(phase+'.log')).open('a')
        p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((p,log,phase))
    for p,log,phase in jobs:
        code=p.wait();log.close()
        if code:raise RuntimeError(f'{phase} failed with status {code}; completed disjoint artifacts preserved for resume')
    folder=out/'final';folder.mkdir(exist_ok=True);(folder/'predictions').mkdir(exist_ok=True)
    seen={};tables=[]
    for p,log,phase in jobs:
        source=out/phase
        schema=json.loads((source/'folds.json').read_text())
        if (folder/'folds.json').exists():assert json.loads((folder/'folds.json').read_text())==schema
        else:save(folder/'folds.json',schema);save(folder/'task_coverage.json',json.loads((source/'task_coverage.json').read_text()))
        scores=pd.read_csv(source/'scores_long.csv')
        for _,r in scores.iterrows():
            identity=(r.feature,r.task);filename=f'{r.feature}__{r.task}'
            result=json.loads((source/(filename+'.json')).read_text())
            if identity in seen:
                assert result['identity']==seen[identity]['identity']
                for k,v in result['metrics'].items():
                    if isinstance(v,float):assert abs(v-seen[identity]['metrics'][k])<1e-8
                continue
            seen[identity]=result
            target=folder/'predictions'/(filename+'.npz')
            if not target.exists():os.link(source/'predictions'/(filename+'.npz'),target)
            result['source_result_file']=str(source/(filename+'.json'))
            result['predictions']=str(target);save(folder/(filename+'.json'),result)
        tables.append(scores)
    d=pd.concat(tables,ignore_index=True).drop_duplicates(['feature','task'])
    assert len(d)==(len(registry)+2)*len(schema['tasks'])
    d.to_csv(folder/'scores_long.csv',index=False)
    wide=[]
    for name,g in d.groupby('feature',sort=False):
        r=dict(feature=name)
        for _,q in g.iterrows():
            metric='body_balanced_accuracy' if q.task=='celltype_six' else ('balanced_accuracy' if q.kind=='classification' else 'r2')
            r[q.task+'__'+metric]=q[metric]
        wide.append(r)
    pd.DataFrame(wide).to_csv(folder/'performance.csv',index=False)
    save(folder/'complete.json',dict(features=len(registry)+2,results=len(d),elapsed_seconds=time.time()-started,workers=workers,
         threads_per_worker=2,protocol_sha256=sha(out/'protocol.json'),evaluation_code_sha256=sha(Path(__file__).parent/'evaluate.py')))
    for script in ['verify_results.py','diagnose.py','scale_control.py']:
        subprocess.run([sys.executable,str(Path(__file__).parent/script),'--output',str(out),'--phase','final'],check=True)
    save(out/'evaluation_complete.json',dict(completed_unix=time.time(),elapsed_seconds=time.time()-started))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args();main(a.output,a.workers)
