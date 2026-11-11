"""Sequential stage export on one explicitly selected free GPU."""
import argparse,os,subprocess,sys,time
from pathlib import Path
from common import save
def main(out,gpu):
    out=Path(out)
    assert (out/'extraction/fusion50/complete.json').exists(),'finish Fusion50 first'
    for stage in ['pointmae100','nna0','nna20','em_base']:
        if (out/'extraction'/stage/'complete.json').exists():continue
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        assert not any(gpu in line for line in processes.splitlines()),'selected GPU currently has a compute process; inspect before proceeding'
        cmd=[sys.executable,'-u',str(Path(__file__).parent/'extract.py'),'--output',str(out),'--stage',stage,'--gpu',gpu]
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
        print('Starting',stage,flush=True)
        with (out/f'extract_{stage}.log').open('a') as log:
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Completed',stage,flush=True)
    save(out/'stages_complete.json',dict(stages=['fusion50','pointmae100','nna0','nna20','em_base'],completed_unix=time.time()))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--gpu',required=True);a=p.parse_args();main(a.output,a.gpu)
