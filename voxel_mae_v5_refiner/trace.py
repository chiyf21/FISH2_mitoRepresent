"""Optional low-volume stage traces and Python stack diagnostics."""
import faulthandler,json,os,time
from pathlib import Path
_pid=None
_stack=None
def trace(stage,**values):
    global _pid,_stack
    root=os.getenv('MITOMAE_TRACE_DIR')
    if not root:return
    pid=os.getpid()
    if _pid!=pid:
        Path(root).mkdir(parents=True,exist_ok=True)
        _stack=open(Path(root)/f'stacks_{pid}.log','a');faulthandler.enable(_stack)
        faulthandler.dump_traceback_later(120,repeat=True,file=_stack);_pid=pid
    row=dict(time=time.time(),pid=pid,rank=os.getenv('RANK'),stage=stage,**values)
    with (Path(root)/f'phases_{pid}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')

def worker_init(worker):
    import torch
    torch.set_num_threads(1)
    torch.multiprocessing.set_sharing_strategy('file_system')
    trace('worker_init',worker=worker)
