import json
import os
from pathlib import Path
import signal
import subprocess
import time

import torch

run = Path('/path/to/project_data/EM/point-MAE/runs/voxel_mae_native_v1_20261014')
pid = 1278086
proc = Path('/proc') / str(pid)
args = proc.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
assert str(run) in args and 'mito_voxel_mae.train' in args
env = dict(item.split('=', 1) for item in proc.joinpath('environ').read_bytes().decode().strip('\0').split('\0'))
cwd = os.readlink(proc / 'cwd')
args[args.index('--epochs') + 1] = '10'
args += ['--resume', str(run / 'last.pt')]
(run / 'requested_epochs.json').write_text(json.dumps({'epochs': 10, 'resume_after_epoch': 6, 'status': 'waiting_for_epoch_checkpoint'}))
print('Waiting for completed epoch 6 checkpoint, then resume through epoch 10.', flush=True)
while True:
    if not proc.exists():
        raise RuntimeError('Training exited before checkpoint transition')
    metrics = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    if metrics and metrics[-1]['epoch'] >= 5:
        state = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
        if state['epoch'] >= 5:
            break
    time.sleep(5)
assert state['epoch'] < 10
os.kill(pid, signal.SIGSTOP)
os.kill(pid, signal.SIGTERM)
os.kill(pid, signal.SIGCONT)
for _ in range(60):
    if not proc.exists() or proc.joinpath('stat').read_text().split()[2] == 'Z':
        break
    time.sleep(1)
else:
    raise RuntimeError('Original training did not terminate; refusing duplicate launch')
with (run / 'train.log').open('a') as log:
    log.write('\n' + json.dumps({'resume_transition': True, 'completed_epochs': state['epoch'] + 1, 'new_total_epochs': 10}) + '\n')
    log.flush()
    child = subprocess.Popen(args, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(run / 'train.pid').write_text(str(child.pid) + '\n')
(run / 'requested_epochs.json').write_text(json.dumps({'epochs': 10, 'resumed_after_epoch': state['epoch'] + 1, 'status': 'resumed', 'pid': child.pid}))
print(json.dumps({'resumed_pid': child.pid, 'total_epochs': 10}), flush=True)
