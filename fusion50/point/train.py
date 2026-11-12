"""Historical surface Point-MAE model setup and batches for the DDP trainer."""
import contextlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from data import coordinates, load_rows, views

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'shared'))
import point_mae_adapter as adapter


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def save(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def append(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value) + '\n')


class Trainer:
    def __init__(self,cfg):
        self.cfg=cfg;self.out=Path(cfg['output']);self.device=torch.device(cfg['device']);torch.set_num_threads(cfg.get('cpu_threads',2));seed(cfg['seed'])
        self.rows=load_rows(self.out/'data/manifest.tsv');self.points=np.load(self.out/'data/pointclouds.npy',mmap_mode='r');self.split=json.loads((self.out/'split.json').read_text())
        settings=dict(num_group=64,group_size=32,mask_ratio=cfg['mae']['mask_ratio'],trans_dim=384,encoder_dims=384,depth=12,heads=6,decoder_depth=4,decoder_heads=6,drop_path=.1)
        self.mae=adapter.build_model(Path(cfg['point_mae_repo']),SimpleNamespace(**settings),self.device)
        self.step=0;self.amp=cfg.get('amp',True) and self.device.type=='cuda' and torch.cuda.is_bf16_supported()
    def autocast(self):return torch.autocast('cuda',dtype=torch.bfloat16) if self.amp else contextlib.nullcontext()
    def xyz(self,ids,epoch=0,augment=False,jitter=None):
        batch=[]
        for j in ids:
            rng=np.random.default_rng(self.cfg['seed']+int(j)*1009+epoch*100003)
            x=views(self.points[j],self.rows[j],self.cfg,rng,jitter) if augment else coordinates(self.points[j,0],self.rows[j],self.cfg['spacing_xyz_nm'],self.cfg['normalization'])
            batch.append(x)
        return torch.as_tensor(np.stack(batch),device=self.device)
    def batch_ids(self,epoch,batch):
        ids=np.array(self.split['train']);np.random.default_rng(self.cfg['seed']+epoch).shuffle(ids)
        chunks=[ids[j:j+batch] for j in range(0,len(ids),batch)]
        return [a for a in chunks if len(a)>=2][:self.cfg.get('max_batches_per_epoch') or None]
