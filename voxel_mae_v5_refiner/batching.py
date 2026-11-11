import json,math,random
from pathlib import Path
import numpy as np
from .data import TILE_THRESHOLD

class BucketBatches:
    """Same number of steps across ranks; repeated tail fillers have zero weight."""
    def __init__(self,dataset,rank=0,world=1,budget=64,max_batch=8,seed=42,shuffle=True):
        self.dataset=dataset;self.rank=rank;self.world=world;self.budget=budget;self.max_batch=max_batch;self.seed=seed;self.shuffle=shuffle;self.epoch=0
        self.buckets={};self.bucket_shapes={}
        self._cached_epoch=None;self._cached_batches=None
        for i,r in enumerate(dataset.records):
            z,y,x=r['shape_zyx']
            # Primary Z bins; spatial workload bins prevent one long/wide instance
            # from being batched with many cheap instances. No instance is discarded.
            tiles=max(1,math.ceil(y/256)*math.ceil(x/256))
            shape=np.array([z,tiles,1])
            for view in range(dataset.views):
                # Elastic and affine augmentation have very different CPU costs.
                # Homogeneous augmentation buckets retain every original view.
                kind=view%4 if dataset.augmentations else 0
                key=(int(math.ceil(z/8)),int(math.ceil(math.log2(tiles+1))),kind)
                self.bucket_shapes[key]=np.maximum(self.bucket_shapes.get(key,shape),shape)
                self.buckets.setdefault(key,[]).append(i+view*len(dataset.records))
    def batches(self):
        if self._cached_epoch==self.epoch:return self._cached_batches
        rng=random.Random(self.seed+self.epoch);batches=[]
        for key,indices in sorted(self.buckets.items()):
            ids=list(indices)
            if self.shuffle:rng.shuffle(ids)
            upper=self.bucket_shapes[key];volume=int(np.prod(upper))
            local=1 if volume>TILE_THRESHOLD else min(self.max_batch,max(1,self.budget//volume))
            for start in range(0,len(ids),local*self.world):
                chunk=[(i,1.) for i in ids[start:start+local*self.world]]
                actual=math.ceil(len(chunk)/self.world)
                while len(chunk)<actual*self.world:chunk.append((chunk[0][0],0.))
                # Equal instance counts alone do not balance full-volume CPU
                # processing. Greedily distribute larger contexts across ranks.
                def work(item):
                    if not item[1]:return 0
                    r=self.dataset.records[item[0]%len(self.dataset.records)]
                    return int(np.prod(r['shape_zyx']))
                parts=[[] for _ in range(self.world)];loads=[0]*self.world
                for item in sorted(chunk,key=work,reverse=True):
                    target=min((r for r in range(self.world) if len(parts[r])<actual),key=lambda r:loads[r])
                    parts[target].append(item);loads[target]+=work(item)
                batches.append(parts[self.rank])
        if self.shuffle:rng.shuffle(batches)
        self._cached_epoch=self.epoch;self._cached_batches=batches
        return batches
    def __iter__(self):return iter(self.batches())
    def __len__(self):return len(self.batches())
