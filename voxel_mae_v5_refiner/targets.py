"""Strict hidden 3^3 target crops and physical-frequency-ordered 3D DCT."""
import itertools
import numpy as np

OFFSETS=np.array(list(itertools.product((-1,0,1),repeat=3)),dtype=np.int64) # ZYX

def low_frequency_basis(spacing=(30.,8.,8.),count=10):
    spacing=np.asarray(spacing,dtype=float)
    modes=np.array(list(itertools.product(range(3),repeat=3)))
    frequency=np.sqrt(((modes/(6*spacing))**2).sum(1)) # cycles / nm
    order=np.argsort(frequency,kind='stable')[:count];modes=modes[order]
    n=np.arange(3)
    dct=np.cos(np.pi*(2*n[None]+1)*np.arange(3)[:,None]/6)
    dct[0]*=1/np.sqrt(3);dct[1:]*=np.sqrt(2/3)
    basis=np.stack([np.einsum('i,j,k->ijk',dct[z],dct[y],dct[x]).reshape(-1) for z,y,x in modes])
    return basis.astype(np.float32),modes.tolist(),frequency[order].tolist()

def safe_query_centres(mask,valid,block_visible,lo,size,origin,stride=(1,4,4)):
    """All 27voxels hidden and valid; >=2 target foreground voxels. No quota."""
    stride=np.asarray(stride);phase=(-np.asarray(origin,dtype=np.int64))%stride
    axes=[]
    for n,p,s in zip(mask.shape,phase,stride):
        axis=np.arange(int(p),n,int(s));axes.append(axis[(axis>=1)&(axis<n-1)])
    if any(len(ax)==0 for ax in axes):return np.empty((0,3),dtype=np.int64)
    shape=tuple(map(len,axes));allowed=np.ones(shape,bool);counts=np.zeros(shape,np.uint8)
    bins=[[np.clip(np.floor(((ax+d+.5-l)/n)*4).astype(int),0,3) for d in (-1,0,1)]
          for ax,l,n in zip(axes,lo,size)]
    # XY stride4 and nativeZ stride1 correspond to32x32x30nm candidates.
    for dz,dy,dx in OFFSETS:
        ix=np.ix_(axes[0]+dz,axes[1]+dy,axes[2]+dx)
        counts+=np.asarray(mask[ix],dtype=np.uint8)
        allowed&=valid[ix]
        allowed&=~block_visible[np.ix_(bins[0][dz+1],bins[1][dy+1],bins[2][dx+1])]
    ids=np.argwhere(allowed&(counts>=2))
    return np.stack([axes[j][ids[:,j]] for j in range(3)],1) if len(ids) else np.empty((0,3),dtype=np.int64)
