"""Same physical affine/elastic transforms, with separable cubic field upsampling."""
from functools import lru_cache
import numpy as np
from scipy import ndimage

@lru_cache(maxsize=256)
def _cubic_weights(length):
    # A 4-point cubic spline is a linear operator. Apply its 1D interpolation
    # matrices along each axis instead of evaluating 64 coefficients per voxel.
    eye=np.eye(4,dtype=np.float64)
    return np.stack([ndimage.zoom(eye[:,i],length/4,order=3) for i in range(4)],axis=1)

def resize_field(coarse,shape):
    z,y,x=map(int,shape)
    tmp=np.einsum('zi,ijk->zjk',_cubic_weights(z),coarse)
    tmp=np.einsum('yj,zjk->zyk',_cubic_weights(y),tmp)
    return np.einsum('xk,zyk->zyx',_cubic_weights(x),tmp).astype(np.float32)

def augment(em,mask,valid,spacing,rng,kind):
    if kind==0:return em,mask,valid
    shape=np.array(em.shape);center=(shape-1)/2
    angle=rng.uniform(-10,10)*np.pi/180 if kind==2 else rng.uniform(-3,3)*np.pi/180
    cs,sn=np.cos(angle),np.sin(angle)
    rot=np.array([[1,0,0],[0,cs,-sn],[0,sn,cs]])
    scale=np.diag(rng.uniform(.97,1.03,3)) if kind==1 else np.eye(3)
    shear=np.eye(3)
    if kind==1:shear[1,2]=rng.uniform(-.025,.025)
    inv=np.linalg.inv(rot@shear@scale)
    matrix=np.diag(1/spacing)@inv@np.diag(spacing)
    shift=rng.uniform(-.015,.015,3)*shape;offset=center-matrix@(center+shift)
    if kind==3:
        coords=np.indices(tuple(shape),dtype=np.float32)
        source=np.einsum('ij,j...->i...',matrix,coords)+offset[:,None,None,None]
        for axis in range(3):
            coarse=rng.normal(size=(4,4,4)).astype(np.float32)
            source[axis]+=resize_field(coarse,shape)*.008*shape[axis]
        fn=lambda a,order:ndimage.map_coordinates(a,source,order=order,mode='constant',cval=0,prefilter=False)
    else:
        fn=lambda a,order:ndimage.affine_transform(a,matrix,offset,output_shape=tuple(shape),order=order,mode='constant',cval=0,prefilter=False)
    m=fn(mask,0)>0;v=fn(valid.astype(np.float32),1)>.999;e=fn(em,1)
    if not m.any() or any(m.take(i,axis=a).any() for a in range(3) for i in (0,-1)) or (m&~v).any():
        return em,mask,valid
    return e,m,v
