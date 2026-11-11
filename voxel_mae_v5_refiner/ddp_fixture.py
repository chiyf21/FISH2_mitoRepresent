import argparse,json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
records=[]
for i in range(3):
    path=root/str(i);path.mkdir(exist_ok=True);shape=(9,25,29);z,y,x=np.indices(shape)
    mask=np.zeros(shape,'uint8');mask[2:7,5:20,5:24]=1
    if i==1:mask[:]=0;mask[4,12,14]=1
    np.save(path/'mask.npy',mask);np.save(path/'em.npy',np.clip(x*3+y*2+z,0,255).astype('uint8'));np.save(path/'valid.npy',np.ones(shape,bool))
    records.append(dict(record_id=str(i),body_id=str(i),relative_dir=str(i),shape_zyx=shape,spacing_zyx_nm=[30,8,8],lo=0,hi=255,split='val' if i==2 else 'train'))
(root/'manifest.json').write_text(json.dumps(dict(records=records,body_splits=dict(train=['0','1'],val=['2'],test=[]))))
