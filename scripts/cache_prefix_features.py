"""Cache frozen release-model embeddings, with row IDs and weight fingerprints."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'SketchScape'))
from model import SBIRModel

DATA=ROOT/'data/fscoco'
PREFIX=ROOT/'data/time_prefix_v1'
CACHE=ROOT/'outputs/adapter/cache'
VIEWS=('photo','100','25','50','75')
WEIGHTS={'normal':'6fd930276c8addf506b67a1a897fb671ec7c16e41fccf90b8b177223df1f973a',
         'unseen':'5ebbbdb0c134a2cdb24d70118012e978ba3443a85c21b41390486c9e930d57fe'}

def read_view(user, key, view, transform):
    if view=='photo':
        path=DATA/'images'/user/(key+'.jpg')
    elif view=='100':
        path=DATA/'raster_sketches'/user/(key+'.png')
    else:
        path=PREFIX/view/user/(key+'.png')
    with Image.open(path) as image:
        return transform(image.convert('RGB' if view=='photo' else 'L'))

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('protocol',choices=WEIGHTS)
    parser.add_argument('--device',default='mps')
    parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args()
    torch.set_num_threads(2)
    ckpt=ROOT/'models'/f'fscoco_{args.protocol}.pth'
    with ckpt.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
    assert sha==WEIGHTS[args.protocol]
    model=SBIRModel.load_module(str(ckpt),strict=True,cuda=False).requires_grad_(False).to(args.device).eval()
    pics=sorted((DATA/'images').glob('*/*.jpg'),key=lambda p:(int(p.parent.name),p.stem))
    keys=[(p.parent.name,p.stem) for p in pics]
    assert len(set(keys))==len(keys)==10000
    out=CACHE/args.protocol
    out.mkdir(parents=True,exist_ok=True)
    meta=dict(protocol=args.protocol,weight_sha256=sha,source_commit='bbfc2499d817fc64dfafcf02d84bf98e7a44ad24',dtype='float32',
              prefix_rule='inclusive elapsed time v1',ids=keys,views=VIEWS,batch=args.batch_size,device=args.device)
    manifest=out/'manifest.json'
    serialized=json.dumps(meta)
    if manifest.exists() and manifest.read_text()!=serialized:
        raise ValueError('Cache provenance differs')
    manifest.write_text(serialized)
    start=time.perf_counter()
    for view in VIEWS:
        if view not in ('photo','100') and not (PREFIX/'manifest.csv').exists():
            raise RuntimeError('Generate and verify time prefixes first')
        transform=model.create_image_transforms() if view=='photo' else model.create_sketch_transforms()
        for lower in range(0,len(keys),1000):
            dest=out/f'{view}_{lower:05d}.npy'
            if dest.exists():
                continue
            result=[]
            for begin in range(lower,lower+1000,args.batch_size):
                samples=keys[begin:min(begin+args.batch_size,lower+1000)]
                x=torch.stack([read_view(user,k,view,transform) for user,k in samples])
                y=model(x.to(args.device)).cpu()
                assert y.shape==(len(x),512) and torch.isfinite(y).all()
                assert torch.allclose(y.norm(dim=1),torch.ones(len(x)),atol=1e-4)
                result.append(y.numpy())
            tmp=dest.with_suffix('.tmp')
            with tmp.open('wb') as f: np.save(f,np.concatenate(result))
            os.replace(tmp,dest)
            print(args.protocol,view,lower+1000,'elapsed_sec',round(time.perf_counter()-start,1),flush=True)

if __name__=='__main__':
    main()
