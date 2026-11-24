"""Run SketchScape stages with strict loading and the author's dataset, transforms and metrics."""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'SketchScape'))
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader, Subset
from data import DatasetAdapter, load_dataset
from model import SBIRModel
from utils import compute_distances, calculate_results, seed_everything

def save(name, obj):
    path = ROOT / 'outputs/metrics' / name
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')

def make_examples(protocol, collection, distances, order, ranks):
    output = ROOT / 'outputs/examples' / protocol
    output.mkdir(exist_ok=True)
    for label, indices in [('hit', np.flatnonzero(ranks == 1)), ('miss', np.flatnonzero(ranks > 1))]:
        for i in indices[:3]:
            image, sketch = collection[int(i)]
            cells = [(sketch, 'Query'), (image, f'Target rank {ranks[i]}')]
            for k, j in enumerate(order[i, :5], 1):
                path, _ = collection[int(j)]
                cells.append((path, f'#{k}   d2 {distances[i,j]:.3f}'))
            panel = Image.new('RGB', (7*164, 190), 'white')
            draw = ImageDraw.Draw(panel)
            for k, (path, title) in enumerate(cells):
                with Image.open(path) as im:
                    thumb = ImageOps.contain(im.convert('RGB'), (156,156))
                panel.paste(thumb,(164*k+(164-thumb.width)//2,(156-thumb.height)//2))
                draw.text((164*k+5,163), title, fill='black')
            panel.save(output / f'{label}_{i:04}.jpg', quality=90)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('protocol',choices=['normal','unseen'])
    p.add_argument('stage',choices=['probe','pair','small','full'])
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    p.add_argument('--batch-size',type=int,default=2,help='Upstream argument; effective batch size = 3 × value')
    args = p.parse_args()
    assert args.batch_size > 0
    seed_everything(42)
    torch.set_num_threads(4)
    start = time.perf_counter()
    weight = ROOT / 'models' / f'fscoco_{args.protocol}.pth'
    model = SBIRModel.load_module(str(weight), strict=True, cuda=False).eval().to(args.device)
    if args.device=='mps':
        torch.mps.synchronize()
    start_forward = time.perf_counter()
    report=dict(protocol=args.protocol, stage=args.stage, device=args.device, dtype='float32',
                checkpoint=weight.name, strict=True, model_class=type(model).__name__,
                config=model.dump(), parameters=sum(p.numel() for p in model.parameters()),
                load_s=start_forward-start)
    gpu_peak=0
    def sample():
        nonlocal gpu_peak
        if args.device=='mps':
            torch.mps.synchronize()
            gpu_peak=max(gpu_peak,torch.mps.driver_allocated_memory())
    def encode(x):
        feats=model(x.to(args.device)).cpu()
        sample()
        assert feats.shape==(len(x),512) and torch.isfinite(feats).all()
        assert torch.allclose(feats.norm(dim=1),torch.ones(len(x)),atol=1e-4)
        return feats.numpy()
    with torch.inference_mode():
        if args.stage=='probe':
            x=torch.zeros((1,3,224,224))
            report['probe_shape']=list(encode(x).shape)
            report['input']='synthetic tensor; hardware check only'
        else:
            data=load_dataset(str(ROOT/'data/fscoco'), mode=args.protocol)
            _,collection,_=data.split()
            paired=DatasetAdapter(collection, model.create_sketch_transforms(), model.create_image_transforms())
            n={'pair':5,'small':32,'full':len(paired)}[args.stage]
            ids=collection.file_names()
            if args.stage=='full':
                split_file='val_normal.txt' if args.protocol=='normal' else 'val_unseen_user.txt'
                split_ids=[int(v) for v in (ROOT/'data/fscoco'/split_file).read_text().splitlines()]
                assert sorted(split_ids)==sorted(int(key) for _,key in ids)
            batch=1 if args.stage=='pair' else args.batch_size*3
            records=DataLoader(Subset(paired,range(n)),batch_size=batch,shuffle=False)
            s,im=[],[]
            for a,b,_ in records:
                if args.stage != 'pair' or not s:
                    t=time.perf_counter()
                    s.append(encode(a))
                    if args.stage=='pair':
                        report['single_sketch_forward_s']=time.perf_counter()-t
                t=time.perf_counter()
                im.append(encode(b))
                if args.stage=='pair' and len(im)==1:
                    report['single_photo_forward_s']=time.perf_counter()-t
            s,im=np.concatenate(s),np.concatenate(im)
            if args.stage=='pair':
                s=s[:1]
            dist=compute_distances(s,im)
            order=dist.argsort(axis=1)
            rank=np.argmax(order==np.arange(len(s))[:,None],axis=1)+1
            counts=calculate_results(dist)
            assert tuple((rank<=k).sum() for k in (1,5,10))==counts[:3]
            report.update(queries=len(s),gallery=len(im),batch=batch,feature_dim=s.shape[1],
                          first_sketch_norm=float(np.linalg.norm(s[0])),first_photo_norm=float(np.linalg.norm(im[0])),
                          first_pair_top5=[dict(id=ids[j],d2=float(dist[0,j])) for j in order[0,:5]],
                          hits=list(map(int,counts[:3])),recall_pct=[100*int(c)/len(s) for c in counts[:3]])
            if args.stage=='full':
                rows=[dict(query=ids[i],rank=int(rank[i]),top5=[ids[j] for j in order[i,:5]],
                           d2=[float(dist[i,j]) for j in order[i,:5]]) for i in range(n)]
                save(f'{args.protocol}_ranks.json',rows)
    end=time.perf_counter()
    report.update(elapsed_s=end-start,post_load_s=end-start_forward,
                  peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  max_sampled_metal_driver_bytes=gpu_peak if args.device=='mps' else None)
    suffix='_cpu' if args.stage=='probe' and args.device=='cpu' else ''
    save(f'{args.protocol}_{args.stage}{suffix}.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    if args.stage=='full':
        make_examples(args.protocol,collection,dist,order,rank)

if __name__=='__main__':
    main()
