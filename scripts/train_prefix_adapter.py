"""Train only on a protocol's official training IDs. Test after checkpoint selection."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from cache_prefix_features import CACHE, DATA, ROOT, WEIGHTS
from utils import calculate_results, compute_distances

OUT=ROOT/'outputs/adapter'
RATES=(25,50,75)
SEED=20261122
MILESTONES=(1,2,4,8,12,16,24)

class SketchAdapter(nn.Module):
    """Zero-initialized residual projection of a unit-length sketch vector."""
    def __init__(self):
        super().__init__()
        self.down=nn.Linear(512,128)
        self.up=nn.Linear(128,512)
        self.dropout=nn.Dropout(0.1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self,x):
        h=F.gelu(self.down(x*math.sqrt(512)))
        return F.normalize(x+self.up(self.dropout(h))/math.sqrt(512),dim=-1)

def read_cache(protocol):
    folder=CACHE/protocol
    meta=json.loads((folder/'manifest.json').read_text())
    assert meta['weight_sha256']==WEIGHTS[protocol] and meta['protocol']==protocol
    ids=[k for _,k in meta['ids']]
    assert len(set(ids))==len(ids)==10000
    feats={}
    for view in meta['views']:
        blocks=[np.load(folder/f'{view}_{i:05d}.npy') for i in range(0,len(ids),1000)]
        a=np.concatenate(blocks)
        assert a.shape==(10000,512) and a.dtype==np.float32
        assert np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-4)
        feats[view]=torch.from_numpy(a)
    return ids,feats

def partition(ids,protocol):
    split_name='val_normal.txt' if protocol=='normal' else 'val_unseen_user.txt'
    official={int(k) for k in (DATA/split_name).read_text().splitlines()}
    test=np.array([i for i,k in enumerate(ids) if int(k) in official])
    train_pool=np.array([i for i,k in enumerate(ids) if int(k) not in official])
    shuffled=np.random.default_rng(SEED).permutation(train_pool)
    val,train=np.sort(shuffled[:700]),np.sort(shuffled[700:])
    assert (len(train),len(val),len(test))==(6300,700,3000)
    assert len(set(np.concatenate([train,val,test])))==10000
    return train,val,test

def write_json(path,content):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(content,indent=2)+'\n')
    os.replace(tmp,path)

@torch.no_grad()
def dev_recalls(head, feats, idx):
    gallery=feats['photo'][idx]
    target=torch.arange(len(idx))
    scores={}
    for r in RATES:
        x=feats[str(r)][idx]
        sim=(x if head is None else head(x))@gallery.T
        pred=sim.topk(10,dim=1).indices
        scores[r]=[round(float((pred[:,:k]==target[:,None]).any(1).sum())*100/len(idx),5) for k in [1,5,10]]
    return scores

def train(protocol,ids,feats,train_ids,dev_ids):
    work=OUT/protocol
    work.mkdir(exist_ok=True)
    write_json(work/'development_split.json',dict(seed=SEED,train=[ids[i] for i in train_ids],dev=[ids[i] for i in dev_ids]))
    log=[]; best_key=None
    target_path=ROOT/'models'/f'adapter_{protocol}.pth'
    base=dev_recalls(None,feats,dev_ids)
    t0=perf_counter()
    # Two fixed objectives. Share the random stream across both comparisons.
    for strength in (0.,2.):
        torch.manual_seed(SEED)
        head=SketchAdapter()
        opt=torch.optim.AdamW(head.parameters(),lr=0.0005,weight_decay=0.01)
        rng=np.random.default_rng(SEED)
        for epoch in range(1,max(MILESTONES)+1):
            head.train()
            loss_sum=0.; total=0
            # Each training sketch is seen at each completion once per epoch.
            # Independently permute each pass and rotate rates within a batch;
            # paired photo IDs stay unique among in-batch negatives.
            rates=rng.permutation(np.arange(3).repeat(math.ceil(len(train_ids)/3)))[:len(train_ids)]
            for pass_num in range(3):
                for batch in np.array_split(rng.permutation(len(train_ids)),math.ceil(len(train_ids)/192)):
                    idx=train_ids[batch]
                    rr=(rates[batch]+pass_num)%3
                    x=torch.empty(len(idx),512)
                    for j,r in enumerate(RATES):
                        mask=rr==j
                        x[mask]=feats[str(r)][idx[mask]]
                    z=head(x)
                    sim=z@feats['photo'][idx].T/0.035
                    target=torch.arange(len(idx))
                    contrast=(F.cross_entropy(sim,target)+F.cross_entropy(sim.T,target))/2
                    loss=contrast+strength*(1-(z*feats['100'][idx]).sum(1)).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(head.parameters(),1.)
                    opt.step()
                    loss_sum+=loss.item()*len(idx); total+=len(idx)
            if epoch not in MILESTONES:
                continue
            head.eval()
            rec=dev_recalls(head,feats,dev_ids)
            # Primary criterion is early R@1; ties use early R@5 then earliest epoch.
            key=(rec[25][0]+rec[50][0],rec[25][1]+rec[50][1],-epoch)
            log.append(dict(epoch=epoch,teacher_weight=strength,mean_loss=loss_sum/total,dev=rec))
            if best_key is None or key>best_key:
                best_key=key
                best=dict(epoch=epoch,teacher_weight=strength,dev=rec)
                ck=dict(state_dict=head.state_dict(),protocol=protocol,base_sha256=WEIGHTS[protocol],
                        seed=SEED,epoch=epoch,teacher_weight=strength,mode='partial_only')
                tmp=target_path.with_suffix('.tmp')
                torch.save(ck,tmp); os.replace(tmp,target_path)
            print(protocol,'development',epoch,strength,rec,flush=True)
    write_json(work/'training.json',dict(seed=SEED,train_n=len(train_ids),dev_n=len(dev_ids),
        trainable=sum(p.numel() for p in head.parameters()),optimizer='AdamW',lr=.0005,dropout=.1,
        batch_max=192,temperature=.035,baseline=base,selected=best,history=log,seconds=perf_counter()-t0))

def ranks_for(query,gallery):
    # Same exact squared Euclidean metric and argsort rule as upstream utils.py.
    d=compute_distances(query.numpy(),gallery.numpy())
    order=d.argsort(axis=1)
    ranks=np.argmax(order==np.arange(len(query))[:,None],axis=1)+1
    counted=calculate_results(d)
    assert np.array_equal([(ranks<=k).sum() for k in [1,5,10]],counted[:3])
    return ranks,order[:,0]

def paired_delta(before,after,rng):
    delta=100*((after<=1).astype(float)-(before<=1).astype(float))
    means=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(3000)])
    return [float(v) for v in np.quantile(means,[.025,.975])]

@torch.inference_mode()
def evaluate(protocol,ids,feats,test_ids):
    t0=perf_counter()
    work=OUT/protocol
    model_path=ROOT/'models'/f'adapter_{protocol}.pth'
    saved=torch.load(model_path,map_location='cpu',weights_only=True)
    assert saved['protocol']==protocol and saved['base_sha256']==WEIGHTS[protocol]
    head=SketchAdapter().eval();head.load_state_dict(saved['state_dict'],strict=True)
    gallery=feats['photo'][test_ids]
    rows={}; metrics={}; rng=np.random.default_rng(SEED+1)
    for r in (*RATES,100):
        q=feats[str(r)][test_ids]
        before,base_top=ranks_for(q,gallery)
        after,adapt_top=ranks_for(head(q),gallery) if r!=100 else (before,base_top)
        metrics[r]=dict(baseline=[int((before<=k).sum()) for k in (1,5,10)],
                        adapter=[int((after<=k).sum()) for k in (1,5,10)],
                        gained_top1=int(((before!=1)&(after==1)).sum()),lost_top1=int(((before==1)&(after!=1)).sum()),
                        delta_r1_ci95_pp=paired_delta(before,after,rng))
        for i,j in enumerate(test_ids):
            rows.setdefault(ids[j],{'id':ids[j]}).update({
              f'base_rank_{r}':int(before[i]),f'adapter_rank_{r}':int(after[i]),
              f'base_first_{r}':ids[test_ids[base_top[i]]],f'adapter_first_{r}':ids[test_ids[adapt_top[i]]]})
    # Complete sketches bypass the head in deployment; also measure always-on behavior.
    always,_=ranks_for(head(feats['100'][test_ids]),gallery)
    ref=json.loads((ROOT/'outputs/metrics'/f'{protocol}_full.json').read_text())
    with (work/'test_ranks.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=next(iter(rows.values())).keys())
        writer.writeheader();writer.writerows(rows.values())
    with model_path.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
    write_json(work/'test_metrics.json',dict(queries=len(test_ids),gallery=len(gallery),checkpoint_sha256=sha,
        seed=SEED,metrics=metrics,complete_always_on=[int((always<=k).sum()) for k in (1,5,10)],
        previous_complete_hits=ref['hits'],seconds=perf_counter()-t0))
    print(protocol,'test',metrics,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('protocol',choices=WEIGHTS)
    p.add_argument('stage',choices=['train','test'])
    args=p.parse_args()
    torch.set_num_threads(3)
    ids,feats=read_cache(args.protocol)
    train_idx,dev_idx,test_idx=partition(ids,args.protocol)
    if args.stage=='train':
        train(args.protocol,ids,feats,train_idx,dev_idx)
    else:
        evaluate(args.protocol,ids,feats,test_idx)

if __name__=='__main__':
    main()
