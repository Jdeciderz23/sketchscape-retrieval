"""Verify rank-level results and prepare the course report and paired retrieval examples."""
from pathlib import Path
import csv
import json
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/adapter'
DATA=ROOT/'data/fscoco'
KS=(1,5,10)
def read(path):
    return json.loads(path.read_text())

def example(protocol,rows,which,lookup):
    def top(row,key):
        return int(row[f'{key}_rank_25'])
    def match(row):
        return top(row,'base')!=1 and top(row,'adapter')==1 if which=='gained' else top(row,'base')==1 and top(row,'adapter')!=1
    row=next((r for r in rows if match(r)),None)
    if row is None: return
    k=row['id']; folder=lookup[k]
    cells=[(ROOT/'data/time_prefix_v1/25'/folder/f'{k}.png','25% time prefix'),
           (DATA/'raster_sketches'/folder/f'{k}.png','Complete sketch (reference)'),
           (DATA/'images'/folder/f'{k}.jpg','Paired photo')]
    for model in ['base','adapter']:
        j=row[f'{model}_first_25']
        cells.append((DATA/'images'/lookup[j]/f'{j}.jpg',f'{model.title()} Top 1 | target #{top(row,model)}'))
    canvas=Image.new('RGB',(5*224,255),'white')
    draw=ImageDraw.Draw(canvas)
    for i,(path,title) in enumerate(cells):
        with Image.open(path) as im: im=ImageOps.contain(im.convert('RGB'),(214,215))
        canvas.paste(im,(224*i+(224-im.width)//2,(218-im.height)//2))
        draw.text((224*i+5,227),title,fill='black')
    canvas.save(OUT/protocol/f'first_{which}_25.jpg',quality=92)
    return dict(id=k,before=top(row,'base'),after=top(row,'adapter'))

def main():
    doc=['# 不完整草图适配器实测结果','',
         '每行使用 3,000 条测试查询和同一协议的全部 3,000 张测试照片。命中率单位为 %，差值单位为百分点。100% 行使用完成模式直接返回原特征。','',
         '| 协议 | 绘制用时 | R@1 基线 → 适配 | R@5 基线 → 适配 | R@10 基线 → 适配 | R@1 差值 [95% 区间] |',
         '|---|---:|---:|---:|---:|---:|']
    notes=[]; summary={}
    for protocol in ['normal','unseen']:
        work=OUT/protocol
        res=read(work/'test_metrics.json')
        training=read(work/'training.json')
        meta=read(OUT/'cache'/protocol/'manifest.json')
        lookup={key:folder for folder,key in meta['ids']}
        split_file='val_normal.txt' if protocol=='normal' else 'val_unseen_user.txt'
        test=set((DATA/split_file).read_text().splitlines())
        with (work/'test_ranks.csv').open() as f: rows=list(csv.DictReader(f))
        assert {r['id'] for r in rows}==test and len(rows)==res['queries']==res['gallery']==3000
        split=read(work/'development_split.json')
        assert len(set(split['train'])|set(split['dev'])|test)==10000
        assert len(split['train'])==6300 and len(split['dev'])==700
        with (ROOT/'models'/f'adapter_{protocol}.pth').open('rb') as f:
            assert hashlib.file_digest(f,'sha256').hexdigest()==res['checkpoint_sha256']
        diffs=[]
        for rate in [25,50,75,100]:
            d=res['metrics'][str(rate)]
            pre=np.array([int(r[f'base_rank_{rate}']) for r in rows])
            post=np.array([int(r[f'adapter_rank_{rate}']) for r in rows])
            assert all(1<=k<=3000 for k in np.r_[pre,post])
            assert [(pre<=k).sum() for k in KS]==d['baseline'] and [(post<=k).sum() for k in KS]==d['adapter']
            for row in rows:
                for side in ('base','adapter'):
                    assert (row['id']==row[f'{side}_first_{rate}'])==(int(row[f'{side}_rank_{rate}'])==1)
            delta=100*((post==1).astype(float)-(pre==1).astype(float))
            if rate in (25,50): diffs.append(delta)
            lo,hi=d['delta_r1_ci95_pp']
            cells=' | '.join(f'{a/30:.2f} → {b/30:.2f}' for a,b in zip(d['baseline'],d['adapter']))
            doc.append(f'| {protocol.title()} | {rate}% | {cells} | {delta.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] |')
        early=np.mean(diffs,axis=0); rng=np.random.default_rng(20261124)
        means=[rng.choice(early,len(early),replace=True).mean() for _ in range(3000)]
        lo,hi=map(float,np.quantile(means,[.025,.975]))
        summary[protocol]=dict(early_mean_delta_pp=float(early.mean()),early_ci95_pp=[lo,hi],examples={})
        for which in ['gained','lost']:
            summary[protocol]['examples'][which]=example(protocol,rows,which,lookup)
        if res['metrics']['100']['baseline']!=res['previous_complete_hits']:
            comparison=f"此次计数 {res['metrics']['100']['baseline']}，上阶段 {res['previous_complete_hits']}，两次批量大小分别为16与6。"
        else:
            comparison='本次完整草图的三项命中数与上阶段逐项相同。'
        chosen=training['selected']
        notes.extend([f"### {protocol.title()}",'',
           f"预定主指标（25% 与 50% 的平均 R@1）差值为 {early.mean():+.2f} 个百分点，95% 配对查询区间 [{lo:+.2f}, {hi:+.2f}]。验证集选中完整草图对齐权重 {chosen['teacher_weight']:g}、第 {chosen['epoch']} 轮。两种候选合计训练 {training['seconds']:.1f} 秒。",
           '',f"强制对完整草图启用适配器时，R@1 / R@5 / R@10 为 {' / '.join(f'{n/30:.2f}%' for n in res['complete_always_on'])}。{comparison}",
           '',f"25% 查询中，新增 {res['metrics']['25']['gained_top1']} 个首位命中，同时丢失 {res['metrics']['25']['lost_top1']} 个原有首位命中。两张示例图按官方查询顺序选取第一条改善和第一条退步记录。",''])
    notes.extend(['## 结果图与复核材料','',
                  '[各阶段检索效果图](../outputs/adapter/retrieval_effect.png) · [实验设计与复现命令](05_prefix_adapter_experiment.md)',''])
    for protocol in ['normal', 'unseen']:
        links = [f'[逐条排名](../outputs/adapter/{protocol}/test_ranks.csv)',
                 f'[训练权重](../models/adapter_{protocol}.pth)']
        for which, label in [('gained','首位改善案例'), ('lost','首位退步案例')]:
            if summary[protocol]['examples'][which]:
                links.append(f'[{label}](../outputs/adapter/{protocol}/first_{which}_25.jpg)')
        notes.extend([f"{protocol.title()}　" + ' · '.join(links), ''])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (ROOT/'docs/06_prefix_adapter_results.md').write_text('\n'.join(doc+['']+notes)+'\n')

if __name__=='__main__':
    main()
