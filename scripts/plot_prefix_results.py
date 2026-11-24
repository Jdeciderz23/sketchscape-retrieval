"""Plot audited phase-two metrics. Requires matplotlib (separate from training)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/adapter'
font_path=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
if font_path.exists():
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font_path)).get_name()
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10,
                     'axes.unicode_minus':False,'axes.labelcolor':'#374151','xtick.color':'#374151','ytick.color':'#374151'})
fig,axes=plt.subplots(1,3,figsize=(11.6,4.1),gridspec_kw={'width_ratios':[1,1,1.16]})
rates=np.array([25,50,75,100])
for panel,protocol in enumerate(['normal','unseen']):
    data=json.loads((OUT/protocol/'test_metrics.json').read_text())
    met=[data['metrics'][str(r)] for r in rates]
    a,b=(100*np.array([m[tag][0] for m in met])/data['queries'] for tag in ['baseline','adapter'])
    ax=axes[panel]
    ax.plot(rates,a,'o--',color='#718096',ms=5,lw=1.6,label='原模型')
    ax.plot(rates,b,'s-',color='#126665',ms=4.6,lw=1.8,label='适配后')
    ax.set(xticks=rates,xlim=(20,105),ylim=(0,75),yticks=np.arange(0,76,15),xlabel='绘制用时占比（%）')
    ax.set_title(f'{protocol.title()}  ·  R@1',loc='left',fontweight='bold')
    ax.grid(axis='y',alpha=.18)
    if panel==0:
        ax.set_ylabel('首位命中率（%）')
        ax.legend(loc='lower right',frameon=False,fontsize=9)
    d=b[:3]-a[:3]
    ci=np.array([m['delta_r1_ci95_pp'] for m in met[:3]])
    axes[2].errorbar(np.arange(3)+(panel-.5)*.18,d,yerr=np.maximum(0,np.stack([d-ci[:,0],ci[:,1]-d])),
                     fmt=['o','s'][panel],capsize=3.5,label=protocol.title(),color=['#126665','#af552d'][panel])
ax=axes[2];ax.axhline(0,ls='--',color='#8b9299',lw=1)
ax.set_title('部分草图的 R@1 变化',loc='left',fontweight='bold')
ax.set(xticks=[0,1,2],xticklabels=['25%','50%','75%'],xlim=(-.45,2.5),xlabel='绘制用时占比',ylabel='适配后 − 原模型（百分点）')
ax.grid(axis='y',alpha=.18)
ax.legend(loc='best',fontsize=8.5,frameon=False)
fig.suptitle('FS-COCO 完整图库上的草图检索',x=.07,y=.97,ha='left',fontsize=16,fontweight='bold')
fig.subplots_adjust(left=.075,right=.985,top=.79,bottom=.215,wspace=.42)
fig.text(.075,.035,'每个协议 3,000 个查询 × 3,000 张照片。误差线为配对查询的 95% 自助法区间。完整草图使用原特征。',size=9,color='#475569')
fig.savefig(OUT/'retrieval_effect.png',dpi=185,facecolor='white')
