# 不完整草图适配器训练与检索评测

## 固定实验设计

使用 FS-COCO 的原始笔画，从首个记录点到最后一个记录点的经过时间定义 25%、50%、75%。保留截至该时刻的全部点（包括同一时间戳），仍用 SketchScape 的 _render_sketch 渲染。全体 10,000 个样本的 100% 重绘与原基线 PNG 逐像素相同，三种前缀图的前景像素依次包含。时间戳偶尔回退时，按原笔顺取累计最大值。涉及 5 个样本，最大修正为 785 个原始时间戳单位。25%、50% 各有 1 张空白前缀，均保留。阶段取决于绘制用时，与笔画数占比有区别。

两个协议单独使用各自的官方权重、训练 ID 和图库。冻结共享视觉编码器，使用原图像预处理，严格载入官方 checkpoint，MPS 批量 16，FP32 缓存每个样本的四种草图与照片表示。每个协议的 manifest.json 记录行序、图像 ID、官方权重的 SHA-256、视图次序。包含测试样本的无标签冻结特征仅用于后续按 ID 取相应子集。每个协议从 7,000 个官方训练 ID 中按固定种子 20261122 划出 6,300 个训练、700 个验证。图库划分、适配器参数和模型选择在两个协议间独立。官方基础模型的原训练可能已包含这 700 个样本，因此验证集用于选择新增模块，最终效果由官方测试集衡量。

对单位向量 x 使用 512→128→512 的残差 MLP，隐藏层 GELU 和 dropout 0.1，上投影零初始化，残差相加后做 L2 归一化。照片始终使用冻结特征。在绘制进行中，三种前缀共用同一个适配器，推理无需知道具体完成比例。用户明确完成时，直接复用原模型的完整草图特征。另对完整草图强制启用适配器作对照。

每个训练样本每轮使用全部三种前缀，批内以不同照片为负样本。使用双向对比交叉熵，温度 0.035。比较两种预定训练目标：单独用照片监督，以及加入权重为 2 的完整草图余弦对齐项。AdamW 学习率 0.0005，weight decay 0.01，最大批量 192，共 24 轮。第 1、2、4、8、12、16、24 轮在 700 张验证图库上比较。按验证集 25% 和 50% 的平均 R@1 选模型，平分时比较对应的平均 R@5，再优先较早轮次。两协议的权重选定后才执行最终测试。

正式评测对各协议的 3,000 张测试照片使用原版平方欧氏距离和命中规则，分别对 3,000 个查询在四个完成度测 R@1、R@5、R@10。主指标为同一查询 25%、50% 两阶段的平均 R@1。百分点差为配对差。95% 区间按查询重采样 3,000 次，反映当前测试样本的不确定性；训练使用固定单一种子。

## 复现入口

在 project 目录使用现有环境和已验证的数据、权重。重复缓存时会复用已完成的 1,000 个样本分块；正式训练从零初始化适配器。

```bash
.venv/bin/python scripts/render_prefixes.py
.venv/bin/python scripts/cache_prefix_features.py normal --device mps --batch-size 16
.venv/bin/python scripts/cache_prefix_features.py unseen --device mps --batch-size 16
.venv/bin/python scripts/train_prefix_adapter.py normal train
.venv/bin/python scripts/train_prefix_adapter.py unseen train
.venv/bin/python scripts/train_prefix_adapter.py normal test
.venv/bin/python scripts/train_prefix_adapter.py unseen test
.venv/bin/python scripts/report_prefix_results.py
```

权重位于 models/adapter_normal.pth 和 models/adapter_unseen.pth。outputs/adapter 下的各协议目录保留开发集划分、逐次验证记录、正式指标、逐查询名次和预测 ID。训练与评测使用 CPU，深度编码使用 MPS。

## 加载训练权重

在 project 目录执行。这里的 sketch_features 是对应协议冻结编码器输出的 FP32 单位向量，photo_features 仍由官方编码器直接产生。不同协议各用与本协议匹配的适配器。

```python
import sys
import torch

sys.path.insert(0, 'scripts')
from train_prefix_adapter import SketchAdapter

checkpoint = torch.load('models/adapter_normal.pth', map_location='cpu', weights_only=True)
adapter = SketchAdapter().eval()
adapter.load_state_dict(checkpoint['state_dict'], strict=True)
with torch.inference_mode():
    query = sketch_features if drawing_is_finished else adapter(sketch_features)
    similarity = query @ photo_features.T
```

完整评测在 CPU 上对单位向量计算平方欧氏距离，从小到大排列。上例的点积可用于从大到小展示相似度。
