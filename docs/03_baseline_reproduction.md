# 完整草图检索基线

## 正式结果

M1 Pro / MPS，FP32，每批 6 对，每个协议 3,000 条查询各对完整 3,000 张照片。表中括号是相对论文表 1 的百分点差。对每条查询的目标名次重新计数，与官方指标函数的整数命中数一致。

| 协议 | R@1 | R@5 | R@10 | 命中数 @1 / @5 / @10 | 评测秒数 | RSS 峰值 / Metal 驱动采样最大值 |
|---|---:|---:|---:|---|---:|---|
| Normal | 62.10% (+0.20) | 81.47% (+0.07) | 87.20% (0.00) | 1863 / 2444 / 2616 | 128.65 | 1.23 / 1.30 GiB |
| Unseen | 59.93% (−0.07) | 80.37% (+0.17) | 86.23% (+0.13) | 1798 / 2411 / 2587 | 128.95 | 1.23 / 1.30 GiB |

参考值按论文公开的一位小数比较。差异的具体来源未定位。两阶段使用同一套 FP32 实现、官方配对和作者发布的不同权重。每个协议留有 pair、small、full 三份 JSON、3,000 条逐图排名和 6 张固定案例。首张真实草图与第一张真实照片单独前向及同步传回时间，Normal 0.356 / 0.031 秒，Unseen 0.216 / 0.029 秒。每次进程中草图先运行，包含首调用开销。这与批量吞吐是不同测法。

固定上游提交 bbfc2499d817fc64dfafcf02d84bf98e7a44ad24，原仓库工作树独立。检查 README、requirements、model、data、eval、utils、train 和 loss；训练脚本默认 CLIP;convnext_base，损失 ICon。未训练任何模型。

复现入口 scripts/evaluate.py 复用官方权重加载、模型、数据集、预处理、距离与 Recall 函数。与上游 eval.py 的差异是严格装载参数、提供显式 MPS 设备和检验/计时/逐图导出。上游默认 strict=False 且仅自动选择 CUDA 或 CPU。Unseen 的实际参数是 --unseen。批量乘数仍取 3，默认实际 180 对，本机全量评测实际 6 对。按协议官方顺序分别编码完整测试集里的全部草图和配对照片，图库只取该协议的完整测试照片。utils.compute_distances 实际计算平方欧氏距离，在单位向量上与余弦相似度反向排序等价。每行升序 argsort，对角配对在前 K 名即命中。固定 seed 42，无测试集调参。

从 project 目录运行以下命令。主机终端可用 mps，沙箱内可改 cpu。pair 用第一幅草图检索 5 张照片，small 使用固定前 32 对；这两步仅检查流水线。full 保存正式成绩和每条查询的目标名次、前五样本和距离，并固定选取排序顺序中最先出现的三组成功案例和三组失败案例。语义相似不计实例命中。

```bash
git -C SketchScape rev-parse HEAD
.venv/bin/python -m pip check
python3 scripts/fetch_dataset.py
.venv/bin/python scripts/validate_data.py
bash scripts/run_protocol.sh normal mps
bash scripts/run_protocol.sh unseen mps
```

固定环境详见 outputs/logs/environment.txt。复装时在独立 Python 3.12 venv 执行 pip install -r outputs/logs/environment.txt。论文参考值依次为 Normal 61.9 / 81.4 / 87.2%，Unseen 60.0 / 80.2 / 86.1%。

复现时钟从创建模型开始，包含严格载入、数据枚举、预处理、前向、距离/排序及逐图排名写出；不包含依赖导入、权重下载或示例图片排版。峰值 RSS 为 macOS getrusage 字节数；Metal 数值取每次编码同步后的驱动占用采样最大值，可能低于瞬时峰值，统一内存下二者不可相加。

## 已核查的案例

Normal 测试索引 0（图像 ID 000000083613）在完整图库排第 1，树木、围栏和动物的位置与查询的构图一致。索引 3（ID 000000190767）的目标排第 4。模型的首选也是草地、围栏和一群动物，场景语义相近，但实例和动物位置不同；按官方定义 R@1 记错、R@5 记对。图中 d2 是平方欧氏距离，越小排名越靠前。见 outputs/examples/normal/hit_0000.jpg 与 miss_0003.jpg。
