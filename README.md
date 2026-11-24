# SketchScape 渐进式草图检索

智能计算系统期末项目。输入一张还在绘制中的草图，在 3,000 张照片中找出对应场景。项目复现了 SketchScape 完整草图检索，并在冻结编码器后训练了一个轻量适配器，专门改善只画了几笔时的检索效果。

## 检查项目与制作汇报

以下文件可直接下载、阅读或插入幻灯片。检查报告和结果无需安装 Python。

| 想说明的问题 | 对应材料 |
| --- | --- |
| 做了什么，系统如何工作 | [技术报告 PDF](docs/SketchScape渐进式草图检索系统技术报告.pdf) · [可编辑 Word](docs/SketchScape渐进式草图检索系统技术报告.docx) |
| 官方基线能否复现 | [双协议基线和论文数值对照](docs/03_baseline_reproduction.md) |
| 适配器如何训练，为什么对早期草图有用 | [实验设计](docs/05_prefix_adapter_experiment.md) · [各阶段全量指标](docs/06_prefix_adapter_results.md) · [效果图 PNG](outputs/adapter/retrieval_effect.png) |
| 一笔笔画时结果如何变化 | [斑马案例截图](outputs/demo/replay.png) · [答辩投影视图](outputs/demo/projector.png) · [现场手绘截图](outputs/demo/live.png) |
| 能否复核结果，哪些情况会退步 | [Normal 逐图排名](outputs/adapter/normal/test_ranks.csv) · [退步案例](outputs/adapter/normal/first_lost_25.jpg) · [演示验收记录](outputs/demo/api_check.log) |

在 Normal 协议、25% 绘制用时处，首位命中率从 11.90% 提高到 16.17%。在 75% 处从 51.80% 变为 50.47%。上述指标来自全部 3,000 条测试查询。界面中的 12 个案例按效果精选，适合用来解释某一幅草图的排名变化。[完整实验结果](docs/06_prefix_adapter_results.md)

## 拉取项目

```bash
git clone --recurse-submodules https://github.com/Jdeciderz23/sketchscape-retrieval.git
cd sketchscape-retrieval
```

已有普通克隆可运行 `git submodule update --init --recursive`，获取本项目使用的固定版本上游代码。[SketchScape 源码和许可证](https://github.com/Emil-Demic/SketchScape/tree/bbfc2499d817fc64dfafcf02d84bf98e7a44ad24) · [FS-COCO 数据来源](https://fscoco.github.io/)

## 在新电脑运行交互演示

仓库包含两份训练好的适配器。第一次运行还需要下载 FS-COCO 与官方 Normal 编码器，并生成特征缓存。以下步骤适用于安装了 Python 3.12、Git 与 curl 的 macOS 或 Linux 终端，在仓库根目录执行。

1. 建立 Python 环境并安装复现时记录的依赖。

   ```bash
   python3.12 -m venv .venv
   .venv/bin/python -m pip install -r outputs/logs/environment.txt
   ```

2. 按[数据与权重说明](docs/02_dataset_and_model_validation.md)下载 `models/fscoco_normal.pth`。演示使用 Normal 权重。获取官方数据并生成草图。

   ```bash
   .venv/bin/python scripts/fetch_dataset.py
   .venv/bin/python scripts/validate_data.py
   .venv/bin/python scripts/render_prefixes.py
   ```

3. 首次生成 Normal 特征缓存，随后启动演示。Apple Silicon 可将缓存命令的 `cpu` 换为 `mps`。

   ```bash
   .venv/bin/python scripts/cache_prefix_features.py normal --device cpu --batch-size 16
   .venv/bin/python demo/server.py --open
   ```

准备好资源后，macOS 也可以双击仓库根目录的“启动检索演示.command”。完整基线复现和训练命令分别见上方实验文档。[演示操作与验收指南](docs/07_interactive_demo.md)
