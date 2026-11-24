# 数据和权重校验

上游版本 bbfc2499d817fc64dfafcf02d84bf98e7a44ad24。正常协议与未见画者协议分别使用作者共享的 fscoco_normal.pth 和 fscoco_unseen.pth。下载采用同一条 Dropbox 分享的官方下载域名；首次直连超时，改用主机配置的 HTTPS 代理。文件摘要为本地接收时的指纹，作者 README 未列出用于独立比对的 SHA-256。

模型入口为 SBIRModel.load_module(..., strict=True, cuda=False)。先将 state 映射到 CPU，按文件的 type 和 config 创建网络，再严格恢复全部参数并转至推理设备。指定 _nopretrain=True 避免再次下载通用预训练权重。严格加载失败会终止评测。Normal 已通过一次主机 MPS 前向，88,090,752 参数，单位范数的 1×512 有限浮点输出。证据见 outputs/metrics/normal_probe.json。

官方模型的默认预训练配置为 OpenCLIP convnext_base / laion400m_s13b_b51k。草图与照片共用 model.visual。官方 DatasetAdapter 先将草图转 L、照片转 RGB，预处理再统一到 RGB，以带 antialias 的双三次插值缩放到 224×224，转换 FP32 至 [0,1]，按 CLIP 均值 (0.48145466,0.4578275,0.40821073) 和标准差 (0.26862954,0.26130258,0.27577711) 归一化。最后对每条特征做 L2 归一化。

数据来源为 FS-COCO 官方站点。分段下载验证 HTTP Content-Range 和字节数，合并后通读 gzip 校验尾部 CRC，解压时只接受 fscoco/ 内的常规文件与目录。脚本校验照片、笔画 JSON、栅格草图的键、逐图解码、笔画字段、测试 ID 唯一性及训练/测试 ID 互斥。官方渲染函数使用归一化坐标、256×256 画布、1 像素线与 3×3 膨胀；已有栅格优先使用，缺少时按原函数补齐。测试集合由 val_normal.txt 或 val_unseen_user.txt 的数值 ID 选取，未命中构成对应训练集。

## 获取入口

[官方模型与预处理代码](https://github.com/Emil-Demic/SketchScape/tree/bbfc2499d817fc64dfafcf02d84bf98e7a44ad24) · [FS-COCO 官方数据页](https://fscoco.github.io/) · [论文](https://arxiv.org/abs/2509.06566)

复用现有完整文件。在项目根目录可按协议选择作者公布的链接，使用 curl -fL -C - 将文件先写到 models 下相应的 .part，再在传输成功后改名为 fscoco_normal.pth 或 fscoco_unseen.pth。分段数据脚本支持 HTTPS_PROXY，使用时沿用本机正常访问网络的配置。

```bash
curl -fL -C - --retry 3 'https://dl.dropboxusercontent.com/scl/fi/6s7dhekt4ikihuoptobhy/fscoco_normal.pth?rlkey=7jklykr8zq9ioxx31q2gdbeqx&dl=1' -o models/fscoco_normal.pth.part && mv -n models/fscoco_normal.pth.part models/fscoco_normal.pth
curl -fL -C - --retry 3 'https://dl.dropboxusercontent.com/scl/fi/9fv399bupkandn6ouj41g/fscoco_unseen.pth?rlkey=pk3t5t2winuejff8p0e7mxap6&dl=1' -o models/fscoco_unseen.pth.part && mv -n models/fscoco_unseen.pth.part models/fscoco_unseen.pth
```

## 全量文件与划分实测

官方 tar.gz 的 SHA-256 为 4c71f8976bd9085baa1e530c45d38ce715ea3fb06bb1a9a3f1b8a7c789912f13。60,609 个归档条目，压缩大小 2,378,884,718 B，原文件逻辑总量 5,293,619,523 B。10,000 张照片和 10,000 个原始笔画 JSON 一一配对，10,000 个数值图像 ID 唯一。归档自带 raster_sketches 下的 JPG，而官方代码读取同名 PNG。因此用作者的 _render_sketch 从笔画生成全部 10,000 张完整 PNG，逐个解码评测使用的照片和 PNG，逐个解析笔画字段，均通过。原归档保留。归档信息及逐类计数见 outputs/metrics/archive.json 与 dataset.json。

每个协议均为 7,000 训练配对、3,000 测试配对，协议内训练和测试图像 ID 交集为零。两协议的测试集交集为 873 个图像 ID，不同协议独立评测。Normal 划分文件 SHA-256 为 91623e2514b73a2a1f17a64c548aa604b2c9787c9d32985fe9535e483b1bed5b，Unseen 为 ee299ada6fc4813327866b656c34fa7dd94c621609506cdbc935c80314e42f87。

需要区分文件夹编号和画者身份。论文对 FS-COCO 的划分描述给出 Unseen 按画者隔离。按实际图像路径统计，Normal 测试集覆盖全部 100 个编号目录，每目录 18–41 张；Unseen 测试集也覆盖全部 100 个编号目录，每目录正好 30 张。两种协议的训练集均覆盖这些目录。仅凭目录号无法独立验证画者隔离。复现严格使用官方文件和代码按图像 ID 构建图库。课程报告可准确称为“官方 Unseen 划分”。[论文评测协议](https://arxiv.org/html/2509.06566v1)

Normal 和 Unseen 两份 checkpoint 各为 352,511,492 字节。Normal 的 SHA-256 为 6fd930276c8addf506b67a1a897fb671ec7c16e41fccf90b8b177223df1f973a，Unseen 为 5ebbbdb0c134a2cdb24d70118012e978ba3443a85c21b41390486c9e930d57fe。两份严格载入与全量实图前向通过，参数键、形状无缺失或冗余。另完整解码验证了归档中的全部 10,000 张原始草图 JPG。
