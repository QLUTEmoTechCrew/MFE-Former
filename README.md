# MFE-Former

## 目录结构

```text
(JBHI2025)MFEFormer/
├── MFEFormer.py              # 总模型与随机前向测试
├── speech_encoder.py         # 词级 CNN-BiLSTM 与多尺度入口
├── attention.py              # Full/ProbSparse attention
├── encoder.py                # 三分支 3N/4 编码器
├── informer.py               # ProbSparse encoder 封装
├── contrastive_learning.py   # 多粒度时间—实例对比损失
├── identity_encoder.py       # CLIP prompt、adapter 与数值身份兼容路径
├── disentanglement.py        # F_emo/F_id 与正交约束
├── reconstruction.py        # 身份引导重构与掩码 MSE
├── ewre_dataset.py           # EWRE 索引、Mel 提取与受试者五折
├── train_ewre.py             # 完整训练/测试入口
├── requirements.txt
└── tests/
```

## 环境

建议使用 Python 3.10 或更高版本。

```bash
pip install -r requirements.txt
```

CLIP 模式第一次运行时会通过 Hugging Face 下载 `openai/clip-vit-base-patch32`。无网络环境需要提前缓存该模型，或先使用数值身份模式进行代码检查。

## EWRE 数据格式

代码不会将 EWRE 数据或人口统计信息复制到仓库。期望的数据目录如下：

```text
data_home/
├── EWRE.csv                  # GBK/GB18030 编码的合并人口统计表
├── depression.csv            # 兼容回退
├── normal.csv                # 兼容回退
└── audio/
    └── wav/
        ├── 00_0_011_1.wav
        ├── 00_0_011_2.wav
        ├── ...
        └── 46_1_099_72.wav
```

WAV 文件名必须为：

```text
<HAMD总分>_<二分类标签>_<受试者编号>_<词序号>.wav
```

约束：

- 标签 `0` 为健康对照，`1` 为抑郁。
- 每名受试者必须恰好包含词序号 `1..72`。
- 缺词、重复词、元数据缺失或 HAMD 不一致都会立即报错。
- 当前数据实测为 140 名受试者、70/70 类别平衡、48 kHz 双声道。
- 加载时双声道取平均得到单声道。

## 张量接口

```python
speech.shape == [batch_size, 72, max_frames, 80]
frame_mask.shape == [batch_size, 72, max_frames]
clip_text_features.shape == [batch_size, 512]
```

`N=72` 时：

```text
word_features:  [B, 72, C]
scale_features: 3 × [B, 18, C]
X_MS:           [B, 54, C]
reconstruction: [B, 72, 80]
logits:         [B, 2]
```

## 快速验证

### 1. 单元测试

```bash
python -m pytest tests -q
```

### 2. 随机张量前向

```bash
python MFEFormer.py
```

### 3. 真实 EWRE 单步前向与反向

数值身份模式不需要下载 CLIP，适合检查数据和 CUDA：

```bash
python train_ewre.py \
  --data-root "D:/Py_code/Datasets/EWRE/data_home" \
  --identity-mode numeric \
  --dry-run \
  --max-folds 1
```

`--dry-run` 只执行一个 batch，不保存模型。

## 正式五折训练

论文对齐路径使用 CLIP 文本身份特征：

```bash
python train_ewre.py \
  --data-root "D:/Py_code/Datasets/EWRE/data_home" \
  --identity-mode clip \
  --clip-model openai/clip-vit-base-patch32 \
  --epochs 100 \
  --warmup-epochs 10 \
  --batch-size 2 \
  --output-dir outputs/ewre
```

训练会执行以下操作：

1. 从 140 名受试者生成确定性的分层五折；
2. 每折使用 112 名训练、28 名测试受试者；
3. 固定训练 100 epoch；
4. 仅在训练结束后评估测试折，避免通过测试集选择 epoch；
5. 保存 `fold_1.pt` 至 `fold_5.pt` 和汇总 `metrics.json`。

数据目录中已有的 `audio/kfold_split.json` 只有三折，不符合论文的五折协议，因此默认不使用。

## 联合损失

```text
L = L_CE
  + lambda_cont  * L_cont
  + lambda_recon * L_recon
  + lambda_reg   * L_reg
```

命令行默认值：

```text
lambda_cont  = 1.0
lambda_recon = 1e-3
lambda_reg   = 1.0
```

`lambda_recon=1e-3` 是为了平衡未归一化 log-Mel MSE 的量纲而设置的工程默认值，不是论文公开参数。正式实验应仅在训练折内部确定损失权重，不能根据测试折调参。

## 音频参数

论文没有给出完整 Mel 配置。本实现使用可修改的明确默认值：

```text
sample_rate = 48000
window      = 25 ms
hop         = 10 ms
n_mels      = 80
max_frames  = 832
```

可通过 `--sample-rate`、`--window-ms`、`--hop-ms`、`--n-mels` 和 `--max-frames` 修改。当前数据最长词对应 809 个 Mel 帧，因此默认 832 不截断现有样本。超出 `max_frames` 的词会在尾部截断；较短的词使用零填充和显式 `frame_mask`。
