# 数据采集、版本固定与完整响应缓存

没有原始逐条缓存，所有真实补实验都必须重新推理。原论文表 2 的 20 个 revision prefix 已在 Hugging Face 上核验并解析为完整 commit，见 `core20.locked.json`；`core20.json` 保留论文前缀。`independent_judges8.json`、`nli4.json` 已固定到完整 commit，无须再取 main。它们是新测量，不能假定与已丢失的原推理产物逐 bit 一致。

本脚本不改变原 `experiments/score_panel.py`。每次只采集指定 judges，可并行写入一个共享目录；只允许各 worker 负责不同 judges。缓存校验 manifest/input/protocol fingerprint，版本或数据变了必须用新目录。使用 atomic writes，缺少任一 judge/context 的完整 cache 时不合并、不插补。保存逐项概率、原始输入哈希、prompt hash、版本、tokenization 审计、warmup、加载时间、batch timing、环境与峰值显存。默认拒绝静默截断；确实要测试截断后的接口时显式传 `--allow-truncation`，并把它作为单独协议报告。

## 准备输入（CPU）

数据版本已于 2026-11-09 核验。以下是新采集的数据版本，不声称是原稿未提供的历史数据 revision。

```bash
.venv/bin/python supplementary/exp00_response_cache/run.py prepare \
  --kind rewardbench --revision 7ff08853b0d5686e79b13fda8677024f566a104a \
  --output results/supplementary/data/rewardbench.jsonl

.venv/bin/python supplementary/exp00_response_cache/run.py prepare \
  --kind nli --revision bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c \
  --output results/supplementary/data/nli.jsonl
```

RewardBench 每个源任务至多两个 comparison pairs，按原构造器复建；NLI 用 MNLI validation_matched，按 premise hash 分组，避免同 premise 的多 hypothesis 跨 split 泄漏。NLI4 只有 clean context，不能用它证明多上下文鲁棒性。`prepare --limit N` 按完整源组抽 N 组，供试跑。不要把 smoke 的小样本结果混进正式结果。

## 推理（GPU 环境）

推理需要单独配置 GPU 环境；依赖及原测量使用的版本见 `../runtime/requirements-gpu-requested.txt` 和 `../runtime/requirements-gpu-frozen.txt`。以下是调用示例，会下载锁定版本权重并使用指定 GPU；并不表示目标仓库已安装环境或正在运行任务。运行前检查 GPU，确认当前账户有各受限模型的权重访问权限。正式复现实验应使用结果包 `protocol.json` 记录的 batch size、repeats、dtype 和截断设置。
```bash
# 分片示例：当前主机 GPU 0；其余 worker 应负责互不重叠的 judge 集合。
CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python supplementary/exp00_response_cache/run.py score \
  --manifest supplementary/exp00_response_cache/core20.locked.json \
  --pairs results/supplementary/data/rewardbench.jsonl \
  --judges J01 J02 J03 J04 J05 J06 J07 --batch-size 4 --repeats 1 \
  --output results/supplementary/cache/core20

# 当前主机 GPU 0，新组建且不与原模型 ID 重叠的 judge panel。
CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python supplementary/exp00_response_cache/run.py score \
  --manifest supplementary/exp00_response_cache/independent_judges8.json \
  --pairs results/supplementary/data/rewardbench.jsonl --batch-size 4 \
  --output results/supplementary/cache/judges8

# 独立 NLI 接口；最大 token 长度必须在四个模型中一致且兼容。
CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python supplementary/exp00_response_cache/run.py score \
  --manifest supplementary/exp00_response_cache/nli4.json \
  --pairs results/supplementary/data/nli.jsonl --max-tokens 512 \
  --dtype float32 --batch-size 4 --output results/supplementary/cache/nli4
```

实际测量成本时，在冻结的代表性 workload 上用相同 batch size、多次 repeats，区分 cold load 和 warm inference；`--repeats 5` 会付出五倍实际打分工作，不能当成免费计时。为完整成本 frontier，需要对所有端点都用同一输入/版本/计时协议。不同 timing 参数必须使用独立目录，然后重新 merge；不能把不同协议的块拼接。

NLI 原生输出标签顺序在 config 中核验并与 `canonical_indices` 对照。judge 使用源码 `LabelScorer` 的 continuation scoring 和 chat template，不请求 free-form generation。不同模型库兼容性需要首次 smoke 验证。遇到不一致 label 编码、截断或模型配置时会明确失败，不能把失败模型静默从生态删除。

## 合并成分析输入

```bash
.venv/bin/python supplementary/exp00_response_cache/run.py merge \
  --manifest supplementary/exp00_response_cache/core20.locked.json \
  --pairs results/supplementary/data/rewardbench.jsonl \
  --cache results/supplementary/cache/core20 \
  --split-manifest data/split_seed20260818.json \
  --output results/supplementary/data/core20.npz

.venv/bin/python supplementary/exp00_response_cache/run.py merge \
  --manifest supplementary/exp00_response_cache/independent_judges8.json \
  --pairs results/supplementary/data/rewardbench.jsonl \
  --cache results/supplementary/cache/judges8 \
  --split-manifest data/split_seed20260818.json \
  --output results/supplementary/data/judges8.npz

.venv/bin/python supplementary/exp00_response_cache/run.py merge \
  --manifest supplementary/exp00_response_cache/nli4.json \
  --pairs results/supplementary/data/nli.jsonl \
  --cache results/supplementary/cache/nli4 \
  --output results/supplementary/data/nli4.npz
```

原 `data/split_seed*.json` 是 checksum 摘要，脚本依据记录的 seed 重建原 grouped-stratified assignment 并核验。如果历史数据已变，会报 mismatch；不能强行宣称原划分已复现。也支持 `prepare` 输出的显式 assignment。重划分需要分别 merge 成不同 NPZ，实际概率缓存可复用。

完整 archive 至少包括 JSONL / 数据重建 revision、锁定清单、所有 score blocks、NPZ、inventory/checksum、运行环境版本以及补实验 JSON/图表。请保留所有端点；cache 不全时 analysis 不能启动。当前脚本不自动上传发布这些产物。
