# 审稿补实验源码

这里包含针对审稿意见新增的缓存采集、五组分析实验、统计审计、运行工具及绘图源码。每个实验独立一个子目录，复用仓库原有 `epcrc/` 和 `data/split_seed*.json`；不需要原工作目录。

本目录是源码迁移，不包含实际响应缓存、数据集、模型权重、实验结果或机器运行状态。结果需重新生成或从独立结果包恢复；不能将 `--demo` 产物作为实证结果。现有缓存中的输入、模型版本与协议哈希必须保持一致，不能为绕过校验修改元数据。此次迁移的范围和验证见 [VALIDATION.md](VALIDATION.md)。

## 入口

| 目录 | 用途 |
|---|---|
| [exp00_response_cache](exp00_response_cache/README.md) | 锁定 core20、judges8、NLI4 的模型版本，准备数据、采集和合并完整缓存 |
| [exp01_dependency_significance](exp01_dependency_significance/README.md) | 独立与联合移除的配对统计、依赖矩阵和组合界；另含固定三模型五切分审计 |
| [exp02_reconstruction](exp02_reconstruction/README.md) | 相同支持上的四类重建器，以及冻结支持后的收敛审计 |
| [exp03_ecosystem_replication](exp03_ecosystem_replication/README.md) | 多生态来源、模型重叠审计和跨生态汇总 |
| [exp04_selection_benchmarks](exp04_selection_benchmarks/README.md) | QR 等同预算对比、枚举检查、冻结损失的区间精化 |
| [exp05_measured_cost](exp05_measured_cost/README.md) | 实测端点服务工时、成本–保真度和摊销 |
| [runtime](runtime/README.md) | 持久流水线、队列、完整性验证、只读进度和通用绘图 |
| [figures](figures/README.md) | 原先位于结果目录的三份专项绘图脚本 |
| `common.py` / `tests/` / `tools/` | 统一数据协议、共享分析、测试与合成端到端检查 |

## 环境与验证

从仓库根目录执行，Python ≥3.10，迁移验证使用 Python 3.12。分析只需 CPU：

```bash
python -m venv .venv
.venv/bin/python -m pip install -r supplementary/requirements.txt
.venv/bin/python -m pytest -q supplementary
.venv/bin/python supplementary/tools/smoke_all.py
```

`environment.freeze.txt` 保存原分析环境的版本清单。GPU 依赖单独列在 `runtime/requirements-gpu-requested.txt` 和 `runtime/requirements-gpu-frozen.txt`，其中 PyTorch 带 CUDA 12.8 构建；应使用独立 GPU 环境。运行器的 `--python` 和 `--cpu-python` 可指定已有环境，默认分别为仓库下的 `.venv-gpu/bin/python` 和 `.venv/bin/python`。若在仓库根目录创建 `.venv-gpu`，请将其加入本机 `.git/info/exclude`，避免提交环境目录。该目录不随源码复制。

模型版本来自 JSON 锁定清单，不自动更新到 main。受限模型需运行者自己的有效访问权限；凭据不随源码复制。GPU 访问和分配由运行者显式指定：

```bash
# 默认只检查当前主机，不连接其他节点。
.venv/bin/python supplementary/tools/check_gpus.py
# 可选远程节点：替换 SSH 目标、端口和私钥路径。
.venv/bin/python supplementary/tools/check_gpus.py \
  --host user@gpu-host --ports 22 --identity /path/to/private-key
```

合成检查写入已被根目录 `.gitignore` 忽略的 `results/supplementary/smoke/`。各实验的 `outputs/`、GPU 快照、预检报告和锁文件另由本目录 `.gitignore` 忽略；模型清单和配置模板仍可提交。

## 建议优先级

| 优先级 | 核心问题与实验 | 应交付的证据 | 对应审稿人 |
|---|---|---|---|
| P0 | **独立 vs 联合移除的统计差值 + 定理 2 实测**（exp01） | J07/J08/J11 的 individual/joint TV 与成组配对区间；依赖矩阵 A、rho(A)、H=(I−A)^−1、Hε、真实组合误差与直接重拟合误差 | msmm Q1/Q2；Kdfu 理论未使用 |
| P0 | **重建器敏感性**（exp02） | 相同保留集合下，全局凸组合、上下文凸组合、温度校准、输入依赖门控的 TEST k–TV 曲线；gamma=.12/.20 下可行预算、removed-only verdict agreement、开销 | W23h Q2；msmm Q3；Kdfu 实用性 |
| P0 | **独立组建的生态 + 不同接口**（exp03） | 原 core20、模型 ID 无重叠的新 8-judge panel、4 个 NLI 分类器，各自报告 gap、frontier 和没有出现 gap 的情况 | W23h Q1；msmm 泛化 |
| P1 | **coverage 相对 QR 的增益及代价**（exp04） | 同 k、同重建器、同 TEST 下 paired ΔTV / 区间 / CPU 时间；小实例完成枚举后才报告经验最优差距 | Kdfu Q2 |
| P1 | **真实成本与完整缓存**（exp00 + exp05） | 固定模型/数据版本、逐项概率/提示哈希/输入长度、实测 warm inference 时间、重建开销、成本–误差曲线、摊销盈亏平衡 | W23h Q3；Kdfu 实际价值 |

若只能补三组实证，优先 exp01、exp02、exp03；成本计时在生成缓存时顺手采集，exp04 则复用缓存。不要把主要算力花在继续证明很小的有限实例中 2-swap 更接近最优；原稿已有 59/60 的结果，这不能解决只有一个生态或缺乏实际节省的问题。

## 如何解释结果

1. 原图 .208–.216 相对 .20 的超出只有 .008–.016。五个重叠划分不是五个独立实验。需要对**源任务**做 paired cluster bootstrap，所有 comparison pairs、contexts 和方法一起抽样。给 individual、joint、joint−individual 和 joint−gamma 的区间，不只给跨 seed 均值。
2. 定理 2 是**充分条件**。Hε 超出阈值不能证明失败，rho(A)=1 也不能证明共同支持下不存在其他替代。实验能展示证书的依赖和误差放大，以及该证书能否安全组合；算法仍可选择直接 refit。若界很松，论文必须承认理论的诊断作用与限制。
3. 18/20 是指定响应接口、固定凸权重、容差与目标下的结论，不能直接称模型生态普遍难以压缩。exp02 如果明显改善，应改写为“固定重建族的限制”；如果没有改善，也只能对测试过的替代族下结论。
4. 同一 20 个模型换 JudgeBench 是 benchmark shift；重抽子面板或重划分不是独立生态。新 8-judge panel 与原 panel 没有完全相同的模型 ID，但仍共享 Qwen/OLMo/Falcon 等家族。NLI panel 检验不同接口，只有四个模型，且仍为三分类概率接口；这不能支持 token distributions/embeddings 等广义接口的普遍结论。
5. 保留端点的误差为零，应分开报告 removed-only 指标。约 25% 的被移除端点 verdict 翻转不能被全生态平均数稀释。允许如实得到“严格保真收益小”“QR 相当”“更复杂重建仍失败”。
6. k 减少 50% 不代表推理成本减少 50%。现有 exp05 是**实测单端点工作量的加总核算**，加上实测重建 CPU 开销；不等于测过线上队列、并发和网络后的端到端延迟。后者需要部署环境下另外测量，不能用 sum/max 的估计冒充。

Kdfu 关于可读性、与 convex-hull/archetypoid selection 的区别，还需要修正文稿。实验不替代这部分：明确主贡献是“哪些端点能同时替换及何时有效”，搜索器是否胜过 QR 由结果决定；不要承诺所有生态都大幅压缩。

## 统一 NPZ 数据协议

所有分析 CLI 明确要求 `--input panel.npz` 或 `--demo`。真实输入缺失或无效时直接失败，绝不回退到 synthetic。

| 字段 | 形状 / 说明 |
|---|---|
| probabilities | `[contexts, items, endpoints, 3]`，有限、非负、每行和为 1 |
| groups | `[items]`，源任务 ID；所有上下文和派生 comparison pair 保持一致 |
| pair_ids | `[items]`，唯一、与每个模型缓存完全对齐 |
| judge_ids / model_ids | `[endpoints]`，端点名 / 精确模型仓库名 |
| context_names | `[contexts]` |
| split | 可选 `[items]`，FIT / CERT / TEST；同一组不许跨 split |
| gold | 可选 `[items]`，统一坐标系下的 0/1/2 |
| metadata_json | JSON 字符串，含版本、输入哈希、来源、原始 cache inventory、synthetic 状态 |

未提供 split 时按源组确定性划分 50/25/25；使用原论文划分时，在 exp00 merge 传入原 `data/split_seed*.json`，加载器重建并检查其 checksum。已有固定 split 时更换 `--seed` **不会**重新分配：可重新 merge 对应 seed 的原 manifest，写到不同 NPZ；固定三模型审计也可用 `fixed_case.py --split-manifests ...` 明确校验并覆盖存储划分，保留每份分组哈希。`--bootstrap-seed` 只影响重采样，不改变划分。五个重叠切分只有预先指定的主划分作主分析，其余作敏感性分析，不汇总成五次独立重复。

NLI 统一坐标为 entailment / neutral / contradiction；不同模型原生 label 顺序不同，清单保存 permutation 和官方 config 来源。原 judge 坐标为 A / B / tie，order-swap 后统一还原。概率保真不等于校准到人类偏好；温度校准实验保留原端点为目标，不能把平滑后的目标偷偷改称原问题。

完整缓存应随补充材料提供或放可访问下载位置，并带 manifest/hash、逐条概率、prompt hashes、模型 revision、运行环境、split 清单。将受数据集许可约束的原输入与可分发的衍生产物分别列明。此目录没有替用户上传或发布任何文件。
