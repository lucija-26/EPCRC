# 真实实验运行工具

所有命令从仓库根目录执行。脚本使用已有环境；`pipeline.py` 默认 GPU Python 为 `.venv-gpu/bin/python`，CPU Python 为 `.venv/bin/python`，可以用 `--python` / `--cpu-python` 覆盖。代码迁移不包含环境、数据或旧任务状态。

## 流水线

输入准备与模型清单见 [exp00](../exp00_response_cache/README.md)。先用 `--dry-run` 查看计划，正式运行去掉该选项。以下 core20 示例采用 batch 16、单次计时、bfloat16、8192 token 上限且禁止截断；其他生态必须使用各自固定协议。

```bash
.venv/bin/python supplementary/runtime/pipeline.py \
  --manifest supplementary/exp00_response_cache/core20.locked.json \
  --pairs results/supplementary/data/rewardbench.jsonl \
  --cache results/supplementary/cache/core20 \
  --split-manifest data/split_seed20260818.json \
  --output results/supplementary/real/core20 \
  --gpu 0 --batch-size 16 --repeats 1 --dtype bfloat16 --max-tokens 8192 \
  --budgets 2 4 10 16 18 19 20 \
  --relationship original_panel --benchmark RewardBench2-pinned --dry-run
```

GPU 编号是当前主机 `nvidia-smi` 的物理编号；子进程只看到选定的一张卡。流水线依次 score → 完整性检查 → merge → `verify_merged.py` → exp01…05。任一模型/context 缺失、指纹冲突或阶段非零退出都会停止并记录失败。`--allow-truncation` 必须在运行前明确选择并记录，截断接口不能等同于原始完整输入。

`--score-only --judges ...` 支持互不重叠的端点分片：共享完整 manifest 和缓存目录，每个 runner 使用不同 output。全部块齐全后，用同样评分协议参数执行 `--analyses-only` 合并、验证和分析；此阶段不占 GPU。`--experiments` 可选择 CPU 阶段。N≤8 默认测试全部预算 1…N；更大面板默认稀疏网格，可用 `--budgets` 预先固定。模型、输入、精度、batch size、repeats、截断设置或源码变动时使用新缓存/输出，不能混合不一致的块。

运行记录 `protocol.json`、`plan.json`、`status.json` 和分阶段日志。`--detach` 启动独立会话并返回 PID，写入 `launch.json` 与 `runner.log`。重试会验证旧块、输入/源码/结果哈希后续跑；GPU 检查会拒绝已有 compute processes 或显存不足的设备，不停止已有任务。主机/GPU 锁与端点锁属于协作式保护，不能代替集群调度器。

## 进度与队列

```bash
.venv/bin/python supplementary/runtime/progress.py \
  --manifest supplementary/exp00_response_cache/core20.locked.json \
  --cache results/supplementary/cache/core20 \
  --runs results/supplementary/real/core20
```

这是只读快照，显示缺块、PID/status、当前阶段和日志尾部；不能代替正式 merge 校验。`after_stage.py --wait-status PATH --stage score --output PATH -- COMMAND...` 可在指定阶段完成后执行下一命令；上游失败或超时都会记录并退出，队列自身不会自动 SSH。

`complete_core20.py` 保留原 core20 完成协调流程：检查 J07/J08 固定权重访问、等待完整缓存、分析默认预算 `[2,4,10,16,18,19,20]`。它会扫描当前主机 GPU 0/1/2，只接受通过进程/显存检查的卡。若需明确指定其他 GPU 或不同调度方案，直接使用 `pipeline.py`。协调器各路径与超时可通过 `--help` 查看；默认路径位于当前仓库 `results/supplementary/real/`。这些工具不表示当前存在后台作业。

## 可选 SSH 启动

`launch_remote.py` 适用于本地和远程具有相同绝对仓库/输出路径的共享工作区。远程需有 `python` 可运行启动器，后续命令可显式指定虚拟环境。必须指定 SSH 目标；端口和私钥可沿用 SSH 配置。示例中的主机、路径和命令需替换：

```bash
.venv/bin/python supplementary/runtime/launch_remote.py \
  --host user@gpu-host --port 22 --identity /path/to/private-key \
  --job-dir results/supplementary/real/remote_job \
  --without-hf-auth -- /path/to/cpu-python -u supplementary/runtime/pipeline.py --help
```

默认可读取本机标准 Hugging Face 配置并经 SSH stdin 传给远端进程环境，不写入命令行或工作区。无需传凭据时使用 `--without-hf-auth`；可选 `--hf-hub-cache /remote/cache/path` 覆盖远端缓存位置。代码没有内置用户名、私钥文件名、节点地址或个人缓存路径。

## 预检与图表

`preflight_models.py` 在 GPU 环境执行固定版本的权重 HEAD、config/tokenizer 检查，生成本目录下已被忽略的 `model_preflight.json`。它会访问网络；HEAD 成功不替代 GPU 推理检查。兼容性约束见 [MODEL_PREFLIGHT.md](MODEL_PREFLIGHT.md)。

`plot_results.py --run-dir PATH --output-dir PATH` 根据同一缓存对应的 exp02/04/05 真实结果绘图，校验来源，不重新拟合或编造区间。专项图入口见 [figures](../figures/README.md)。
