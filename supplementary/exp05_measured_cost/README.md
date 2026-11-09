# 实验 05：实测成本与重建保真度

回应 W23h 的实际推理成本问题，也检验 Kdfu 所说的实际收益有限。比较相同 k 下 backward、QR、最低实测耗时选择、按“FIT 误差增加 / 节省服务时间”逐步删除的 cost-aware 选择。输出最差 endpoint/context TV、被移除 endpoint 的 verdict agreement、实测服务工作量节省、CPU 混合开销和估计回本请求数。

```bash
# 软件检查：这些 endpoint 耗时是人为生成的，不是 GPU 实测结果。
.venv/bin/python supplementary/exp05_measured_cost/run.py --demo --ks 2 4 6 --timing-repeats 20 --output supplementary/exp05_measured_cost/outputs/demo.json

# 真实数据：必须使用实验 00 score 生成的原始块和 merge 生成的对应 NPZ。
.venv/bin/python supplementary/exp05_measured_cost/run.py --input /absolute/panel.npz --timing-cache /absolute/raw_cache --ks 4 8 10 12 16 18 20 --gamma .20 --output supplementary/exp05_measured_cost/outputs/real.json

# 如有事先明确的实际请求分布，可加 --context-weights /absolute/context_weights.json。
# 文件内容为所有 context 的名称到概率的 JSON 映射，非负且和为 1。
# 有真实租赁单价时再加 --gpu-hour-rate <每张GPU每小时单价> --requests <预计请求数>。
```

先在相同 GPU 型号/显存、torch/transformers 版本、batch size、精度和截断设置下测量各 endpoint。实验 00 应使用至少三次 `--repeats`，保留 warmup 和完整逐批计时；换计时设置时用新缓存目录。脚本检查缓存 SHA-256 必须与 NPZ inventory 一致，模型和响应轴一致，所有 endpoint 收到同一 context 下相同的输入指纹，每个 repeat 恰好覆盖全部有序输入。缺项、重复批次、NaN 时间、不同硬件等会报错，不能用参数量补齐。跨主机仍会记录标记；相同 GPU 不能消除主机 CPU、负载和调度差异，正式成本对比最好在相同空闲节点复测。

`seconds / items` 衡量 warm serialized 服务工作量：包括 tokenizer、device inference 和回传 CPU。它不是单请求延迟，更不是纯 GPU kernel 时间。脚本同时保留原始 batch 时间 p50/p95，但不能把它当成生产服务 p95 latency。完整生态的 additive cost 是各 endpoint 服务时间之和；保留生态减去被移除 endpoint 的测量成本，并单独测 CPU 概率混合。所有模型都保留时走 identity，无重建成本。

默认在线请求均匀来自各 context；`--context-weights` 可指定实际混合。成本按该分布加权，保真度依然取所有 context 的最差值，不能用低权重掩盖失败条件。完整缓存的采集成本按每个 context 实际执行一次计算，不会随未来请求分布重新加权。

选择和拟合只读 FIT。每个方法在预先固定的 k 网格中，用 CERT≤gamma 选实测保留成本最低的点；TEST 只评估这个冻结选择，不能用 TEST 选预算。两种主要回本数分别为：一次完整缓存的估计服务成本 / 每请求节省服务工作量，以及加上 CPU 选择拟合时间、减去在线 CPU 混合开销后的串行时间估计。还报告已记录全部 repeats + warmups 的服务成本对应的回本下界；模型加载、下载和其他未记录费用会使真实回本更晚。

可选价格只按用户给定的每张 GPU 小时单价估算独占单 GPU worker 服务时间的费用，不猜测云价格。`estimated_gpu_cost_net_after_collection` 扣除一次完整缓存采集的估计成本，没有给 CPU 拟合/混合定价，也不是实际账单。使用付费 API 时需另加真实 token/请求计费账本，当前工具不把本地 GPU 时间冒充 API 价格。

本实验提供可复查的成本核算和 CPU mixing benchmark；没有测到并发服务的端到端延迟、网络、排队、模型驻留/冷启动、能源、电费或 nonlinear gate 开销，因此不对这些指标做节省声称。JSON/CSV 保留完整限制说明和来源，真实结论需填入实际响应缓存后运行。
