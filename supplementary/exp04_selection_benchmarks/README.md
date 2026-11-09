# 实验 04：coverage 搜索到底比 QR 多带来什么？

对应 Kdfu 的核心质疑。对同一个物理模型预算 k，比较 backward、forward、pivoted QR、k-medoids、随机选择、backward 初始化的 1/2-swap。所有选择只读 FIT；全部方法使用相同的 global-simplex minimax 重建器，FIT 权重冻结后评估 CERT/TEST。

这里的 swap 是固定 k 的交换，最多接受四轮改进，不保证达到局部最优；与原论文减少 cardinality 的交换过程不同，输出将其作为新增等预算基线。计时按每个 k 的统一 API 调用记录；原库 QR/k-medoids 每次会构造整个 k-chain，因此不应把这些时间当作经过缓存/批量优化后的 frontier 运行效率排名。

```bash
.venv/bin/python supplementary/exp04_selection_benchmarks/run.py --demo --k 2 3 --bootstrap 200 --random-repeats 2 --output supplementary/exp04_selection_benchmarks/outputs/demo.json
.venv/bin/python supplementary/exp04_selection_benchmarks/run.py --input /absolute/panel.npz --k 4 8 10 12 16 18 --bootstrap 5000 --random-repeats 20 --output supplementary/exp04_selection_benchmarks/outputs/real.json
```

输出 JSON、平行 CSV、逐 item 的 TEST TV 损失 NPZ。主指标是整个原始生态中最差 endpoint、最差 context 的平均 TV，保留的 endpoint 损失为零。另报被移除 endpoint 的 verdict agreement、选择耗时、重建/评估耗时，以及条件于冻结模型的 paired base-item cluster bootstrap。每次 bootstrap 对所有 context/endpoint/方法使用同一组 base-item 抽样，按原始 pair 加权，重新计算最差 endpoint/context；不将 context 当作独立样本。

主要看同 k 的 TEST TV 与 backward 的差值及区间，负值表示该方法更好。报告 pointwise 95% 区间及跨 k/方法的 Bonferroni percentile 区间；尾部抽样数量不足会给 `tail_resolution_warning`，应增加 repetitions。随机重复不是独立数据集，不能汇总成独立样本 t-test。

N≤10 默认枚举所有大小 k 的子集，最多 10,000 次。只有所有组合都完成时 `exact_search.complete=true`，才计算 FIT objective gap；超过预算只报 best observed，绝不称为 optimum。更大生态的 exact audit 应事先固定小子面板，并显式写成子面板实验，不能当作独立生态。脚本报告的是原始 solver 数值精度下的经验最优，不是 population 最优。计时为本机 CPU wall time，含共享库调用，不包含原始 GPU 推理。

五个原论文 split seeds 是敏感性分析，不能将它们当作五次独立复制。为新增独立生态重复本实验时使用不同 `--input` 和 `--output`，保留各自结果。预先冻结 k、方法和主要比较；不要看 TEST 后选择有利 k。若 QR 相当，诚实报告适用条件，论文贡献应落在 joint coverage 的问题定义/失败诊断，而不是声称普遍更强的搜索器。
