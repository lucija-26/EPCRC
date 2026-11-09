# 实验 02：重建方法是否限制了压缩率？

对应 W23h 的 context-dependent/nonlinear replacement 问题，以及 msmm 关于 calibrated / probability-preserving alternative 的问题。核心比较是在**相同保留集合**上，仅改变重建映射，避免把选择算法差异误认为重建收益。

| 方法 | 可变部分 | 输出约束 |
| --- | --- | --- |
| `global_convex` | 每个被移除端点只有一个权重向量，跨样本/上下文共享 | 原论文 simplex minimax TV |
| `context_convex` | 每个已登记上下文独立的 simplex 权重 | 每个上下文仍为凸组合 |
| `temperature_convex` | 原始 convex mixture 经 `softmax(log(p)/T)` 校准 | 合法概率，可能超出保留预测的凸包 |
| `nonlinear_gate` | 基于保留端点概率的 softmax 门控；权重随样本变化 | 每个样本仍为凸组合 |

非线性门控输入只有保留端点的概率；推理时不会调用被移除端点。门控通过 SciPy L-BFGS-B 拟合平滑的 worst-context TV 目标；使用解析梯度，无 GPU 依赖。`context_convex` 需要推理时知道已登记上下文，不能解释为对未知上下文的泛化结果。温度校准对齐的是原端点分布，**不是**重新定义为人类标签准确率。后三种方法不直接继承论文针对固定全局凸权重的全部理论保证。

## 划分与统计

1. 保留集合只由 FIT 选择，默认 backward coverage；每个预算的四种重建共用该集合。
2. 全局/context 权重、门控系数、门控特征标准化只使用 FIT。
3. CERT 从预声明的温度网格和门控正则化网格选择每个端点的参数。
4. TEST 输出 worst endpoint/context mean TV、各端点/上下文 TV、被移除端点的 verdict agreement、概率合法率。以 base item 为簇做 paired bootstrap；同一抽样在全部上下文和方法间共享，且每次重新取最大风险。
5. 在给定 gamma 下，用 CERT 选择测试过的最小预算，并报告该冻结选择在 TEST 上的误差。另报各方法的 FIT 风险和观测预算，用于对照原文的 fitting frontier。它们都是离散预算/共享集合路径上的结果，不能称为所有可能集合上的精确最小保留数。

Bootstrap 区间是给定 FIT/CERT 拟合结果的 TEST 条件区间，没有重新训练。多个 split seed 共享同一批原始样本，不能当作独立样本重复做 t-test。调参诊断会保留每次优化的收敛标记和报错信息；未收敛候选不会被伪装为精确解，正式报告应单列并复核。

## 运行

从项目根目录运行：

```bash
.venv/bin/python supplementary/exp02_reconstruction/run.py \
  --demo --budgets 2 4 --maxiter 30 --bootstrap 50 \
  --output results/supplementary/exp02_demo

.venv/bin/python supplementary/exp02_reconstruction/run.py \
  --input /absolute/path/panel.npz --budgets 4 8 10 12 16 18 19 20 \
  --gammas 0.12 0.20 --maxiter 150 --bootstrap 2000 \
  --output results/supplementary/exp02_real

.venv/bin/python -m pytest -q supplementary/exp02_reconstruction/test_reconstruction.py
```

输入格式见 `supplementary/README.md`。默认运行全部四种方法；可以 `--selection pivoted_qr` 做另一条共享集合路径的稳健性检查。全 20 端点下 backward 搜索可能比重建器拟合更耗时。程序每完成一个预算写入 checkpoint；不要根据已经看到的 TEST 结果修改后续实验配置。

## 产物与正文建议

- `results.json`：风险、paired 差异区间、CERT 选定预算、拟合诊断、子集选择/拟合/CPU 重建时间。
- `k*_test_losses.npz`：每种方法、每个上下文的逐样本逐端点 TV 与 base item 分组，可重新计算统计。
- `k*_maps.json`：所有拟合权重、温度、门控系数和标准化参数。

建议正文增加一张固定支持集的风险–预算图和一张 gamma=0.12/0.20 表。若更灵活的方法明显改善，明确原论文高保留率受重建类限制；若没有改善，只能支持在这些已测方法/生态中的稳健性。CPU 重建时间不等于 LLM 推理节省；真实推理成本由独立成本实验测量。`--demo` 是合成数据的软件检验，不能作为审稿回复的实证结果。

## 原20端点的统一迭代预算审计

`convergence_audit.py` 针对已完成的主运行与额外k=17运行，读取并校验原FIT支持和结果哈希，对全部八个预算、全部正则候选统一使用1000次门控迭代上限。它不重新选择支持，不能把支持读取耗时与从头搜索耗时相比较；原搜索耗时另行保存。示例输出必须是尚不存在的新目录：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python supplementary/exp02_reconstruction/convergence_audit.py \
  --input results/supplementary/real/core20/responses.npz \
  --source-runs results/supplementary/real/core20 results/supplementary/real/core20_k17 \
  --output results/supplementary/real/core20_convergence1000/exp02_reconstruction
```

审计验证三个非门控方法的拟合映射、各划分风险、条件区间和逐条TEST损失不变，保存`audit_protocol.json`与`audit_status.json`。只有状态为`complete`且结果SHA一致时，才算完成；中间的`results.json`可能只是checkpoint。

本次core20从150增至1000次后，所选门控未收敛数从32/54降至19/54，仍不能标成充分收敛。两个迭代预算下，gamma=.20的CERT选择均为global/context/temperature k=18、gate k=17；gamma=.12为20/19/20/19。这说明所测选择对该次预算增加稳定，不证明其他优化设置、未测支持或更高预算不会改变结论。judges8的独立1000次审计则全部所选门控收敛；两个生态应分开报告。
