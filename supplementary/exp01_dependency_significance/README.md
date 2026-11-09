# 实验 1：独立移除与联合移除的差异、统计不确定性、依赖界

直接回应 **msmm 的问题 1/2** 和 **Kdfu 关于定理 2 没有实验用途的问题**。无需新推理：有原始逐项概率缓存即可在 CPU 上运行。只有合成数据时，输出仅用于代码验证，不能作为论文证据。

```bash
.venv/bin/python supplementary/exp01_dependency_significance/run.py \
  --input /absolute/path/panel.npz --gamma 0.20 --bootstrap 5000 --plot \
  --output supplementary/exp01_dependency_significance/results_real.json

.venv/bin/python supplementary/exp01_dependency_significance/run.py \
  --demo --bootstrap 200 --output /tmp/dependency_demo.json
```

数据格式见上级目录的公共数据说明。运行一次固定的主划分；不同 `--seed` 只作敏感性分析，不能把重叠的五个划分视为五个独立样本做 t 检验。

## 固定协议

1. 在 **FIT** 上拟合每个模型的 leave-one-out 凸组合，组成零对角、行随机矩阵 `W`。
2. 在 **CERT** 上用预定阈值 `gamma=0.20` 固定可移除集合 `R`。不使用 TEST 选择模型或阈值。
3. 固定保留集合 `S`，在 **FIT** 上直接重新拟合对所有 `R` 的替代权重。
4. 在 **TEST** 上逐模型报告：独立移除 TV、联合移除 TV、二者差值，以及各上下文均值。报告 `max_context` 和 `max_judge,max_context`，避免平均值掩盖失败。
5. 按 `base_item_id` 做 paired cluster bootstrap，同一基项的所有 comparison pairs、全部上下文、所有模型和两种方法一起重采样。组大小不同时，保留论文的 pair-weighted 均值，而不是擅自改成 group-weighted 均值。

主不确定性输出是两种方法 × 全部上下文 × 被移除模型的近似同时置信带，再由该带推导最大误差及差值的区间。由于 max 统计量在并列最坏上下文处不光滑，paired percentile 差值区间只标记为辅助描述。默认 `alpha=.05`，不报告把 bootstrap 抽样比例误叫作 p 值的数值。

严格的“测试集支持不具可组合性”条件同时要求：所有独立移除的同时置信上界不超过阈值，联合移除的同时置信下界超过阈值。只满足联合误差大于阈值，还不足以证明同一测试分布上的全部独立移除均可接受。`positive_gap_by_simultaneous_band` 是另一个更弱的差异结论。

这些区间是在固定 FIT/CERT 决策条件下的渐近 bootstrap 区间，需要基项组之间独立。它们不是有限样本、分布无关证书；不能因未显著就解释为没有差异。

## 定理 2 的可检验量

把 `W` 中被移除模型对被移除模型的部分写作 `A=W[R,R]`，对保留模型的部分写作 `B=W[R,S]`。

- 输出完整 `A`、内部替代权重行和、谱半径 `rho(A)`、`I-A` 条件数。
- 若 `rho(A)<1`，计算 `M=(I-A)^(-1) B`，检查其概率约束。它可以直接部署，只调用保留模型。
- 按论文定理 2，先取每个模型的最坏上下文独立误差 `epsilon`，计算 `(I-A)^(-1) epsilon`。另报告更紧的逐上下文界：对每个 `epsilon_c` 先算 `(I-A)^(-1) epsilon_c` 再对上下文取最大值。两者均与实际组合替代误差及直接重新拟合误差对照。
- 若存在闭合依赖循环或数值不稳定，明确报告不适用，不强行使用伪逆制造“可部署”替代。

加 `--plot` 会输出每个移除模型的 individual / direct-refit joint / composed 误差以及定理上界，旁边画 `A` 的热图，同时保存 PNG 和 PDF。合成数据图明确标注为软件验证。导出的 JSON 保留所有绘图数据及权重。

**解释边界：** 定理界是充分条件。上界超过 `gamma` 只能说该界无法保证成功，不能证明联合移除必然失败；`rho(A)=1` 也不能证明其他直接拟合的替代不存在。依赖图取决于拟合得到的证书，LP 多解时可能有不同图。这个实验把理论与观察联系起来，但不把相关图误称为因果解释。

**与原图的关系：** 原实验可能已经在 CERT 上报告选择结果。本实验另用 TEST 审计新冻结的选择，数值和选中的三个模型不保证完全复现原图。拿到原始缓存后应同时提供原图的描述性逐模型误差，并把本实验的独立验证清楚分开。

## 输出与检查

`results_real.json` 包含数据来源元信息、模型/上下文顺序、完整 LOO 和联合权重、CERT 选择依据、TEST 逐模型与总体统计、依赖矩阵和三个划分上的理论界。元信息继承公共加载器，因此 demo 与真实数据可区分。

```bash
.venv/bin/python -m pytest -q supplementary/exp01_dependency_significance/test_dependency.py
```
