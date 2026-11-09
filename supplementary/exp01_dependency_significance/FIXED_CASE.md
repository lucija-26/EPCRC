# 原论文固定三模型案例

`fixed_case.py` 专门核验论文 Figure 2a 的 **J07、J08、J11**，始终联合移除这三个模型，保留其余 17 个。它与 `run.py` 的“按当前 CERT 误差重新选择可移除集合”是两个不同问题，不能互相替代。

本脚本仅分析完整概率缓存，不调用 GPU。下面是真实缓存完整合并之后的运行接口；不应在缓存仍不完整时运行：

```bash
.venv/bin/python supplementary/exp01_dependency_significance/fixed_case.py \
  --input results/supplementary/real/core20/responses.npz \
  --pairs results/supplementary/data/rewardbench.jsonl \
  --split-manifests \
    data/split_seed20260818.json \
    data/split_seed20260819.json \
    data/split_seed20260820.json \
    data/split_seed20260821.json \
    data/split_seed20260822.json \
  --primary-index 0 --gamma 0.20 --bootstrap 5000 \
  --bootstrap-seed 20260818 \
  --output results/supplementary/real/core20/exp01_dependency_significance/fixed_trio_five_splits.json
```

每份原始 split manifest 实际是带 SHA-256 的摘要。脚本用原始 JSONL 中的 domain/tie/group 信息和该 manifest 声明的 seed 重建分组，核验样本数和三个 split 的 group hash，随后显式切分未分组的缓存张量。也支持 `{"seed": ..., "splits": {"FIT": [...], "CERT": [...], "TEST": [...]}}` 形式的显式组列表。缓存原有的 stored split 被明确替换，其是否与此次划分一致也会记录。`--bootstrap-seed` 只控制 bootstrap 随机数，绝不改变数据划分。

每个划分都重新在 FIT 上拟合三份 leave-one-out 权重和三份共同保留 17 个模型的联合权重。输出同时给出 FIT、CERT、TEST 的逐模型/逐上下文误差；CERT 与 TEST 各自按 base item 配对重采样，提供近似同时置信区间和辅助的 paired percentile 区间。固定案例即使不通过独立移除条件也仍然保留，禁止按显著性筛选结果。输入必须是原 core20 的全部模型，pair order 与 source group 也必须严格匹配。

`split_reports[i].splits.CERT` 对应原图使用的验证集统计；`splits.TEST` 是单独的冻结权重测试集审计。不能把 TEST 数字直接标成对原图 CERT 数字的复现。主结论只取 `--primary-index` 预先指定的主划分 TEST；另外四个划分共享原始样本，只能作敏感性分析，不能当五个独立样本合并 p 值。由于使用重采集的响应和已有公开 benchmark，这也不是新的前瞻性测试集。

最强的阈值结论看 `splits.TEST.bootstrap.overall.noncomposability_supported_by_simultaneous_band`：它同时要求独立误差上界不超过 0.20、联合误差下界超过 0.20。仅 `positive_gap_by_simultaneous_band` 为真不能证明阈值两侧均成立。CERT 区间作原固定案例的描述性诊断，不与 TEST 混用。

脚本还计算固定三模型的依赖矩阵、谱半径及可组合时的定理 2 上界。这里的上界是对冻结证书在当前样本上的充分界，不是必要条件或总体风险证书。应检查 `solver_warnings`、条件数和 `max_bound_violation_numerical`，并同时展示 direct-refit、composed 和 bound。

另一个必要的相邻预算检查是 **k=17**：当前 core20 计划为 `[2,4,10,16,18,19,20]`，若要回答“更灵活的重建是否将原来的 18 个减少至 17 个”，需在完整缓存上补这一预算。共享保留集合上的重建消融只给出观测前沿，不能称为每种替代函数类的全局最小模型数。

验证：

```bash
.venv/bin/python -m pytest -q supplementary/exp01_dependency_significance/test_fixed_case.py
```
