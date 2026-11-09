# 实验 03：独立组建生态中的复现

对应 W23h 和 msmm 的泛化质疑。至少新增一个、最好两个按不同规则独立组建的模型生态；在看响应之前固定模型注册表、checkpoint revision、选模标准、任务、prompt、划分和 gamma。一个可用于检验近亲模型冗余，另一个检验不同家族的多样性，但不能挑选最容易出现 gap 的面板再称普遍规律。

同一 core20 换 JudgeBench 只算 benchmark shift；删掉一部分 core20 只算 subpanel；五个 split seeds 只算同一数据集重划分。这些有用，但不能满足“independently assembled ecosystems”。本脚本强制记录 `relationship` 和 `assembly_protocol`，审计模型 ID 的交集；标成 independent 的面板若是另一面板的相同模型集合或真子集会被拒绝。其他部分重叠会如实输出 Jaccard，需要人工判断组建机制，代码不替代科学独立性的审查。

`ecosystems.json` 已填入原 core20、新 judges8、NLI4 的实际固定注册表、采样依据和预期 NPZ 路径；真实缓存生成后可直接使用。各生态的 `budgets` 允许分别指定适合 N=20/8/4 的 k 网格；命令行 `--k` 若提供则对所有生态覆盖。不要把设计时冻结清单称为经过外部平台注册的 preregistration。

```bash
.venv/bin/python supplementary/exp03_ecosystem_replication/run.py \
  --manifest supplementary/exp03_ecosystem_replication/ecosystems.json \
  --gamma .20 --bootstrap 5000 --output results/supplementary/replication.json
```

```bash
.venv/bin/python supplementary/exp03_ecosystem_replication/run.py --demo --gamma .2 .4 --k 2 3 --bootstrap 200 --output supplementary/exp03_ecosystem_replication/outputs/demo.json
# 先填写 manifest.template.json，input 相对于 manifest 的位置解析。
.venv/bin/python supplementary/exp03_ecosystem_replication/run.py --manifest /absolute/ecosystems.json --gamma .12 .15 .20 .25 .30 --k 4 8 10 12 --bootstrap 5000 --output supplementary/exp03_ecosystem_replication/outputs/real.json
```

每个模型的 LOO 权重在 FIT 拟合，CERT 决定 gamma 下哪些模型可单独移除；然后对共同保留集在 FIT 重新拟合，冻结后评估 TEST。主 gamma=0.20；其余阈值是探索性敏感性分析。直接报告每个移除模型的 individual/joint TV、joint-minus-individual 差值和按 base item 成组、跨 context 同步抽样的 paired bootstrap 区间，以及 TEST 上是否同时满足 individual≤gamma 和 joint>gamma。若 CERT 选的 individual 在 TEST 已超阈值，不能把这一行当成已验证的 individual-versus-joint 反例。

同时输出相同重建器下 k–TV frontier。选择在 FIT；CERT 只在预先固定的 k 网格上选可接受预算；TEST 是该决定的最终检查。`smallest_tested_k_accepted_on_cert` 不是全局最小 k，更不是总体保证。默认稀疏网格适合运行检查，正式 frontier 建议预先固定全部 k=1…N 或至少覆盖低、中、高压缩区间。所有已有/未出现 gap 的生态都要报告。

JSON 保存来源说明和输入 SHA-256；NPZ 保存每个 split 的逐 item TV 和 base-item group IDs，供复查。多个生态分别报告效果，不把它们的数据简单混成一个大 N，也不从少数生态估计“所有 LLM 生态”的概率。当前实现专用于三分类概率接口，不能作为 token distribution、embedding 或一般非判别输出的泛化证据。`--demo` 仅是程序执行检查，不算真实实验。
