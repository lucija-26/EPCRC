# 固定版本模型预检

`preflight_models.py` 使用清单中的完整 commit SHA 检查模型元信息、实际权重 HEAD 和 tokenizer，生成 `model_preflight.json`。不下载权重正文、不启动 GPU；报告不含 token 或签名下载 URL。迁移包不包含旧预检结果，也不假设当前账户具有模型访问权限。

使用配置好的 GPU 环境执行 `.venv-gpu/bin/python supplementary/runtime/preflight_models.py`。当前账户须自行获得受限模型访问权；缺失端点不能从完整生态中静默删除。以下为原固定软件/模型版本的兼容性记录，供复现时核查。

## 原始 scorer 的兼容性

- Transformers 4.57.6 已将 `gemma3` 同时映射到 causal-LM 与 image-text AutoModel 的 `Gemma3ForConditionalGeneration`；就已安装版本的映射而言，不需要换加载器。
- OLMo2 原 7B 和新增 1B 的配置均声明 bfloat16。该版本 `from_pretrained` 接受 `dtype` 参数，OLMo2 实现支持 `logits_to_keep`。实际 GPU 加载、显存与数值正确性仍需 smoke 验证。
- 原预检记录中，J13 Phi-3.5 的 ` A/B/C` 标签各有两个 token，走原 scorer 的逐样本多 token 路径；不能假设所有端点均使用单 token 标签。
- J11 Phi-4-mini、J12 Phi-4、J13 Phi-3.5 的独立标签编码，与同一标签拼到 assistant chat template 后整体编码的续接 token 不同。原 scorer 定义的是附加独立标签 token ID 的协议；为复现原协议应保留，并记录这个差异。若改为上下文编码，须另标记协议版本，不能混用缓存。
- J15 Mistral-Nemo 的固定 tokenizer 在 4.57.6 下提示 regex 修正警告。启用 `fix_mistral_regex=True` 会改变分词，应作为显式协议变化评估，不能静默用于原协议复现。
- NLI4 的四个 manifest 类别置换均与固定配置中的语义标签一致。

`preflight_models.py` 可在具有 `transformers==4.57.6`、`huggingface_hub==0.36.2`、`sentencepiece`、`protobuf`、`jinja2`、`requests` 的隔离环境复跑。HEAD 检查成功不替代推理 smoke，也不证明 benchmark 输出正确。
