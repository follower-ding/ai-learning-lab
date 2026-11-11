# 岸港酒店：演示与评测

这一层把已经写好的检索、工具循环、评测和追踪串成能反复跑的入口。它不是新的一课，也不进 `make lessons`。

默认路径不需要 API 密钥，也不访问网络。

## 这里有什么

- `demo.py`：四条固定案例，一次跑完。每条先打印检索或工具过程，再打印给客人的回答，然后用 `emit_trace` 打出六个字段。
- `eval.py`：调用现有的 `run_eval()`。不传客户端时仍是第 08 课的 Mock。报告用 `format_report`，再为每一条附上一张追踪。
- `ask.py`：只问一句。`--rag` 或 `--agent` 由你选，默认检索。这里不按问法自动分流。
- `loop.py`：上面三个入口共用的过程打印、追踪和可选的 `--live`。

## 怎么跑

在仓库根目录：

```bash
python apps/hotel-rag/demo.py
make hotel-rag-demo

python apps/hotel-rag/eval.py
make hotel-rag-eval

python apps/hotel-rag/ask.py "入住时间是什么时候？"
python apps/hotel-rag/ask.py --agent "取消政策是什么？"
```

`make hotel-rag-demo` 和 `make hotel-rag-eval` 没有接进 `make lessons`。

离线时，演示应当走完四条并以退出码 0 结束。评测应当打印 `合计 9/9`。

应用评测是另一份题，不替代上面的 9 条：

```bash
python apps/hotel-rag/app_eval.py
make hotel-rag-app
```

`make hotel-rag-app` 也没有接进 `make lessons`。它没有 `--live`，不访问网络，也不打六字段追踪。

要留下文件时加上 `--trace-dir`。追踪先出现在标准输出，文件是同一段已经脱敏的文字。不写这个参数就不会在磁盘上留副本。

```bash
python apps/hotel-rag/demo.py --trace-dir /tmp/hotel-rag-traces
```

## 四条演示

问法和路径与第 08 课的这四条相同。工具循环保持原来的默认，不打开缺槽先问。

1. 入住时间是什么时候？检索。引用里有 `faq-checkin`。
2. 游泳池几点开放？检索没过线，拒绝原因是 `low_score`。客人看到的仍是那句固定拒绝。追踪的拒绝原因字段只写 `low_score`，不抄那句给客人的话。
3. 延迟退房 3 小时要多少钱？调用 `quote_late_checkout`。回答里有 90。
4. 取消政策是什么？调用 `lookup_policy`。回答里有 24。

## 应用评测

`app_cases.json` 是给这个入口用的题，至少 12 条，id 不能重复。题面来自现有规章和两只工具的边界，分成有据、无据、改写、工具边界。每条都有 `basis`，写明预期对应哪条资料或哪条工具规则。

`app_eval.py` 对每一题调用一次现有的 `run_eval([case], client=meter)`。评分仍是原来的 `_grade`，这里不复制一套。`meter` 包住离线 Mock 的 `complete`，签名不变，返回原来的 Completion。`calls` 是尝试次数，进入调用之前加一。只有成功返回的整数 token 才累加。某次尝试没有返回时，整题标 `unknown`，不印数字，也不把中途小计当成完整用量。耗时只包住这一次 `run_eval`。

低分拒绝不会调用模型，`calls=0` 且 token 为 0、`token_source=estimated`。这是 Mock 没被调用，不是供应商 usage 的 0。工具要先调用再回答时，同一次里成功返回的多步 token 加在一起。某一题抛出异常时，这一行记成 `status=error`，不算通过。

Mock 的 token 来自 `estimate_tokens`，报告里标 `token_source=estimated`。这不是供应商 usage，也不是账单。现有 Completion 证明不了真实用量；字段不是整数，或客户端不是这个 Mock 时，标 `token_source=unknown`，并且不把数字印出来。

这不是语义检索的生产效果，也不是把所有秘密都滤掉的过滤器。报告和用例集错误会走现有的 `redact`，它仍然只有原来的两种样式。

字面向量对不上的改写，预期仍按关联写。对不上就在报告里记失败，不把预期改成拒绝来凑全过。`app-paraphrase-floor` 只检查会不会引用停车资料，正文并没有写车位楼层，引用成功也不证明楼层。

退出码：

- 0：全部通过。
- 1：有失败或错误。失败的题照实列在报告里。当前这套字面向量下，`app-paraphrase-floor` 和 `app-paraphrase-wifi-name` 会失败，所以直接运行脚本时退出码是 1。
- 2：用例集不合法（重复 id、kind 和 expect 不配、require 与 forbid 重叠等）。还没有开始评分。

`make hotel-rag-app` 只是调用这个脚本。脚本返回 1 时，make 按自己的惯例以 2 结束。make 的 2 不是「用例集不合法」；要看脚本打印的合计。

课程用的 `eval_cases.json` 仍是原来的 9 条。`DEFAULT_MIN_SCORE` 和 `run_eval()` 不传客户端时的行为都不改。

## 评测

`eval.py` 在离线时调用 `run_eval()`，不传入客户端。用例仍是 `eval_cases.json` 里原来的 9 条，这里不增删。

追踪是同一条路径再走一次，用来把六个字段打出来。离线 Mock 两次结果相同。真实接口如果答得不稳定，报告和追踪可能不是同一次生成。

## 可选的真实接口

加上 `--live` 时，只用 `live_client_from_env`。没有 `OPENAI_API_KEY` 就打印「跳过真实接口」，退出码为 0，不构造 HTTP 客户端，也不把 Mock 记成真实结果。

超时和 `max_tokens` 与第 15 课相同：没写时是 30 秒和 400。报错文本走已有的 `redact`。这次对照不要求真实模型也得 9/9。

默认的 `python -m pytest` 不带 `--live` 去访问网络。没有密钥时，第 01 课那条标记为 integration 的测试仍会跳过。

## 这一层不改什么

- 不改 `src/ailab` 的默认路径和函数签名。`RagPipeline.answer`、`run_agent` 的默认参数、不传客户端的 `run_eval()`、工具允许列表、`DEFAULT_MIN_SCORE` 都保持原样。
- 不改 `hotel_faq.json`、`hotel_policy.md`、`eval_cases.json` 和 `redact`。
- 检索仍用现有的字面向量（`build_lab_rag` 和现有的问答索引）。不读 `OPENAI_EMBEDDING_MODEL`。不把混合检索、重排序、流式输出或夹具注入变成这一层的默认路径。
- 不改 `lessons/`，也不新增第 16 课。
- 没有网页界面，也没有 Docker。
