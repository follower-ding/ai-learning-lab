# 岸港酒店：演示与评测

这一层把已经写好的检索、工具循环、评测和追踪串成能反复跑的入口。它不是新的一课，也不进 `make lessons`。

默认路径不需要 API 密钥，也不访问网络。

## 这里有什么

- `demo.py`：四条固定案例，一次跑完。每条先打印检索或工具过程，再打印给客人的回答，然后用 `emit_trace` 打出六个字段。
- `eval.py`：调用现有的 `run_eval()`。不传客户端时仍是第 08 课的 Mock。报告用 `format_report`。每一条追踪来自刚刚评分的那一次作答。
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

## 评测

`eval.py` 在离线时调用 `run_eval()`，不传入客户端。用例仍是 `eval_cases.json` 里原来的 9 条，这里不增删。

`run_eval` 多了一个可选关键字 `observe`，默认不传。不传时，评分、返回值和模型调用次数都和原来一样。传了以后，每条用例评分一结束，就把刚刚用过的那个结果对象交回来。这里用现有的 `trace_rag` / `trace_agent` 打出六个字段，不再为了补追踪重新调用模型或工具循环。因此报告和追踪是同一次作答，即便客户端每次回答都不一样。

追踪按用例 id 和这次评分对上。现在 `eval_cases.json` 里的 9 个 id 都不重复。这一层没有做重复 id：如果两条用例共用一个 id，字典只会留下后写入的那条。

## 可选的真实接口

加上 `--live` 时，只用 `live_client_from_env`。没有 `OPENAI_API_KEY` 就打印「跳过真实接口」，退出码为 0，不构造 HTTP 客户端，也不把 Mock 记成真实结果。

超时和 `max_tokens` 与第 15 课相同：没写时是 30 秒和 400。这次对照不要求真实模型也得 9/9。

标准输出、追踪文件，以及这个入口打印的异常，都走已有的 `redact`。工具名和工具参数也走这一道。`redact` 没有加规则，仍然只处理它原来的两种样式。走一遍不等于所有秘密都会被抹掉，这里也不是一层全面的安全过滤。

三个命令的异常由入口接住，不把 traceback 打到标准错误。这和直接调用库不一样：库函数仍然把原来的异常抛出去。

- 离线运行中途失败：打印脱敏后的异常文本（含异常链和附注），退出码为 1。
- `--live` 在客户端交出来之前失败：打印脱敏后的异常，写明没有补写结果，也写明这次没有调用网络，退出码为 0。
- `--live` 在运行或关闭客户端时失败：打印脱敏后的异常，写明没有补写结果，退出码为 0。运行已经发生过的输出仍留在屏幕上，这里不另编一份结果。
- 没有密钥：仍打印跳过真实接口，退出码为 0，不构造 HTTP 客户端，也不把 Mock 记成真实结果。

默认的 `python -m pytest` 不带 `--live` 去访问网络。没有密钥时，第 01 课那条标记为 integration 的测试仍会跳过。

## 这一层不改什么

- 不改默认路径。`RagPipeline.answer`、`run_agent` 的默认参数、不传客户端也不传 `observe` 的 `run_eval()`、工具允许列表、`DEFAULT_MIN_SCORE` 都保持原样。`run_eval` 只多了可选关键字 `observe`。
- 不改 `hotel_faq.json`、`hotel_policy.md`、`eval_cases.json` 和 `redact`。
- 检索仍用现有的字面向量（`build_lab_rag` 和现有的问答索引）。不读 `OPENAI_EMBEDDING_MODEL`。不把混合检索、重排序、流式输出或夹具注入变成这一层的默认路径。
- 不改 `lessons/`，也不新增第 16 课。
- 没有网页界面，也没有 Docker。
