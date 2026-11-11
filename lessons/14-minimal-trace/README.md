# 第 14 课：最小追踪

第 08 课的失败，客人看到的是同一句「知识库里没有足够依据，不能回答。」只看这句话，分不清是检索没过阈值，还是引用检查把草稿扔了。

这一课在作答之后另记一张追踪。六个字段：问题、最高分、拒绝原因、工具名、步数、停止原因。先打印到标准输出。要留档时，再把同一段已经脱敏的文字写入文件。不打开文件也能看见这张记录。

## 学习目标

- 用一张追踪分开第 08 课的两种拒绝：`low_score`，以及引用检查作废草稿（`unknown_citation` 或 `missing_citation`）。
- 在工具路径上读出工具名、步数和停止原因。检索路径不走循环，这三项是空的或 0。
- 确认追踪里没有 `sk-...`，也没有完整的 `Authorization` 头。用的是已有的 `redact`。

## 概念笔记

追踪不参加作答。`trace_rag` 读一次已经算完的 `RagAnswer`，`trace_agent` 读一次已经跑完的 `AgentResult`。`RagPipeline.answer` 不改。第 06 课到第 08 课的工具协议、第 11 课的 clarify、第 12 课的流式、第 13 课的「资料是数据」，默认路径都不改。

六个字段各一行：

| 字段 | 检索路径 | 工具路径 |
| --- | --- | --- |
| 问题 | 客人问的那句，先脱敏 | 同一套脱敏 |
| 最高分 | 这次命中里的最高余弦。没有命中就是（无） | （无）。这条路径不检索 |
| 拒绝原因 | 拒绝时写下 `reason`：`low_score`、`missing_citation` 或 `unknown_citation`。没拒绝就空着 | （无）。这条路径不用第 05 课的拒绝 |
| 工具名 | （无） | 实际出现过的工具名。没有就是（无） |
| 步数 | 0。不走第 07 课的循环 | `steps` 的长度 |
| 停止原因 | 第 05 课的 `reason`，包括答成了的 `ok` | `final`、`clarify`、`max_steps`、`repeated_tool`、`unknown_tool` |

给客人的那句拒绝不写进追踪。两种失败的回答一样，抄进拒绝原因就又看不出差别。

第 08 课 `refuse-pool`（游泳池几点开放？）在这份字面向量上最高分是 0，低于 `DEFAULT_MIN_SCORE`（0.2）。模型没有被调用。拒绝原因是 `low_score`。

第 08 课 `cite-checkin`（入住时间是什么时候？）最高分大约 0.396，已经过线。Mock 会引用 `faq-checkin`，原因是 `ok`。把草稿换成「游泳池全天开放。来源：[faq-pool]」，引用检查整段作废，拒绝原因是 `unknown_citation`。草稿里根本没有 `[faq-...]` 时，拒绝原因是 `missing_citation`。这两次给客人的句子仍是那句固定拒绝，最高分仍过线。和 `low_score` 不是同一条记录。

写了「延迟退房 3 小时要多少钱？」时，工具名是 `quote_late_checkout`，步数是 2（调用一次，再 `final`），停止原因是 `final`。没写小时数时工具名是（无），步数是 1，停止原因仍是 `final`。停止原因相同，工具名和步数不同。

`emit_trace` 先 `print`。传入路径时，再把刚打印的那段写入文件。文件写失败，屏幕上已经有记录。问题、工具名，以及打印前的每个字符串字段，都走 `ailab.redact.redact`。`sk-...` 和 `Bearer ...` 会变成 `[redacted]`。`Authorization: Bearer sk-...` 不会整段留下。这和对话接口报错时用的是同一个函数，不另写一套。

允许列表仍然只有 `lookup_policy` 和 `quote_late_checkout`。追踪只记下名字，不执行新的工具。

## 跑起来

```bash
python lessons/14-minimal-trace/exercise.py
```

应当先看到那句固定拒绝，然后三张检索追踪：`low_score` 的最高分是 0.000；`unknown_citation` 和 `missing_citation` 的最高分过了 0.2。拒绝原因三行彼此不同。

接着三张工具追踪：`quote_late_checkout` 步数 2，没写小时数时工具名是（无）、步数 1，取消政策的工具名是 `lookup_policy`。停止原因都是 `final`。

最后一张追踪的问题里夹了密钥。它先打到屏幕上，是 `[redacted]`，没有 `sk-testsecret`，也没有 `Bearer`。然后再打一遍并写入临时文件，两段相同。退出码为 0。没有 API 密钥也能跑。

建议改一处：在 `src/ailab/trace.py` 的 `trace_rag` 里，把 `refusal_reason` 改成始终写 `REFUSAL` 那句给客人的话，再跑练习。三张检索追踪的拒绝原因会变成一样，练习以退出码 1 结束。看完改回去。不要为了变绿去改 `tests/test_trace.py` 里的期望，也不要改 `RagPipeline.answer`。

## 先读哪段代码

- `lessons/14-minimal-trace/exercise.py`
- `src/ailab/trace.py` 里的 `TraceRecord`、`trace_rag`、`trace_agent`、`emit_trace`
- `src/ailab/redact.py` 里的 `redact`
- `src/ailab/rag.py` 里 `low_score`、`missing_citation`、`unknown_citation` 各在什么时候写下。这一课只读，不改

## 常见坑

- 把两种拒绝都记成那句固定拒绝语。客人已经分不清，追踪也跟着分不清。
- 只记一个「拒绝了」。`low_score` 和 `unknown_citation` 就变成同一条。
- 先写文件，屏幕上没有字。不打开文件的人看不到记录。
- 把 `Authorization` 头或 `sk-...` 原样写进问题、工具名。报错文本已经会抹掉这些样式，追踪要走同一个函数。
- 为了记追踪去改 `RagPipeline.answer`，或去改第 06 课到第 13 课的默认路径。
- 把模型回复里的数字当成步数。步数是循环里实际走过的 `steps`，检索路径记 0。

## 口头自查

1. 游泳池和引用了 `[faq-pool]` 的入住问题，客人看到的是同一句。追踪里哪一行说明它们不是同一种失败？最高分为什么一个是 0、一个过了 0.2？
2. 没写小时数和写了 3 小时，停止原因都是 `final`。要看哪两个字段，才知道有没有调用 `quote_late_checkout`？
3. 为什么追踪里不能留下完整的 `Authorization` 头？打印和写文件，哪一个先发生？

口头回答即可。说不顺的那一题，把练习里两张检索追踪的「拒绝原因」和「最高分」并排再讲一遍。

## 下一课

[第 15 课：换成真实接口的对照](../15-real-api-compare/README.md)
