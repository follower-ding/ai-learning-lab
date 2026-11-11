# 第 11 课：缺槽时先问

第 06 课没写小时数时，Mock 输出 `final` 反问，不猜 1 小时。那句话的停止原因也是 `final`，和第 07 课「已经答完」用的是同一个字。这一课把反问拆成单独的一步。

## 学习目标

- 让模型在缺小时数时输出 `clarify`，并带上 `question`。
- 这一步不调用 `quote_late_checkout`，输出里不出现 30、60、90，也不把小时数猜成 1。
- 小时数已经写明时，仍走原来的计价。允许列表仍然只有两只工具。

## 概念笔记

三种动作都是一个 JSON 对象：

```json
{"type": "tool_call", "name": "quote_late_checkout", "arguments": {"hours": 2}}
{"type": "final", "answer": "给客人的话"}
{"type": "clarify", "question": "要延迟几小时？"}
```

`final` 表示这轮已经可以交给客人，停止原因是 `final`。`clarify` 表示还缺一个必填槽，停止原因是 `clarify`。给客人看的那句就是 `question`。它不是工具结果，也不是价格。

`parse_action` 认这三种 `type`。`question` 必须是非空字符串。空字符串、缺字段、或者写成数字，都只报解析错误，不会改去调用工具。

`run_agent` 见到 `clarify` 就停下。不查允许列表，不执行函数，也不再要下一轮。步数记录里这一步的 `kind` 是 `clarify`。

离线 Mock 靠系统提示里的标记选脚本。第 06 课到第 08 课的提示里只有【工具协议】，缺小时数时仍输出 `final`，那三课的用例不用改。这一课调用 `run_agent(..., ask_first=True)`。系统提示多一行【缺槽先问】，Mock 才改输出 `{"type":"clarify","question":"要延迟几小时？"}`。

单价仍是每小时 30 元，只存在于 `quote_late_checkout`。没写小时数就调用它，只能先猜一个整数。猜 1 小时会得到 30 元，猜 2 小时是 60 元，猜 3 小时是 90 元。这三句都不是客人要的价。反问里也不写这些数字，免得看起来像已经算过。

写了「2 小时」时，即使用了【缺槽先问】，仍然先调用 `quote_late_checkout`，再 `final`。先问只补缺的槽，不拦住已经给齐的参数。

允许列表没有增加。反问不是第三只工具，没有 `name`，也没有 `arguments`。

## 跑起来

```bash
python lessons/11-clarify-missing-slot/exercise.py
```

应当先看到没写小时数的那次：停止原因 `clarify`，反问是「要延迟几小时？」，没有工具名。然后是写了 2 小时的那次：停止原因 `final`，回答里有「合计 60 元」。最后一行是第 06 课的同一句问法，停止原因仍是 `final`。退出码为 0。没有 API 密钥也能跑。

建议改一处：把练习里两处 `ask_first=True` 删掉，再跑。缺小时数会回到 `final`，练习以退出码 1 结束。看完改回去。

再改一处也可以：在 `src/ailab/mock_front_desk.py` 里，【缺槽先问】的分支改成输出 `quote_late_checkout` 且 `hours` 为 1。反问会消失，并出现 30。看完改回去。不要为了变绿去改 `tests/test_clarify.py` 里的期望，也不要改第 08 课的 9 条用例。

## 先读哪段代码

- `lessons/11-clarify-missing-slot/exercise.py`
- `src/ailab/agent.py` 里的 `parse_action`、`run_agent`
- `src/ailab/mock_front_desk.py` 里缺小时数的那一支

## 常见坑

- 用 `final` 把反问送出去。字面上看都是在问客人，停止原因却变成已经答完。这一课要的是 `clarify`。
- 缺小时数时调用 `quote_late_checkout`，把 `hours` 设成 1。函数会认真算出 30 元。
- 在 `question` 里写下 30、60 或 90。评测把这三个数当成已经报过价。
- 为了新协议去改第 06 课的 Mock，让旧练习也输出 `clarify`。旧练习把非工具调用当成 `final`，去读 `answer`。两条协议分开。
- 往允许列表里加一只「提问」工具。反问不执行函数。列表仍然只有 `lookup_policy` 和 `quote_late_checkout`。

## 口头自查

1. 缺小时数时，为什么停止原因不能仍是 `final`？客人看到的那句问话，和第 07 课答完时的 `answer` 差在哪？
2. 猜 1 小时再调用计价，输出里为什么会出现 30？不猜的时候，为什么 60 和 90 也不该出现？
3. 同一句「延迟退房怎么收费？」，不传 `ask_first` 时为什么仍是 `final`？第 06 课要守住的是哪一种输出？

口头回答即可。说不顺的那一题，把练习印出的两段 JSON 并排再讲一遍。

## 下一课

[第 12 课：流式输出](../12-streaming-output/README.md)

第 13 课到第 15 课还没有练习，写在 [路线图](../../ROADMAP.md)。
