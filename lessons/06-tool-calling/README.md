# 第 06 课：工具调用

有些事实不在规章正文里。延迟退房的实验价是每小时 30 元，只存在于 `quote_late_checkout`。模型可以提出调用，结果必须由这个函数算出来。

## 学习目标

- 让模型只输出一个 JSON：要么 `tool_call`，要么 `final`。
- 工具名必须在允许列表里；参数在函数里校验。
- 没给小时数就反问，不默认成 1 小时。

## 概念笔记

这一课故意不使用厂商响应里的 `tool_calls` 字段。模型把调用写在普通文本里：

```json
{"type": "tool_call", "name": "quote_late_checkout", "arguments": {"hours": 2}}
```

Mock 和真实接口因此共用第 02 课的 JSON 提取。换厂商时，解析层不用改。

允许的工具只有两只，都是纯函数：

- `lookup_policy`：按 `topic` 读取已有规章。`topic` 只能是 checkin、checkout、breakfast、parking、pets、wifi、cancel。
- `quote_late_checkout`：`hours` 必须是 1 到 12 的整数。布尔值在 Python 里也算 `int`，这里单独拒绝。

`hours` 不合法时函数抛 `ToolArgumentError`。第 07 课的循环会把这句话写回消息，而不是让栈回溯中断整个请求。

「延迟退房怎么收费」没有数字。Mock 输出 `final`，请客人说明几小时。规章里没有默认时长，猜 1 小时会算出一个无人要求的价格。

`shell` 这类名字可以出现在模型的 JSON 里，但不会出现在 `hotel_tools()` 的字典里。这一课先解析、再对照允许列表，不执行。

## 跑起来

```bash
python lessons/06-tool-calling/exercise.py
```

第一问应当执行工具并打印「合计 60 元」。第二问应当直接反问小时数，输出里没有 30。最后两行是拒绝 `shell`，以及拒绝 `hours: true`。

建议改一行：把第一问里的「2 小时」删掉。它应当走反问，而不是计价。

## 先读哪段代码

- `lessons/06-tool-calling/exercise.py`
- `src/ailab/tools.py`
- `src/ailab/agent.py` 里的 `parse_action` 和 `build_tool_system`

这一课还没有循环。`run_agent` 留到下一课。

## 常见坑

- 让模型在 JSON 里自己填「合计 60 元」。那不是工具，只是又一次自由生成。
- 把模型给出的字符串交给 shell、拼进 SQL，或当作文件路径。本课的两只工具都没有这些能力。
- 参数不合法时直接崩溃，客人只看到栈回溯。
- 小时数缺省时默认 1。评测会把这种「看起来能跑」判失败。

## 口头自查

1. 合计金额为什么必须由 `quote_late_checkout` 计算，而不是让模型写在 `arguments` 里？
2. 没写小时数时，为什么输出 `final` 反问，而不是调用工具并传入 1？
3. 模型输出的工具名叫 `shell` 时，这一课的练习做了什么？它有没有被执行？

## 下一课

[第 07 课：有上限的循环](../07-agent-loop/README.md)
