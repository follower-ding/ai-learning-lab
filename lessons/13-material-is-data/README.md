# 第 13 课：资料是数据

检索到的段落、工具返回的字符串，都可能含有「忽略之前的规则」这类句子。它们是数据，不是新的系统指令。工具名不从这段文字里读。两只工具都是纯函数，没有命令行：不执行 shell，不拼 SQL，也不按资料去打开路径。

这句话只写在本课的 `fixture.txt` 里。`hotel_policy.md` 和 `hotel_faq.json` 不改，第 05 课、第 08 课、第 09 课的分数和问法才不会被带偏。

## 学习目标

- 看见本课夹具接在 `lookup_policy` 的返回值后面，原文仍在，这句话也在。
- 确认工具名仍是 `lookup_policy`，不是这句资料，也不是从这句资料里切出来的片段。
- 确认允许列表仍然只有两只工具。资料里的字不会打开 shell、SQL 或任意路径。

## 概念笔记

`lookup_policy` 按主题读规章，返回一段字符串。本课在这段字符串末尾接上夹具，再跑第 07 课已有的循环。默认的 `hotel_tools()` 不改。下一次不带夹具的调用，返回值里没有这句话。

循环里工具名只有一个来源：模型输出的 JSON，字段是 `name`。`run_agent` 拿这个名字去允许列表里找。找到了才执行。允许列表仍然只有：

- `lookup_policy`
- `quote_late_checkout`

`tool_name_from_material` 收到整段资料，返回值是 None。它不读正文，也就不会把「忽略之前的规则」或其中某一小段当成工具名。`settle_tool_result` 把返回的原文放进 `text`，把允许列表里已经执行的名字放进 `tool_name`。`name_from_text` 保持 None。原文不删、不改，入住时间的 `14:00` 还在。

最终回答里可以看见这句话。Mock 把工具返回的整段原样带进 `final`。它是被引用的资料，不是下一只工具的名字。

就算把这句话写进 JSON 的 `name`，它也不在允许列表里。循环停在 `unknown_tool`，两只工具都不执行。这一步记录的是模型写出的那个名字被拒绝了，不是从资料里读出了一个名字。

Mock 见到系统提示里的「【工具协议】」才走工具脚本。这个标记只是为了让离线输出每次一样，练习才有稳定的期望。它不是安全边界。资料正文里即使也出现这四个字，`tool_name_from_material` 仍然返回 None，允许列表不增加。没有这个标记，也不会因此多出 shell、SQL 或读任意路径的通道。通道根本不在这两只工具里：它们只返回字符串。

第 05 课的 `RagPipeline.answer`、第 06 课到第 08 课的工具协议、第 11 课的 clarify、第 12 课的流式默认路径，这一课都不改。

## 跑起来

```bash
python lessons/13-material-is-data/exercise.py
```

应当先看到夹具原文，里面有「忽略之前的规则」。然后是入住问题：第 1 步调用 `lookup_policy`，资料里同时有 `14:00` 和这句夹具，从资料读出的工具名是「（无）」。停止原因是 `final`。

接着，资料里再写上【工具协议】，读出的工具名仍是「（无）」。把这句资料写进 JSON 的 `name` 时，停止原因是 `unknown_tool`，工具没有执行。

最后一行说明【工具协议】只选择离线脚本。允许列表是 `lookup_policy`、`quote_late_checkout`。退出码为 0。没有 API 密钥也能跑。

建议改一处：在 `src/ailab/material.py` 里，把 `tool_name_from_material` 的 `return None` 改成 `return material.text`，再跑练习。整段资料会被当成从正文读出的名字，练习以退出码 1 结束。看完改回去。不要为了变绿去改 `tests/test_material.py` 里的期望，也不要改 `hotel_policy.md` 或 `hotel_faq.json`。

## 先读哪段代码

- `lessons/13-material-is-data/exercise.py`
- `lessons/13-material-is-data/fixture.txt`
- `src/ailab/material.py` 里的 `tool_name_from_material`、`settle_tool_result`、`run_with_injected_material`
- `src/ailab/agent.py` 里不允许的工具名如何停
- `src/ailab/mock_front_desk.py` 里「【工具协议】」选择哪一段脚本

## 常见坑

- 从工具返回的字符串里切出一句话，填进下一轮的工具名。正文只留在 `text` 里。
- 为了放进这句话去改 `hotel_policy.md` 或 `hotel_faq.json`。第 05 课、第 08 课、第 09 课读的是那两份默认资料。
- 把【工具协议】当成安全边界。它只让 Mock 选择离线脚本。边界是允许列表，以及工具名不从资料里读。
- 往允许列表里加第三只工具。列表仍然只有 `lookup_policy` 和 `quote_late_checkout`。
- 让工具去执行返回的字符串。这两只工具只算出一段文字，没有命令行。

## 口头自查

1. 工具返回的正文里已经有「忽略之前的规则」。为什么这一步的工具名仍然是 `lookup_policy`？
2. 系统提示里的【工具协议】让 Mock 做了哪一件事？为什么资料里出现同样四个字，也不能把它当成安全边界？
3. 这句话如果写在 JSON 的 `name` 里，停止原因是什么？`lookup_policy` 有没有执行？

口头回答即可。说不顺的那一题，把练习里「资料」和「从资料读出的工具名」两行并排再讲一遍。

## 下一课

第 14 课和第 15 课还没有练习，写在 [路线图](../../ROADMAP.md)。
