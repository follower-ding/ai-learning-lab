# 第 02 课：把回答收成 JSON

自由文本不好分支。这一课让模型只交一个 JSON 对象，并且意图必须落在允许列表里。

## 学习目标

- 用括号匹配取出第一个 JSON 对象。外面可以有说明文字，也可以包一层 json 代码围栏。
- 字段不合法就视为解析失败，而不是带着残缺数据往下走。
- 失败时把上一轮输出发回去，只再要一次。

## 概念笔记

系统提示要求这种形状：

```json
{"intent": "check_in", "question": "客人的原话"}
```

`intent` 只能是 `check_in`、`check_out`、`breakfast`、`parking`、`pets`、`wifi`、`cancel`、`unknown`。列表外的值，例如 `swim`，和「这根本不是 JSON」一样处理。

提取时先去掉围栏，再从第一个 `{` 开始配对，字符串里的括号不算。然后 `json.loads`。不要 `eval`：模型输出是不受信任的文本，`eval` 会把文本当成代码执行。

`classify_intent` 最多调用两次。第一次失败，就把模型刚才的输出放进 `assistant`，再追加一句「只输出一个 JSON 对象」。第二次仍失败就抛出 `JsonExtractError`。无限重试会在坏输出上打转。

关键词顺序写在 Mock 里：`取消` 排在 `入住` 前面，所以「入住日前可以取消吗」归到 `cancel`。这是脚本的规则，用来让练习稳定，不是自然语言理解。

## 跑起来

```bash
python lessons/02-prompt-json/exercise.py
```

三句问法应当分别得到 `check_in`、`pets`、`unknown`，而且都是第 1 次解析成功。最后一段故意先给一句人话，再给 JSON，次数是 2。

建议改一行：把脚本的第一句换成合法 JSON，再跑。次数应当变成 1。

## 先读哪段代码

- `lessons/02-prompt-json/exercise.py`
- `src/ailab/json_prompt.py`

## 常见坑

- 用 `eval` 解析模型输出。
- 解析失败就猜一个意图继续走。猜错的意图会在后面的课里变成错误的工具或错误的检索。
- 修复轮次不把失败的原文发回去，模型不知道你拒绝的是哪一段。
- 允许列表写在提示里，却不在代码里再查一次。提示会被模型忽略。

## 口头自查

1. 为什么这里用 `json.loads`，而不是 `eval`？
2. 模型先写了解释再给 JSON 时，修复轮次把什么发回去？最多再调用几次？
3. `intent` 写成 `swim` 时，程序应当悄悄收成 `unknown`，还是当成解析失败再要一次？

## 下一课

[第 03 课：切块与向量](../03-embedding-chunking/README.md)
