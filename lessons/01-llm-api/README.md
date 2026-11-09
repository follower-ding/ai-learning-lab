# 第 01 课：一次调用

岸港酒店是虚构前台。这一课只做一件事：把一组消息交给客户端，拿回一段文本。

## 学习目标

- 分清 `system`、`user`、`assistant`、`tool` 各放什么。
- 调用方只依赖 `complete(messages)`，不依赖某一个厂商的 SDK。
- 知道练习里的 Mock 是脚本，密钥不进仓库。

## 概念笔记

一次调用的输入是消息列表，不是一整段分不清角色的字符串。`system` 放这次对话都要遵守的规则，`user` 放客人这一问，`assistant` 是模型已经说过的话。`tool` 留到第 06 课，表示代码执行工具之后写回来的结果。

`MockChatClient` 和 `OpenAICompatibleClient` 都有 `complete`。练习构造的是 Mock。它按关键词从 `hotel_faq.json` 抄一句规章，抄不到就说自己只准备了那几类问题。读 `src/ailab/mock_front_desk.py` 就能看见全部规则。把它当成模型，后面的课会无从排错。

打印出来的 token 数是估算：中文大约一字一个，其余字符大约四个算一个。供应商账单以接口返回的 `usage` 为准。真实客户端还会带上超时和 `max_tokens`，避免一次调用一直挂着或写出很长的正文。

密钥放在环境变量里。仓库只有 `.env.example`。练习不读它。异常信息里如果出现 `sk-...` 或 `Bearer ...` 的样式，会被换成 `[redacted]`。

## 跑起来

```bash
python lessons/01-llm-api/exercise.py
```

你应当看到模型名 `mock-hotel-front-desk`，以及下午 14:00 入住这句话。

建议改一行：把问题换成「游泳池几点开放？」。Mock 应当回到那句「只准备了……」，而不是编一个开放时间。

## 先读哪段代码

- `lessons/01-llm-api/exercise.py`
- `src/ailab/messages.py`
- `src/ailab/client.py` 里的 `MockChatClient`
- `src/ailab/mock_front_desk.py` 里的 `plain_reply`

`OpenAICompatibleClient` 这一课可以只扫一眼签名。它请求 `/chat/completions`，并把 `tool` 角色改写成一条 user 消息，因为有的兼容接口不接受 `tool` 这个 role。

## 常见坑

- 把 API 密钥写进源码或提交 `.env`。
- 用估算 token 去对账。
- 角色揉成一条字符串，后面无法把工具结果和客人的话分开。
- 认为 Mock 答得出游泳池、接送、价格。它没有这些句子，就不该现编。

## 口头自查

1. `system` 和 `user` 各约束什么？如果把整份规章放进 `user`、把客人的问题放进 `system`，后面的调用方会难在哪里？
2. 本课打印的 token 数可以拿去和供应商账单对比吗？真实计数从响应的哪里读？
3. 换成 `OpenAICompatibleClient` 时，练习里哪些行要改，`complete` 之后的打印要不要跟着改？

口头回答即可。说不顺的那一题，回到概念笔记里对应的几句，对照练习输出再讲一遍。

## 下一课

[第 02 课：把回答收成 JSON](../02-prompt-json/README.md)
