# AI 应用开发学习实验

这是一套按编号往下做的实验课。练的是应用这一侧：怎么调用、怎么把输出收成 JSON、怎么切块和做向量、怎么在检索之后引用或拒绝、怎么让模型提出工具调用并由你的代码执行、怎么给循环加上限，以及怎么用很小的一组用例把这些行为盯住。

这里不训练模型，也不交付一个酒店产品。贯穿全部练习的「岸港酒店」是虚构前台，规章写在 `src/ailab/data/`，用来固定问题和期望行为。

## 准备

需要 Python 3.11 或更新版本。默认路径不需要 API 密钥。

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest
python lessons/01-llm-api/exercise.py
```

虚拟环境激活之后，用环境里的 `python`。系统上如果只有 `python3`，上面第一行已经用它建好环境。

`make test` 跑测试，`make lessons` 按顺序跑完十五课练习。这两条默认调用 `python3`；要用虚拟环境里的解释器时写成 `make test PY=python`。

## 课程

按编号做。第 05 课和第 06 课都接在向量库之后；第 07 课接工具调用；第 08 课把检索和工具放进同一套用例。第 09 课回到检索：关键词和向量各留一个分，加权之后再决定过不过线。第 10 课仍用第 04 课的库，先宽取 20 条，只对这一批重打分，留下 4 条交给第 05 课。第 11 课仍用第 07 课的循环：延迟退房没写小时数时输出 clarify 先问，不调用计价，也不猜 1 小时。第 12 课把同一段草稿用两种办法送到屏幕上。先收齐再检查时，被第 05 课作废的半句不会出现。边打印边检查时，这半句会先出现在中途，最终回答里仍然不留下。第 13 课把一句「忽略之前的规则」放进本课夹具，接在工具返回的字符串后面。这句话仍是数据，不会变成工具名。允许列表仍然只有 `lookup_policy` 和 `quote_late_checkout`。第 14 课在作答之后记下问题、最高分、拒绝原因、工具名、步数和停止原因，先打到标准输出。第 08 课里没过阈值，和引用检查扔掉草稿，在追踪里不是同一个拒绝原因。第 15 课用第 08 课那 9 条做一次对照：离线 Mock 先跑，本地有密钥时再把 `complete` 换成真实接口，看引用、拒绝和工具是否还成立。没有密钥就跳过，不把这次调用写进默认测试。

| 课 | 目录 | 做完时你手里有什么 |
| --- | --- | --- |
| 01 | [一次调用](lessons/01-llm-api/README.md) | 一组消息进，一段文本出。Mock 和真实客户端同一套方法 |
| 02 | [提示与 JSON](lessons/02-prompt-json/README.md) | 意图落在允许列表里；解析失败只再要一次 |
| 03 | [切块与向量](lessons/03-embedding-chunking/README.md) | 长规章被切开；字面向量能比较，但不会理解换一种说法 |
| 04 | [向量库](lessons/04-vector-store/README.md) | 稳定 id、原文、向量放在一起，能取出 top-k |
| 05 | [引用或拒绝](lessons/05-rag-cite-or-refuse/README.md) | 分数不够不调用模型；引用了没检索到的文档就整段作废 |
| 06 | [工具调用](lessons/06-tool-calling/README.md) | 模型提出调用，允许列表里的函数才执行 |
| 07 | [有上限的循环](lessons/07-agent-loop/README.md) | 步数用尽或重复调用时停下来，不编一个结果圆场 |
| 08 | [小评测](lessons/08-tiny-eval/README.md) | 9 条用例分别检查引用、拒绝和工具 |
| 09 | [混合检索](lessons/09-hybrid-retrieval/README.md) | 关键词分和向量分按固定权重合成；无关问法仍然拒绝 |
| 10 | [重排序](lessons/10-rerank/README.md) | 先宽取 20 条，只对这一批重打分，留下 4 条；不发明新的 id |
| 11 | [缺槽时先问](lessons/11-clarify-missing-slot/README.md) | 缺小时数时单独走 clarify；不调用计价，也不出现 30、60、90 |
| 12 | [流式输出](lessons/12-streaming-output/README.md) | 先收齐时看不到作废的半句；边打印时中途看得到，最终回答里不留 |
| 13 | [资料是数据](lessons/13-material-is-data/README.md) | 夹具里的句子留在资料里；工具名仍来自允许列表，不从正文里读 |
| 14 | [最小追踪](lessons/14-minimal-trace/README.md) | 六个字段先出现在标准输出；没过阈值和引用检查作废草稿不是同一条拒绝原因 |
| 15 | [真实接口对照](lessons/15-real-api-compare/README.md) | 同一套第 08 课用例先看离线 Mock；有密钥再跑一次真实接口，看引用、拒绝和工具是否还成立 |

目前排到第 15 课。若以后再加，写在 [ROADMAP.md](ROADMAP.md)。

## 仓库怎么分

- `lessons/`：每课的笔记和 `exercise.py`。练习尽量短，先跑再读。
- `src/ailab/`：练习调用的实现。
- `src/ailab/data/`：酒店短问答、长规章、评测用例。测试和练习读同一份。
- `tests/`：不访问网络。有一条标记为 `integration` 的测试，没设置 `OPENAI_API_KEY` 时会跳过。第 15 课的真实对照不放进默认测试。

## 密钥

仓库里只有 [`.env.example`](.env.example)。把它复制成 `.env` 之后再填写，`.env` 已经被忽略。第 01 课到第 14 课的练习不读取这些变量。第 15 课只在你要做那一次真实对照时读取；没有密钥就跳过，退出码为 0。

真实调用放在 `OpenAICompatibleClient` 和 `OpenAICompatibleEmbedder`。它们和 Mock、TF-IDF 共用 `complete` / `embed`。练习里构造客户端的那几行可以换，后面的解析不用换。异常文本会抹掉 `sk-...` 和 `Bearer ...` 的样式。第 14 课的追踪用的是同一个 `redact`。

## 再加一课

1. 新建 `lessons/15-你的题目/README.md` 和 `exercise.py`。笔记里写学习目标、概念、怎么跑、常见坑、三道口头自查。
2. 练习打印的第一行含有 `第 15 课`。
3. 把路径加进 `tests/test_exercises.py` 的 `LESSONS`。
4. 需要新行为时，先写一条会失败的测试，再改 `src/ailab/`。不要把密钥写进练习或测试。
5. 更新上面的表，并在路线图里把这一课从「未写」改成「已有练习」。

## 演示与评测

岸港酒店的演示和评测在 [`apps/hotel-rag/`](apps/hotel-rag/README.md)。这不是新的一课。`make hotel-rag-demo` 离线跑四条固定案例，`make hotel-rag-eval` 仍评第 08 课那 9 条。这两条没有接进 `make lessons`。
