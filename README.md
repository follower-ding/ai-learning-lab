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

`make test` 跑测试，`make lessons` 按顺序跑完八课练习。这两条默认调用 `python3`；要用虚拟环境里的解释器时写成 `make test PY=python`。

## 课程

按编号做。第 05 课和第 06 课都接在向量库之后；第 07 课接工具调用；第 08 课把检索和工具放进同一套用例。

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

第 09 课往后还没有练习，写在 [ROADMAP.md](ROADMAP.md)。

## 仓库怎么分

- `lessons/`：每课的笔记和 `exercise.py`。练习尽量短，先跑再读。
- `src/ailab/`：练习调用的实现。
- `src/ailab/data/`：酒店短问答、长规章、评测用例。测试和练习读同一份。
- `tests/`：不访问网络。有一条标记为 `integration` 的测试，没设置 `OPENAI_API_KEY` 时会跳过。

## 密钥

仓库里只有 [`.env.example`](.env.example)。把它复制成 `.env` 之后再填写，`.env` 已经被忽略。八课练习不读取这些变量。

真实调用放在 `OpenAICompatibleClient` 和 `OpenAICompatibleEmbedder`。它们和 Mock、TF-IDF 共用 `complete` / `embed`。练习里构造客户端的那几行可以换，后面的解析不用换。异常文本会抹掉 `sk-...` 和 `Bearer ...` 的样式。

## 再加一课

1. 新建 `lessons/09-你的题目/README.md` 和 `exercise.py`。笔记里写学习目标、概念、怎么跑、常见坑、三道口头自查。
2. 练习打印的第一行含有 `第 09 课`。
3. 把路径加进 `tests/test_exercises.py` 的 `LESSONS`。
4. 需要新行为时，先写一条会失败的测试，再改 `src/ailab/`。不要把密钥写进练习或测试。
5. 更新上面的表，并在路线图里把这一课从「未写」改成「已有练习」。
