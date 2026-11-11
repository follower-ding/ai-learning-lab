# 后面的课

01 到 14 已经有笔记、练习和离线测试。下面这些还没有 `exercise.py`。不要到仓库里找对应目录。

做这些课之前，先让 `python -m pytest` 保持全绿。每一课只加一种机制，并给它至少一条能失败的用例。

## 09 混合检索

已有练习，见 [lessons/09-hybrid-retrieval](lessons/09-hybrid-retrieval/README.md)。

同一套字符二元组上同时保留 BM25 关键词分和 TF-IDF 余弦。两路先压到同一把尺子上，再按 0.5/0.5 加权。第 03 课那三句问法收在 `src/ailab/data/hybrid_queries.json`。无关句两路都是 0，融合分也是 0。只在向量路上擦过 0.2 的泛词和错文档，加权后仍拒绝。报告里同时留下两路分数，失败时能看出是哪一路的问题。

## 10 重排序

已有练习，见 [lessons/10-rerank](lessons/10-rerank/README.md)。

第 04 课的库一次取前几条。资料变多以后，先宽一点取 20 条，再用第二次打分留下 4 条交给第 05 课。

第二次打分必须和第一次用同一批候选，不能偷偷改去读全文库。评测要能看出：重排改变了顺序，但没有制造出检索阶段不存在的 id。占比只看这批正文；`Hit.score` 仍是余弦，第 05 课的 0.2 不和占比比较。第 05 课、第 08 课的默认检索路径不改。

## 11 缺槽时先问

已有练习，见 [lessons/11-clarify-missing-slot](lessons/11-clarify-missing-slot/README.md)。

第 06 课没写小时数时，Mock 用 `final` 反问，不猜 1 小时。这一课把「反问」做成单独的一步，例如 `{"type":"clarify","question":"要延迟几小时？"}`。`parse_action` 读出 `question`，`run_agent` 见到 `clarify` 就停，停止原因也是 `clarify`，不再把这句话当成已经答完。

`ask_first=True` 时，系统提示里多一行【缺槽先问】。启发式 Mock 见到这个标记，缺小时数就输出 clarify。不传这个参数时，第 06 课到第 08 课仍是 `final`。允许列表仍然只有 `lookup_policy` 和 `quote_late_checkout`。

评测要检查：缺小时数时没有调用 `quote_late_checkout`，也没有出现 30、60、90，也没有把小时数猜成 1。写明了小时数时，仍调用计价工具。

## 12 流式输出

已有练习，见 [lessons/12-streaming-output](lessons/12-streaming-output/README.md)。

真实接口可以逐段返回。第 05 课的引用检查要看到完整的 `[faq-...]` 才能决定留还是扔。这一课不改 `RagPipeline.answer`，也不改第 06 课到第 08 课的工具协议。离线只把 `complete` 的整段按句号和方括号切开，当作逐段到达，不联网。

两种办法并排。最终回答都等于 `apply_citation_policy` 的结果，被作废的半句不留下：

- 先收齐再检查：检查完成前屏幕上没有字。作废的半句不会出现。检查通过才整段给出。
- 边打印边检查：每一段先推到屏幕上，检查同时看着已经到达的前缀。半句比完整引用先到，所以中途看得到。未知引用一旦闭合，后面的段不再推。最终回答仍整段换成拒绝语。没有完整引用时要等到流结束才知道，那些段会先出现，然后最终回答清掉。

分数不够时两种办法都不调用模型，中途没有草稿。这和第 05 课的 `low_score` 一样。

## 13 资料是数据，工具没有命令行

已有练习，见 [lessons/13-material-is-data](lessons/13-material-is-data/README.md)。

检索到的段落、工具返回的字符串，都可能含有「忽略之前的规则」这类句子。它们是数据，不是新的系统指令。这句话只写在本课的 `fixture.txt` 里，不写进 `hotel_policy.md` 和 `hotel_faq.json`。

工具名不从这段文字里读。允许列表仍然只有 `lookup_policy` 和 `quote_late_checkout`。没有通道去执行 shell、拼 SQL 或读任意路径。

Mock 靠提示里的「【工具协议】」这类标记选择脚本。那只是为了让离线输出稳定，不能当作安全边界。

## 14 最小追踪

已有练习，见 [lessons/14-minimal-trace](lessons/14-minimal-trace/README.md)。

每次作答记下六个字段：问题、最高分、拒绝原因、工具名、步数、停止原因。先打印到标准输出。文件只是同一段已经脱敏的文字，不打开也能看见记录。

追踪用已有的 `redact`，不记录 `sk-...`，也不记录完整的 `Authorization` 头。第 08 课 `refuse-pool` 的拒绝原因是 `low_score`，最高分是 0，没过 0.2。同一句入住问题如果草稿引用了没检索到的文档，拒绝原因是 `unknown_citation`；草稿没有引用时是 `missing_citation`。给客人的句子仍是那句固定拒绝，最高分却已经过线。这三条拒绝原因在追踪里不能写成一样。

不改 `RagPipeline.answer`，也不改第 06 课到第 13 课的默认路径。允许列表仍然只有 `lookup_policy` 和 `quote_late_checkout`。

## 15 换成真实接口的对照

`OpenAICompatibleClient` 和 `OpenAICompatibleEmbedder` 已经留好，默认测试用假传输，不联网。

这一课自己选一课（建议第 05 课或第 08 课），在本地 `.env` 配好密钥后跑一次真实接口，把引用、拒绝和工具是否还成立记下来。真实调用有超时和 `max_tokens`。不要把这次结果写进默认 `pytest`，没有密钥的同学仍要能做完 01 到 08。

换嵌入模型后维度会变，第 04 课的库要整库重算，第 05 课的 `DEFAULT_MIN_SCORE` 也要重新看分数再定。
