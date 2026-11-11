PY ?= python3

.PHONY: test lessons hotel-rag-demo hotel-rag-eval hotel-rag-app

test:
	$(PY) -m pytest

lessons:
	$(PY) lessons/01-llm-api/exercise.py
	$(PY) lessons/02-prompt-json/exercise.py
	$(PY) lessons/03-embedding-chunking/exercise.py
	$(PY) lessons/04-vector-store/exercise.py
	$(PY) lessons/05-rag-cite-or-refuse/exercise.py
	$(PY) lessons/06-tool-calling/exercise.py
	$(PY) lessons/07-agent-loop/exercise.py
	$(PY) lessons/08-tiny-eval/exercise.py
	$(PY) lessons/09-hybrid-retrieval/exercise.py
	$(PY) lessons/10-rerank/exercise.py
	$(PY) lessons/11-clarify-missing-slot/exercise.py
	$(PY) lessons/12-streaming-output/exercise.py
	$(PY) lessons/13-material-is-data/exercise.py
	$(PY) lessons/14-minimal-trace/exercise.py
	$(PY) lessons/15-real-api-compare/exercise.py

hotel-rag-demo:
	$(PY) apps/hotel-rag/demo.py

hotel-rag-eval:
	$(PY) apps/hotel-rag/eval.py

hotel-rag-app:
	$(PY) apps/hotel-rag/app_eval.py
