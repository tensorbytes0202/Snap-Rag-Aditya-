cd "$(dirname "$0")/../.."
LLM_REQUESTS_PER_MINUTE=10 PYTHONUNBUFFERED=1 ../venv/Scripts/python.exe -m benchmarks.memqa.run --dataset locomo --systems no_memory,full_history,archive,full --questions 15 --resume benchmarks/results/memqa_locomo.json >> benchmarks/results/memqa.log 2>&1
