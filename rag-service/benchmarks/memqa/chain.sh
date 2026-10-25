cd "$(dirname "$0")/../.."
export LLM_REQUESTS_PER_MINUTE=10 PYTHONUNBUFFERED=1
PY=../venv/Scripts/python.exe
$PY -m benchmarks.memqa.run --dataset longmemeval --systems no_memory,full_history,archive,full --questions 14 --resume benchmarks/results/memqa_longmemeval.json >> benchmarks/results/memqa.log 2>&1
$PY -m benchmarks.memqa.run --dataset locomo --systems no_memory,full_history,archive,full --questions 15 --resume benchmarks/results/memqa_locomo.json >> benchmarks/results/memqa.log 2>&1
