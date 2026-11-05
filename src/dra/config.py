"""
Project paths and default hyperparameters.
项目路径与默认超参数。

Every script takes its command-line defaults from here: when moving to another machine or data location,
only this file needs to change (or override the value on the command line).
This file only defines where things are and which settings to use; it does not open or read any file.
所有脚本的命令行参数默认值都从这里读取：换机器或换数据位置时，只需要改这一个文件（或在命令行里用参数覆盖）。
本文件只规定"东西放在哪、用什么设置"，不会打开或读取任何文件。
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths / 路径
# In pathlib, "/" joins path parts (folder / subfolder / file); it is not division.
# pathlib 里的 "/" 是拼接路径（文件夹 / 子文件夹 / 文件），不是除法。
# ---------------------------------------------------------------------------

# Repository root: this file is src/dra/config.py, so go up 3 levels (dra -> src -> repo root).
# 本仓库根目录：本文件在 src/dra/config.py，往上数 3 层（dra -> src -> 仓库根目录）。
REPO_ROOT = Path(__file__).resolve().parents[2]

# Official repository https://github.com/mahta-r/total-recall-qa, as a git submodule under external/
# (dataset, index-building script and official metrics all come from here).
# 官方仓库，以 git submodule 形式放在 external/ 下（数据集、建索引脚本、官方评估指标都从这里来）。
OFFICIAL_REPO = REPO_ROOT / "external" / "total-recall-qa"
INDEX_BUILDER_SCRIPT = OFFICIAL_REPO / "c2_corpus_creation" / "index_builder.py"
OFFICIAL_GEN_METRICS = OFFICIAL_REPO / "c5_task_evaluation" / "metrics" / "generation_eval_metrics.py"

# wiki2 test set: queries (questions + gold answers) and qrels (relevant passages per question).
# wiki2 测试集：queries（问题 + 标准答案）与 qrels（每题的相关段落标注）。
QUERIES_PATH = OFFICIAL_REPO / "dataset" / "TRQA" / "wiki2" / "test" / "queries_wiki2_test.jsonl"
QRELS_PATH = OFFICIAL_REPO / "dataset" / "TRQA" / "wiki2" / "test" / "trec_qrels_wiki2_test.txt"

# Data directory for large files (corpus and index, tens of GB), not tracked by git.
# Default: data/ inside the repo. If the data is elsewhere, set the environment variable TRQA_DATA_DIR
# to point there instead of copying it.
# 数据目录：语料、索引等大文件（几十 GB），不进 git。
# 默认是仓库内的 data/；如果数据已在别处，设置环境变量 TRQA_DATA_DIR 指过去即可，不用拷贝。
DATA_DIR = Path(os.environ.get("TRQA_DATA_DIR", REPO_ROOT / "data"))

# Partial Wikipedia corpus (~5.91M passages), one {"id", "contents": "title\ncontents"} per line.
# 部分维基语料（约 591 万段落），每行一个 {"id", "contents": "标题\n正文"}。
CORPUS_PATH = DATA_DIR / "wiki_partial_corpus.jsonl"

# Contriever + FAISS Flat index (built by step 2).
# Contriever + FAISS Flat 索引（由步骤 2 生成）。
INDEX_DIR = DATA_DIR / "my_indices" / "contriever_test"
INDEX_PATH = INDEX_DIR / "contriever_Flat.index"
# Cache of the index position -> passage id table: built from the corpus the first time (~15 s), read directly after.
# 索引位置 -> 段落 id 映射表的缓存：第一次从语料生成（约 15 秒），之后直接读取。
PASSAGE_IDS_CACHE = INDEX_DIR / "passage_ids.txt"

# Experiment outputs of this repository (retrieval results, evaluation, generated answers).
# 本仓库的实验输出（检索结果、评估结果、生成答案）。
OUTPUT_DIR = REPO_ROOT / "results"
# Retrieval results: written by step 3, read by steps 4 and 5.
# 检索结果：步骤 3 写入，步骤 4、5 读取。
DEFAULT_RUN_FILE = OUTPUT_DIR / "runs" / "contriever_wiki2_test.trec"
# Run logs: every script run writes one timestamped log file here.
# 运行日志：每次运行脚本在这里生成一个带时间戳的日志文件。
LOG_DIR = OUTPUT_DIR / "logs"

# ---------------------------------------------------------------------------
# Models and hyperparameters / 模型与超参数
# ---------------------------------------------------------------------------

# Retriever: must be the model the index was built with (the official index_builder maps "contriever" to it).
# 检索器：必须与建索引时用的模型一致（官方 index_builder 中 contriever 对应此模型）。
RETRIEVER_MODEL = "facebook/contriever-msmarco"
MAX_LENGTH = 512  # max tokens per text, same as the official index_builder / 每段文字最多取多少 token，与官方一致

# Retrieval depth: the largest k needed for R@k; the cut-offs match Table 4.
# 检索深度：评估 R@k 需要的最大 k；各个 k 与 Table 4 一致。
RETRIEVAL_K = 1000
RECALL_KS = (3, 10, 100, 1000)

# Reader (the LLM that writes the answers): Qwen2.5-7B-Instruct in Table 8 and in the official run_evalution.py.
# For local debugging with too little GPU memory, use --model Qwen/Qwen2.5-1.5B-Instruct.
# 阅读器（生成答案的 LLM）：论文 Table 8 与官方 run_evalution.py 默认都是 Qwen2.5-7B-Instruct。
# 本地显存不够时可临时用 --model Qwen/Qwen2.5-1.5B-Instruct 调试。
READER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# Passages put into the prompt (Table 8: K=50; the official code's default is 3).
# 放进 prompt 的段落数（论文 Table 8：K=50；官方代码默认值是 3）。
READER_TOPK = 50
# Max tokens the model may write: same as the official llm_generator.py line 72 (max_new_tokens=1024),
# leaving room for the <think> reasoning so the answer is not cut off.
# 模型最多写多少 token：与官方 llm_generator.py 第 72 行一致，给 <think> 推理留足长度，避免答案还没写出就被截断。
MAX_NEW_TOKENS = 1024
# Sampling temperature, same as generation_temp=0.7 in the official SingleRetrieval.inference.
# 采样温度，与官方 SingleRetrieval.inference 的 generation_temp=0.7 一致。
TEMPERATURE = 0.7

# Random seed: with temperature > 0 generation is random, so the seed must be fixed to reproduce results
# (same default as the official run_evalution.py).
# 随机种子：temperature > 0 时生成带随机性，固定种子才能复现（与官方 run_evalution.py 默认值一致）。
SEED = 42
