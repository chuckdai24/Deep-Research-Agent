"""
项目路径与默认超参数。

所有脚本的命令行参数默认值都从这里读取：换机器或换数据位置时，只需要改这一个文件
（或者在命令行里用参数覆盖）。
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

# 本仓库根目录
REPO_ROOT = Path(__file__).resolve().parents[2]

# 官方仓库 https://github.com/mahta-r/total-recall-qa，以 git submodule 形式放在 external/ 下
# （建索引脚本、数据集、官方评估指标都从这里调用）
OFFICIAL_REPO = REPO_ROOT / "external" / "total-recall-qa"

# 数据目录：语料、索引等大文件（几十 GB）放这里，不进 git。
# 默认是仓库内的 data/；如果数据已在别处，设置环境变量 TRQA_DATA_DIR 指过去即可，不用拷贝。
DATA_DIR = Path(os.environ.get("TRQA_DATA_DIR", REPO_ROOT / "data"))
INDEX_BUILDER_SCRIPT = OFFICIAL_REPO / "c2_corpus_creation" / "index_builder.py"
OFFICIAL_GEN_METRICS = OFFICIAL_REPO / "c5_task_evaluation" / "metrics" / "generation_eval_metrics.py"

# wiki2 测试集：queries（问题 + 标准答案）与 qrels（相关段落标注）
QUERIES_PATH = OFFICIAL_REPO / "dataset" / "TRQA" / "wiki2" / "test" / "queries_wiki2_test.jsonl"
QRELS_PATH = OFFICIAL_REPO / "dataset" / "TRQA" / "wiki2" / "test" / "trec_qrels_wiki2_test.txt"

# 部分维基语料（约 591 万段落），格式为 {"id", "contents": "title\ncontents"}
CORPUS_PATH = DATA_DIR / "wiki_partial_corpus.jsonl"

# Contriever + FAISS Flat 索引
INDEX_DIR = DATA_DIR / "my_indices" / "contriever_test"
INDEX_PATH = INDEX_DIR / "contriever_Flat.index"
# 索引位置 -> doc_id 映射的缓存文件（第一次从语料生成，之后直接读，省去扫描 3.6 GB 语料）
PASSAGE_IDS_CACHE = INDEX_DIR / "passage_ids.txt"

# 本仓库的实验输出（检索结果、评估结果、生成答案）
OUTPUT_DIR = REPO_ROOT / "results"
DEFAULT_RUN_FILE = OUTPUT_DIR / "runs" / "contriever_wiki2_test.trec"
# 运行日志：每次运行脚本生成一个带时间戳的日志文件
LOG_DIR = OUTPUT_DIR / "logs"

# ---------------------------------------------------------------------------
# 模型与超参数
# ---------------------------------------------------------------------------

# 检索器：必须与建索引时用的模型一致（官方 index_builder 中 contriever 对应此模型）
RETRIEVER_MODEL = "facebook/contriever-msmarco"
MAX_LENGTH = 512

# 检索深度：评估 R@k 需要的最大 k
RETRIEVAL_K = 1000
RECALL_KS = (3, 10, 100, 1000)

# 阅读器（生成答案的 LLM）：论文 Table 8 与官方 run_evalution.py 默认都是 Qwen2.5-7B-Instruct
# 本地显存不够时可临时用 --model Qwen/Qwen2.5-1.5B-Instruct 调试
READER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
READER_TOPK = 50  # 放进 prompt 的段落数（论文 Table 8：K=50；官方代码默认值是 3）
# 模型最多写多少 token：与官方 llm_generator.py 第 72 行一致（max_new_tokens=1024），
# 留足 <think> 推理过程的长度，避免答案还没写出就被截断
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7  # 与官方 SingleRetrieval.inference 的 generation_temp=0.7 一致

# 随机种子：temperature > 0 时生成带随机性，固定种子才能复现（与官方 run_evalution.py 默认值一致）
SEED = 42
