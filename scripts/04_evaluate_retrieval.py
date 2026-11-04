"""
Step 4: evaluate retrieval.
步骤 4：检索评估。

Compare the retrieval results of step 3 (TREC run file) with the gold annotations (qrels) and
compute entity-level recall, Entity Recall@k, for every k.
把步骤 3 的检索结果（TREC run 文件）和标准答案（qrels）对比，计算每个 k 下的实体级召回 Entity Recall@k。

Entity Recall@k = (gold entities found in the top k) / (all gold entities of the question)
A passage id looks like "15641-0001" = entity 15641, passage 1. An entity counts as found if
any one of its passages is in the top k: TRQA asks for *all relevant entities*, not all passages.
Entity Recall@k = 前 k 个结果中找到的金标实体数 / 这道题全部金标实体数
段落 id 形如 "15641-0001" = 实体 15641 的第 1 个段落。只要前 k 个里有某实体的任意一个段落，
就算找到了这个实体：TRQA 要求找全的是"相关实体"，不是全部段落。

Outputs / 输出：
    <output_dir>/retrieval_summary.json      mean R@k over all questions, in %   所有题的平均 R@k（百分比）
    <output_dir>/retrieval_per_query.jsonl   R@k of every question, 0-1          每道题各自的 R@k（0-1）
                                             (for error analysis and significance tests / 用于错误分析和显著性检验)

Usage / 用法：
    python scripts/04_evaluate_retrieval.py
    python scripts/04_evaluate_retrieval.py --ks 3 10 100 1000
"""

import argparse
import json
from pathlib import Path

from dra import config
from dra.data import load_qrels, read_trec_run
from dra.evaluation import evaluate_retrieval
from dra.logging_utils import log_args, setup_logging


def main():
    # Read command-line arguments; anything not given falls back to the defaults in config.py.
    # 读取命令行参数；没有指定的参数使用 config.py 里的默认值。
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run_file",
        type=Path,
        default=config.DEFAULT_RUN_FILE,
        help="retrieval results written by step 3 / 步骤 3 写出的检索结果",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=config.QRELS_PATH,
        help="gold relevant passages per question (from the official dataset) / 每题的标准相关段落（官方数据集）",
    )
    # nargs="+": takes one or more values, e.g. --ks 3 10 100 -> [3, 10, 100]
    # nargs="+"：接受一个或多个值，例如 --ks 3 10 100 -> [3, 10, 100]
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=list(config.RECALL_KS),
        help="cut-offs k for R@k (Table 4 uses 3 10 100 1000) / 计算 R@k 的 k 值（Table 4 用 3 10 100 1000）",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=config.OUTPUT_DIR / "eval",
        help="where to write the result files / 结果文件保存目录",
    )
    args = parser.parse_args()

    # Start logging (terminal + results/logs/04_evaluate_retrieval_<time>.log) and record all arguments.
    # 开启日志（终端 + results/logs/04_evaluate_retrieval_<时间>.log），并记录全部参数。
    logger = setup_logging(Path(__file__).stem, config.LOG_DIR)
    log_args(logger, args)

    # 1. Retrieval results of step 3, ranks only (scores are not needed for recall):
    #    {"13_p569": ["70285879-0001", "22685-0005", ...], ...}   1000 passage ids per question, best first
    #    步骤 3 的检索结果，只保留排名（算召回不需要分数）：每题 1000 个段落 id，排名高的在前
    run = read_trec_run(args.run_file)

    # 2. Gold annotations: {"13_p569": {"15641-0001", "8212-0002", ...}, ...}
    #    A set, because only membership matters, not order. Only passages with relevance > 0 are kept.
    #    标准答案：每题一个集合（只关心"在不在"，不关心顺序）。只保留相关度 > 0 的段落。
    qrels = load_qrels(args.qrels)

    # 3. Compare question by question (see dra/evaluation.py):
    #    summary   = {"R@3": 0.93, ..., "num_queries": 1258, "num_skipped": 0}   mean over questions, in %
    #    per_query = [{"query_id": ..., "num_gold_entities": ..., "R@3": ..., ...}, ...]   one row per question, 0-1
    #    Questions without qrels are skipped, because recall is undefined when there is nothing to find.
    #    逐题对比（见 dra/evaluation.py）：
    #    summary   所有题的平均值，百分比；per_query 每题一行，0-1。
    #    没有标准答案的题会被跳过：没有需要找的东西，召回率无从定义。
    summary, per_query = evaluate_retrieval(run, qrels, args.ks)

    for k in args.ks:
        logger.info("R@%d: %.2f", k, summary[f"R@{k}"])
    logger.info("Evaluated queries: %d (skipped, no qrels: %d)", summary["num_queries"], summary["num_skipped"])

    # 4. Save both results. mkdir(parents=True, exist_ok=True): create the folder if missing, no error if it exists.
    #    保存两份结果。mkdir(...)：文件夹不存在就创建，已存在也不报错。
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "retrieval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    # jsonl = one JSON object per line, so the file can be read line by line.
    # jsonl：每行一个 JSON 对象，可以逐行读取。
    with open(args.output_dir / "retrieval_per_query.jsonl", "w", encoding="utf-8") as f:
        for row in per_query:
            f.write(json.dumps(row) + "\n")
    logger.info("Results written to %s", args.output_dir)


if __name__ == "__main__":
    main()
