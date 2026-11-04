"""
步骤 4：检索评估。

读取 TREC run 文件和 qrels，计算实体级召回 Entity Recall@k。
输出：
    <output_dir>/retrieval_summary.json      平均 R@k
    <output_dir>/retrieval_per_query.jsonl   逐题 R@k（便于错误分析、显著性检验）

用法：
    python scripts/04_evaluate_retrieval.py
    python scripts/04_evaluate_retrieval.py --ks 3 10 100 1000
"""

import argparse
import json
from pathlib import Path

from trqa import config
from trqa.data import load_qrels, read_trec_run
from trqa.evaluation import evaluate_retrieval


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_file", type=Path, default=config.DEFAULT_RUN_FILE)
    parser.add_argument("--qrels", type=Path, default=config.QRELS_PATH)
    parser.add_argument("--ks", type=int, nargs="+", default=list(config.RECALL_KS))
    parser.add_argument("--output_dir", type=Path, default=config.OUTPUT_DIR / "eval")
    args = parser.parse_args()

    run = read_trec_run(args.run_file)
    qrels = load_qrels(args.qrels)
    summary, per_query = evaluate_retrieval(run, qrels, args.ks)

    for k in args.ks:
        print(f"R@{k}: {summary[f'R@{k}']:.2f}")
    print(f"评估 query 数: {summary['num_queries']}（无金标跳过: {summary['num_skipped']}）")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "retrieval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(args.output_dir / "retrieval_per_query.jsonl", "w", encoding="utf-8") as f:
        for row in per_query:
            f.write(json.dumps(row) + "\n")
    print(f"结果已写入 {args.output_dir}")


if __name__ == "__main__":
    main()
