"""
步骤 3：检索。

用 Contriever 编码全部测试 query，在 FAISS 索引中检索 Top-K 段落，
结果写成 TREC run 文件。之后的评估和生成都只读这个文件，
不必每次重新加载 18 GB 的索引。

用法：
    python scripts/03_retrieve.py
    python scripts/03_retrieve.py --k 1000 --output results/runs/contriever_wiki2_test.trec
"""

import argparse
from pathlib import Path

from trqa import config
from trqa.data import load_passage_ids, load_queries, write_trec_run
from trqa.retrieval import ContrieverEncoder, DenseRetriever


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=Path, default=config.QUERIES_PATH)
    parser.add_argument("--corpus_path", type=Path, default=config.CORPUS_PATH)
    parser.add_argument("--index_path", type=Path, default=config.INDEX_PATH)
    parser.add_argument("--passage_ids_cache", type=Path, default=config.PASSAGE_IDS_CACHE)
    parser.add_argument("--model", default=config.RETRIEVER_MODEL)
    parser.add_argument("--k", type=int, default=config.RETRIEVAL_K)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=config.DEFAULT_RUN_FILE)
    args = parser.parse_args()

    queries = load_queries(args.queries)
    print(f"queries: {len(queries)}")

    # 映射表 1：索引位置 -> doc_id
    passage_ids = load_passage_ids(args.corpus_path, cache_path=args.passage_ids_cache)

    encoder = ContrieverEncoder(args.model, max_length=config.MAX_LENGTH)
    retriever = DenseRetriever(args.index_path, passage_ids, encoder)
    print(f"索引向量数: {retriever.index.ntotal}")

    ranked = retriever.search([q["question"] for q in queries], k=args.k, batch_size=args.batch_size)
    results = {q["id"]: r for q, r in zip(queries, ranked, strict=True)}

    write_trec_run(args.output, results, run_name="contriever")
    first = queries[0]["id"]
    print(f"示例 {first} -> {[d for d, _ in results[first][:3]]}")
    print(f"检索结果已写入 {args.output}")


if __name__ == "__main__":
    main()
