"""
Step 3: dense retrieval.
步骤 3：稠密检索。

Encode all test questions with Contriever, search the FAISS index for the top-K passages of each
question, and write the results to a TREC run file. Steps 4 and 5 only read this file, so the
18 GB index does not have to be loaded again.
用 Contriever 编码全部测试问题，在 FAISS 索引里为每个问题检索 Top-K 段落，结果写成 TREC run 文件。
步骤 4、5 只读这个文件，不必再加载 18 GB 的索引。

Idea / 思路：
    question text --encode--> 768-dim vector --FAISS inner product--> top-K positions
                  --passage_ids--> top-K passage ids
    问题文字 --编码--> 768 维向量 --FAISS 内积搜索--> 前 K 个位置 --passage_ids--> 前 K 个段落 id

Usage / 用法：
    python scripts/03_retrieve.py
    python scripts/03_retrieve.py --k 1000 --output results/runs/contriever_wiki2_test.trec
"""

import argparse
from pathlib import Path

from dra import config
from dra.data import load_passage_ids, load_queries, write_trec_run
from dra.logging_utils import log_args, setup_logging
from dra.retrieval import ContrieverEncoder, DenseRetriever


def main():
    # Read command-line arguments; anything not given falls back to the defaults in config.py.
    # 读取命令行参数；没有指定的参数使用 config.py 里的默认值。
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--queries",
        type=Path,
        default=config.QUERIES_PATH,
        help="questions to retrieve for / 题目文件",
    )
    parser.add_argument(
        "--corpus_path",
        type=Path,
        default=config.CORPUS_PATH,
        help="corpus, only read if the passage_ids cache is missing / 语料，仅在映射表缓存不存在时读取",
    )
    parser.add_argument(
        "--index_path",
        type=Path,
        default=config.INDEX_PATH,
        help="FAISS index built in step 2 / 步骤 2 建好的索引",
    )
    parser.add_argument(
        "--passage_ids_cache",
        type=Path,
        default=config.PASSAGE_IDS_CACHE,
        help="cached position -> passage id table / 位置 -> 段落 id 映射表的缓存",
    )
    parser.add_argument(
        "--model",
        default=config.RETRIEVER_MODEL,
        help="query encoder; must be the model the index was built with / 问题编码模型，必须和建索引时一致",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=config.RETRIEVAL_K,
        help="passages per question / 每题取前多少个",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.QUERY_BATCH_SIZE,
        help="questions encoded per batch (official: 512) / 每批编码的问题数（官方：512）",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=config.QUERY_MAX_LENGTH,
        help="max tokens per question (official: 64) / 每个问题最多多少 token（官方：64）",
    )
    # BooleanOptionalAction gives both --fp16 and --no-fp16; the default (True) matches the official code.
    # BooleanOptionalAction 同时提供 --fp16 和 --no-fp16 两个开关；默认 True，与官方一致。
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=config.QUERY_USE_FP16,
        help="encode questions in half precision on the GPU (official: yes) / 在 GPU 上用半精度编码问题（官方：是）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.DEFAULT_RUN_FILE,
        help="TREC run file / 结果文件",
    )
    args = parser.parse_args()

    # Start logging (terminal + results/logs/03_retrieve_<time>.log) and record all arguments.
    # 开启日志（终端 + results/logs/03_retrieve_<时间>.log），并记录全部参数。
    logger = setup_logging(Path(__file__).stem, config.LOG_DIR)
    log_args(logger, args)

    # 1. Read the questions: a list of {"id", "question", "answer", "src"}.
    #    读取题目：每道题是 {"id", "question", "answer", "src"}。
    queries = load_queries(args.queries)
    logger.info("Queries: %d", len(queries))

    # 2. Lookup table: index position -> passage id (passage_ids[i] = id of corpus line i).
    #    FAISS only stores vectors, so a search returns positions, not ids. Both the official
    #    index_builder and load_passage_ids go through the corpus line by line, so position i
    #    in the index and passage_ids[i] both refer to corpus line i.
    #    映射表：索引位置 -> 段落 id（passage_ids[i] = 语料第 i 行的 id）。
    #    FAISS 只存向量，搜索返回的是位置而不是 id。官方 index_builder 和 load_passage_ids
    #    都按语料的行顺序处理，所以索引位置 i 和 passage_ids[i] 都对应语料第 i 行。
    passage_ids = load_passage_ids(args.corpus_path, cache_path=args.passage_ids_cache)

    # 3. Prepare the tools (nothing is searched yet):
    #    encoder   loads Contriever, turns text into 768-dim vectors
    #    retriever loads the 18 GB index into memory and checks it has as many vectors as passage_ids
    #    准备工具（还没有开始检索）：
    #    encoder   加载 Contriever，把文字变成 768 维向量
    #    retriever 把 18 GB 索引读进内存，并检查向量数与 passage_ids 长度一致
    #    Query settings (max length 64, fp16, batch 512) match the official retrieval code.
    #    问题的编码设置（最长 64、fp16、每批 512）与官方检索代码一致。
    encoder = ContrieverEncoder(args.model, max_length=args.max_length, use_fp16=args.fp16)
    if args.fp16 and not encoder.use_fp16:
        logger.warning("No GPU: questions are encoded in fp32 instead of fp16; scores may differ very slightly")
    retriever = DenseRetriever(args.index_path, passage_ids, encoder)
    logger.info("Index vectors: %d", retriever.index.ntotal)  # should be 5,910,177 / 应为 5910177

    # 4. Search: encode all questions, find the k most similar passages for each, map positions to ids.
    #    ranked[i] belongs to queries[i]: [(passage_id, score), ...] sorted by score, highest first.
    #    A high score means "similar to the question", not necessarily "relevant".
    #    检索：编码全部问题，为每题找最相似的 k 个段落，再把位置翻译成 id。
    #    ranked[i] 对应 queries[i]：[(段落 id, 分数), ...]，按分数从高到低。
    #    分数高只说明"和问题相似"，不一定"真的相关"。
    ranked = retriever.search([q["question"] for q in queries], k=args.k, batch_size=args.batch_size)

    # 5. Pair each question id with its results: {"13_p569": [(passage_id, score), ...], ...}
    #    strict=True: raise an error if the two lists differ in length, instead of silently misaligning.
    #    把题目 id 和结果配对。strict=True：两边数量不一致就报错，而不是悄悄错位。
    results = {q["id"]: r for q, r in zip(queries, ranked, strict=True)}

    # 6. Write the TREC run file, one line per (question, passage): "qid Q0 passage_id rank score contriever".
    #    写 TREC run 文件，每行一个（题目, 段落）："题目id Q0 段落id 排名 分数 contriever"。
    write_trec_run(args.output, results, run_name="contriever")

    # Show the top 3 of the first question as a sanity check (same as in the original notebook).
    # 打印第一道题的前 3 个结果，确认程序正常（与最初 notebook 的结果一致）。
    first = queries[0]["id"]
    logger.info("Example %s -> %s", first, [d for d, _ in results[first][:3]])
    logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
