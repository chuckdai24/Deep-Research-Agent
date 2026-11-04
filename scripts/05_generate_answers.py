"""
步骤 5：生成答案并判分（单次检索 RAG）。

对每个 query：取检索结果的 Top-k 段落原文 -> 与问题一起交给 LLM -> 解析 <answer>
-> 用官方 soft_exact_match 与标准答案对比。

输出 <output>（jsonl），每行包含问题、标准答案、模型原始输出、解析出的答案和判分，
运行中逐行写入，中断后也能保留已完成的部分。

用法：
    python scripts/05_generate_answers.py --limit 5          # 先跑 5 题看看效果
    python scripts/05_generate_answers.py --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from trqa import config
from trqa.data import load_passage_texts, load_queries, read_trec_run
from trqa.evaluation import load_official_answer_metric
from trqa.llm import LLM, parse_answer
from trqa.methods.single_retrieval import SingleRetrieval


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=Path, default=config.QUERIES_PATH)
    parser.add_argument("--run_file", type=Path, default=config.DEFAULT_RUN_FILE)
    parser.add_argument("--corpus_path", type=Path, default=config.CORPUS_PATH)
    parser.add_argument("--model", default=config.READER_MODEL)
    parser.add_argument("--topk", type=int, default=config.READER_TOPK, help="放进 prompt 的段落数")
    parser.add_argument("--max_new_tokens", type=int, default=config.MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题（调试用）")
    parser.add_argument(
        "--answer_fallback", action="store_true", help="没有 <answer> 标签时取最后一行作答案（非官方设定）"
    )
    parser.add_argument("--output", type=Path, default=config.OUTPUT_DIR / "generation" / "predictions.jsonl")
    args = parser.parse_args()

    queries = load_queries(args.queries)[: args.limit]
    run = read_trec_run(args.run_file)

    # 映射表 2：只加载这些 query 的 Top-k 段落原文，避免把整个语料读进内存
    needed = {doc_id for q in queries for doc_id in run.get(q["id"], [])[: args.topk]}
    id_to_text = load_passage_texts(args.corpus_path, wanted_ids=needed)

    method = SingleRetrieval(LLM(args.model))
    soft_exact_match = load_official_answer_metric(config.OFFICIAL_GEN_METRICS)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_correct = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for q in tqdm(queries, desc="generate"):
            doc_ids = run.get(q["id"], [])[: args.topk]
            passages = [id_to_text[d] for d in doc_ids if d in id_to_text]

            raw = method.answer(
                q["question"], passages, max_new_tokens=args.max_new_tokens, temperature=args.temperature
            )
            pred = parse_answer(raw, fallback=args.answer_fallback)
            score = soft_exact_match(pred, q["answer"], tolerance_pct=5.0)
            n_correct += score["exact_match"]

            f.write(
                json.dumps(
                    {
                        "query_id": q["id"],
                        "question": q["question"],
                        "gold": q["answer"],
                        "prediction": pred,
                        "raw_output": raw,
                        **score,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()

    print(f"Exact match: {n_correct}/{len(queries)} = {100 * n_correct / max(len(queries), 1):.2f}%")
    print(f"预测结果已写入 {args.output}")


if __name__ == "__main__":
    main()
