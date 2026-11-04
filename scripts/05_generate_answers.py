"""
Step 5: generate answers and score them (single-retrieval RAG baseline, Table 8 "Single Step Retriever").
步骤 5：生成答案并判分（单次检索 RAG 基线，对应 Table 8 的 "Single Step Retriever"）。

For every question: take the top-k passages retrieved in step 3 -> give them to the LLM together with
the question -> extract the answer from <answer>...</answer> -> compare it with the gold answer using
the official soft_exact_match. Retrieval happens once and the LLM is called once per question;
it cannot search again (that is what the deep research agents do).
对每道题：取步骤 3 检索到的前 k 个段落 -> 和问题一起交给 LLM -> 从 <answer>...</answer> 取出答案
-> 用官方 soft_exact_match 和标准答案对比。每题只检索一次、只调用一次模型，不能再去搜索
（再次搜索是 DRA 做的事）。

Output / 输出：<output> (jsonl), one line per question with the question, gold answer, the model's full
output, the extracted answer and the scores. Written line by line, so finished questions survive a crash.
每题一行：问题、标准答案、模型完整输出、取出的答案、判分。逐行写入，中途崩溃时已完成的题不会丢失。

Usage / 用法：
    python scripts/05_generate_answers.py --limit 5                                # first 5 questions / 先跑 5 题
    python scripts/05_generate_answers.py --model Qwen/Qwen2.5-1.5B-Instruct --limit 5   # local debugging / 本地调试
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from dra import config
from dra.data import load_passages, load_queries, read_trec_run
from dra.evaluation import load_official_answer_metric
from dra.llm import LLM, parse_answer
from dra.logging_utils import log_args, setup_logging
from dra.methods.single_retrieval import SingleRetrieval
from dra.utils import set_seed


def main():
    # Read command-line arguments; anything not given falls back to the defaults in config.py.
    # 读取命令行参数；没有指定的参数使用 config.py 里的默认值。
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- Inputs / 输入 ---
    parser.add_argument(
        "--queries",
        type=Path,
        default=config.QUERIES_PATH,
        help="questions with gold answers / 题目及标准答案",
    )
    parser.add_argument(
        "--run_file",
        type=Path,
        default=config.DEFAULT_RUN_FILE,
        help="retrieval results written by step 3 / 步骤 3 写出的检索结果",
    )
    parser.add_argument(
        "--corpus_path",
        type=Path,
        default=config.CORPUS_PATH,
        help="corpus, to look up passage texts by id / 语料，用 id 取回段落原文",
    )
    # --- Reader / 阅读器 ---
    parser.add_argument(
        "--model",
        default=config.READER_MODEL,
        help="LLM that reads the passages and writes the answer / 读段落、写答案的大模型",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=config.READER_TOPK,
        help="passages put into the prompt (Table 8: K=50) / 放进 prompt 的段落数（Table 8：K=50）",
    )
    # --- Generation / 生成 ---
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=config.MAX_NEW_TOKENS,
        help="max tokens the LLM may write (1024 = official) / 模型最多写多少 token（1024 与官方一致）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.TEMPERATURE,
        help="randomness of sampling; 0 = always the most likely token / 采样随机性，0 = 总取概率最高的",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help="random seed, makes sampling reproducible / 随机种子，使采样结果可复现",
    )
    # --- Scoring / 判分 ---
    parser.add_argument(
        "--answer_fallback",
        action="store_true",
        help="if there is no <answer> tag, use the last line (NOT the official setting) "
        "/ 没有 <answer> 标签时取最后一行作答案（非官方设定）",
    )
    # --- Run control / 运行控制 ---
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only the first N questions, for debugging / 只跑前 N 题（调试用）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.OUTPUT_DIR / "generation" / "predictions.jsonl",
        help="per-question results / 逐题结果文件",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="log progress every N questions / 每完成 N 题在日志里记一次进度",
    )
    args = parser.parse_args()

    # Start logging (terminal + results/logs/05_generate_answers_<time>.log), record all arguments,
    # and fix the random seed (temperature > 0 makes generation random).
    # 开启日志并记录全部参数；固定随机种子（temperature > 0 时生成带随机性）。
    logger = setup_logging(Path(__file__).stem, config.LOG_DIR)
    log_args(logger, args)
    set_seed(args.seed)

    # 1. Questions (with gold answers) and the retrieval results of step 3.
    #    [:None] keeps everything, so without --limit all 1258 questions are used.
    #    题目（含标准答案）和步骤 3 的检索结果。[:None] 表示全部，所以不加 --limit 就是全部 1258 题。
    queries = load_queries(args.queries)[: args.limit]
    run = read_trec_run(args.run_file)

    # 2. Lookup table 2: passage id -> passage record {"id", "contents"} (same dict as the official retriever).
    #    First list every passage id any question needs (top-k of each question, duplicates removed by the set),
    #    then scan the corpus once and keep only those. Scanning once (~12 s) is much faster than
    #    scanning once per question, and only ~31k of the 5.9M passages are kept in memory.
    #    映射表 2：段落 id -> 段落记录 {"id", "contents"}（与官方检索器返回的字典相同）。
    #    先列出所有题目需要的段落 id（每题前 k 个，集合自动去重），再扫描一遍语料，只保留这些。
    #    只扫一遍（约 12 秒）比每题各扫一遍快得多，内存里也只放约 3 万个段落，而不是 590 万个。
    needed = {doc_id for q in queries for doc_id in run.get(q["id"], [])[: args.topk]}
    id_to_passage = load_passages(args.corpus_path, wanted_ids=needed)
    logger.info("Queries: %d, passages needed: %d", len(queries), len(needed))

    # 3. Prepare the reader: LLM loads the model onto the GPU (a too-large model fails here);
    #    SingleRetrieval defines how it answers: one prompt with all passages, one generate call.
    #    A deep research agent would be another class in dra/methods/ using the same LLM.
    #    准备阅读器：LLM 把模型加载到 GPU（模型太大会在这里报错）；
    #    SingleRetrieval 规定答题方式：一个 prompt 放进全部段落，只生成一次。
    #    DRA 会是 dra/methods/ 里的另一个类，用同一个 LLM。
    logger.info("Loading reader %s", args.model)
    method = SingleRetrieval(LLM(args.model))
    # Official scoring function from external/total-recall-qa, so scores match the paper.
    # 官方判分函数（来自 external/total-recall-qa），保证判分与论文一致。
    soft_exact_match = load_official_answer_metric(config.OFFICIAL_GEN_METRICS)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_correct = 0  # questions answered exactly right / 完全答对的题数
    n_no_answer = 0  # questions without an <answer> tag (and no fallback) / 没有 <answer> 标签（且未用 fallback）的题数
    with open(args.output, "w", encoding="utf-8") as f:
        # 4. Answer the questions one by one. tqdm shows a progress bar; enumerate(start=1) gives i = 1, 2, ...
        #    逐题作答。tqdm 显示进度条；enumerate(start=1) 给出题号 i = 1, 2, ...
        for i, q in enumerate(tqdm(queries, desc="generate"), start=1):
            # a. Records of this question's top-k passages (skip ids missing from the corpus, normally none).
            #    这道题前 k 个段落的记录（语料里找不到的 id 跳过，正常情况下没有）。
            doc_ids = run.get(q["id"], [])[: args.topk]
            passages = [id_to_passage[d] for d in doc_ids if d in id_to_passage]

            # b. The LLM writes its answer, e.g. "<think>...</think><answer>1886</answer>". Slowest step.
            #    模型写出回答，例如 "<think>...</think><answer>1886</answer>"。最慢的一步。
            raw = method.answer(
                q["question"],
                passages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            # c. Extract the text inside <answer>...</answer>; None if there is no tag.
            #    取出 <answer>...</answer> 里的内容；没有标签时为 None。
            pred = parse_answer(raw, fallback=args.answer_fallback)

            # d. Score against the gold answer: {"exact_match": bool, "soft_match": bool}.
            #    TODO: Table 8 needs EM + SM at 10/20/50/90% tolerance; only 5% is computed here for now.
            #    和标准答案对比，返回 {"exact_match": bool, "soft_match": bool}。
            #    TODO：Table 8 需要 EM + SM 10/20/50/90% 容差，这里目前只算了 5%。
            score = soft_exact_match(pred, q["answer"], tolerance_pct=5.0)

            # e. Count. True counts as 1 and False as 0, so this adds 1 only when the condition holds.
            #    计数。True 当作 1、False 当作 0，所以条件成立时才加 1。
            n_correct += score["exact_match"]
            n_no_answer += pred is None

            # f. Write one line for this question. **score unpacks it into "exact_match" and "soft_match" keys;
            #    ensure_ascii=False keeps non-English characters readable; flush() writes to disk immediately,
            #    so finished questions are kept if the run crashes.
            #    写入这道题的一行。**score 展开成 "exact_match"、"soft_match" 两个键；
            #    ensure_ascii=False 让非英文字符保持可读；flush() 立即写入硬盘，崩溃时已完成的题不会丢失。
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

            # g. Log progress every log_every questions (useful on the cluster, where nobody watches the terminal).
            #    每 log_every 题在日志里记一次进度（在集群上没人盯着终端时很有用）。
            if i % args.log_every == 0:
                logger.info("[%d/%d] EM so far: %.2f%%", i, len(queries), 100 * n_correct / i)

    # 5. Summary. max(..., 1) avoids dividing by zero when there are no questions.
    #    汇总。max(..., 1) 避免没有题目时除以 0。
    n = max(len(queries), 1)
    logger.info("Exact match: %d/%d = %.2f%%", n_correct, len(queries), 100 * n_correct / n)
    logger.info("No <answer> tag: %d/%d = %.2f%%", n_no_answer, len(queries), 100 * n_no_answer / n)
    logger.info("Predictions written to %s", args.output)


if __name__ == "__main__":
    main()
