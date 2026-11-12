"""
Step 5: generate answers and score them (Table 8).
步骤 5：生成答案并判分（Table 8）。

Two methods (--method) / 两种方法（--method）：
    single_retrieval  Table 8 "Single Step Retriever": the top-k passages retrieved in step 3 and the question
                      go to the LLM once; the answer is taken from <answer>...</answer>.
                      单次检索：步骤 3 检索到的前 k 个段落和问题一次性交给 LLM；答案从 <answer>...</answer> 取出。
    react             Table 8 "ReAct": the LLM searches by itself during the run (methods/react.py), so the
                      Contriever index (18 GB) is loaded here; the answer is taken from Finish[...].
                      ReAct：LLM 在运行中自己检索（methods/react.py），所以这里要加载 Contriever 索引（18 GB）；
                      答案从 Finish[...] 取出。

Scoring is the same as the official run_evalution.py: exact match, plus soft match at 1/5/10/20/50/90%
tolerance (Table 8 reports EM and SM 10/20/50/90).
判分与官方 run_evalution.py 相同：exact match，加上 1/5/10/20/50/90% 容差的 soft match
（Table 8 报告 EM 和 SM 10/20/50/90）。

Resuming (as in the official run_evalution.py): if <output> already exists, the questions in it are skipped
and new results are appended, so a job that stopped (time limit, error) can simply be submitted again.
For a fresh run, use a new --output or delete the old file. Note: the random seed is set again at every
start, so a resumed run does not continue the exact random sequence of the first one.
断点续跑（与官方 run_evalution.py 一样）：如果 <output> 已经存在，里面已有的题会被跳过，新结果追加在后面，
所以作业中途停止（超时、报错）后直接重新提交即可。想从头跑，就换一个 --output 或删掉旧文件。
注意：每次启动都会重新设置随机种子，所以续跑不会接着第一次的随机序列。

Outputs / 输出：
    <output>                    jsonl, one line per question: question, gold answer, prediction, scores, and
                                for analysis: single_retrieval -> raw output, has_answer_tag, num_passages,
                                token counts; react -> reasoning_path (passage ids of every Search),
                                llm_outputs, num_searches, forced_finish, token counts, entity_recall@k.
                                每题一行：问题、标准答案、预测、判分，以及分析用的字段（见上）。
    <output>_summary.json       EM and SM in % over all questions in <output>, plus method-specific counts
                                <output> 里所有题的 EM 和 SM（%），以及各方法自己的统计
    react only / 仅 react（in --runs_dir, not tracked by git / 不进 git）:
    <stem>.trec                 per question, the rankings of all Search steps merged as in the official code
                                (rank 1 of every step, then rank 2, ...), top 1000; run name contriever_sqN_rankR
                                每题把所有 Search 步骤的排序按官方方式合并（先取每步第 1 名，再取第 2 名……），前 1000 个
    <stem>_steps.jsonl          the full top-1000 ranking of every Search step, for analysing single steps
                                每一个 Search 步骤各自完整的前 1000 名，用于逐步分析

Usage / 用法：
    python scripts/05_generate_answers.py --limit 5                                # single retrieval, 5 questions
    python scripts/05_generate_answers.py --method react --limit 2 \\
        --index_path ... --corpus_path ... --passage_ids_cache ... --offsets_cache ...
    python scripts/05_generate_answers.py --dtype float16           # GPU without bf16 / 不支持 bf16 的 GPU
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from dra import config
from dra.data import load_passages, load_qrels, load_queries, read_trec_run
from dra.evaluation import entity_recall_at_k, load_official_answer_metric, score_answer, summarize_answer_scores
from dra.llm import DTYPES, LLM, extract_tag, parse_answer, resolve_dtype
from dra.logging_utils import log_args, setup_logging
from dra.utils import set_seed

METHODS = ("single_retrieval", "react")


def parse_args():
    # Read command-line arguments; anything not given falls back to the defaults in config.py.
    # 读取命令行参数；没有指定的参数使用 config.py 里的默认值。
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=METHODS, default="single_retrieval", help="how to answer / 答题方法")
    # --- Inputs / 输入 ---
    parser.add_argument(
        "--queries",
        type=Path,
        default=config.QUERIES_PATH,
        help="questions with gold answers / 题目及标准答案",
    )
    parser.add_argument(
        "--corpus_path",
        type=Path,
        default=config.CORPUS_PATH,
        help="corpus, to look up passage texts / 语料，用来取回段落原文",
    )
    # --- single_retrieval only / 仅单次检索 ---
    parser.add_argument(
        "--run_file",
        type=Path,
        default=config.DEFAULT_RUN_FILE,
        help="single_retrieval: retrieval results written by step 3 / 单次检索：步骤 3 写出的检索结果",
    )
    # --- react only / 仅 ReAct ---
    parser.add_argument(
        "--index_path",
        type=Path,
        default=config.INDEX_PATH,
        help="react: Contriever FAISS index / ReAct：Contriever 的 FAISS 索引",
    )
    parser.add_argument(
        "--passage_ids_cache",
        type=Path,
        default=config.PASSAGE_IDS_CACHE,
        help="react: cached position -> passage id table / ReAct：位置 -> 段落 id 映射表的缓存",
    )
    parser.add_argument(
        "--offsets_cache",
        type=Path,
        default=config.LINE_OFFSETS_CACHE,
        help="react: cached byte position of every corpus line / ReAct：语料每一行字节位置的缓存",
    )
    parser.add_argument(
        "--retriever_model",
        default=config.RETRIEVER_MODEL,
        help="react: query encoder, the model the index was built with / ReAct：问题编码模型，与建索引时一致",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=config.QRELS_PATH,
        help="react: relevant passages, for the retrieval metrics / ReAct：相关段落标注，用于检索指标",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=config.REACT_MAX_ITER,
        help="react: max rounds (official: 20) / ReAct：最多轮数（官方：20）",
    )
    parser.add_argument(
        "--eval_k",
        type=int,
        default=config.RETRIEVAL_K,
        help="react: passages retrieved per Search for evaluation (official: 1000) "
        "/ ReAct：每次 Search 为评估检索多少个（官方：1000）",
    )
    parser.add_argument(
        "--runs_dir",
        type=Path,
        default=config.RUNS_DIR,
        help="react: where the retrieval rankings go (not tracked by git) / ReAct：检索排序的存放位置（不进 git）",
    )
    # --- Reader / 阅读器 ---
    parser.add_argument(
        "--model",
        default=config.READER_MODEL,
        help="LLM that reads the passages and writes the answer / 读段落、写答案的大模型",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        default=config.READER_DTYPE,
        help="number format of the reader (official: bfloat16; RTX 2080 Ti: float16) "
        "/ 阅读器的数值格式（官方：bfloat16；RTX 2080 Ti：float16）",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help=f"passages put into the prompt (single_retrieval: {config.READER_TOPK} as in Table 8; "
        f"react: {config.REACT_TOPK} per Search, official default) "
        f"/ 放进 prompt 的段落数（单次检索：{config.READER_TOPK}，同 Table 8；"
        f"ReAct：每次 Search {config.REACT_TOPK} 个，官方默认）",
    )
    # --- Generation / 生成 ---
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=config.MAX_NEW_TOKENS,
        help="max tokens the LLM may write per call (1024 = official) / 模型每次最多写多少 token（1024 与官方一致）",
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
    # --- Run control / 运行控制 ---
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only the next N unfinished questions, for debugging / 只跑接下来 N 道还没做的题（调试用）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.OUTPUT_DIR / "generation" / "predictions.jsonl",
        help="per-question results; finished questions in it are skipped / 逐题结果文件；里面已有的题会被跳过",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="log progress every N questions / 每完成 N 题在日志里记一次进度",
    )
    args = parser.parse_args()
    if args.topk is None:
        args.topk = config.READER_TOPK if args.method == "single_retrieval" else config.REACT_TOPK
    return args


# ---------------------------------------------------------------------------
# Resuming / 断点续跑
# ---------------------------------------------------------------------------


def load_finished(output_path, logger):
    """
    Records already in the results file (empty list if there is none).
    If the last line is incomplete (the job was killed while writing it), it is removed from the file.
    结果文件里已有的记录（没有文件时为空列表）。
    如果最后一行不完整（作业在写这一行时被终止），把它从文件里删掉。
    """
    if not output_path.exists():
        return []
    lines = output_path.read_text(encoding="utf-8").splitlines()
    records = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise  # broken line in the middle: something else is wrong / 中间有坏行：说明有别的问题
            logger.warning("Removing incomplete last line of %s", output_path)
            output_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return records


def keep_finished_lines(path, finished_ids, query_id_of):
    """
    Remove lines of questions that are not in the results file from a react run file. A job can stop after a
    question's rankings were written but before its result line; that question is redone, so its old lines must go.
    从 ReAct 的检索排序文件里删掉不在结果文件里的题。作业可能在某题的排序已写出、结果行还没写出时停止；
    这道题会重做，所以旧的行要删掉。
    """
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [line for line in lines if query_id_of(line) in finished_ids]
    if len(kept) != len(lines):
        path.write_text("".join(kept), encoding="utf-8")


# ---------------------------------------------------------------------------
# Methods: each returns a function answer(question) -> dict
# 方法：每个都返回一个函数 answer(题目) -> 字典
#   "prediction"  the answer to score (None = no answer)   用来判分的答案（None = 没有答案）
#   "fields"      extra fields for the results line          结果行里的其他字段
#   "trec_lines", "step_lines"  react only: lines for the run files   仅 ReAct：检索排序文件的行
# ---------------------------------------------------------------------------


def make_single_retrieval(args, llm, queries, logger):
    from dra.methods.single_retrieval import SingleRetrieval

    # Retrieval results of step 3, and the passage texts they point to.
    # Lookup table: passage id -> passage record. First list every passage id any question needs
    # (top-k of each question, duplicates removed by the set), then scan the corpus once and keep only those.
    # 步骤 3 的检索结果，以及它们对应的段落原文。映射表：段落 id -> 段落记录。
    # 先列出所有题目需要的段落 id（每题前 k 个，集合自动去重），再扫描一遍语料，只保留这些。
    run = read_trec_run(args.run_file)
    needed = {doc_id for q in queries for doc_id in run.get(q["id"], [])[: args.topk]}
    id_to_passage = load_passages(args.corpus_path, wanted_ids=needed)
    logger.info("Passages needed: %d", len(needed))
    method = SingleRetrieval(llm)

    def answer(q):
        # Top-k passages (ids missing from the corpus are skipped, normally none) -> one LLM call
        # -> the text inside <answer>...</answer>, None if there is no tag.
        # 前 k 个段落（语料里找不到的 id 跳过，正常没有）-> 调用一次 LLM -> 取 <answer>...</answer>，没有标签时为 None。
        doc_ids = run.get(q["id"], [])[: args.topk]
        passages = [id_to_passage[d] for d in doc_ids if d in id_to_passage]
        raw = method.answer(q["question"], passages, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        return {
            "prediction": parse_answer(raw),
            "fields": {
                "raw_output": raw,
                # For analysis: whether the official <answer> tag is there, how many passages went into the
                # prompt, token counts (llm.last_usage).
                # 分析用：有没有官方要求的 <answer> 标签、prompt 里放了几个段落、token 数（llm.last_usage）。
                "has_answer_tag": extract_tag(raw, "answer") is not None,
                "num_passages": len(passages),
                **llm.last_usage,
            },
        }

    return answer


def make_react(args, llm, logger):
    from dra.data import CorpusReader, load_passage_ids
    from dra.methods.react import ReAct, interleave_rankings, reasoning_path_for_output
    from dra.retrieval import ContrieverEncoder, DenseRetriever, StepRetriever

    # Retriever, set up as in step 3 (official query settings: 64 tokens, fp16), plus the line positions of
    # the corpus, so the text of the passages the LLM should read can be fetched at once.
    # 检索器，和步骤 3 的设置一样（官方的问题编码设置：64 token、fp16），再加上语料的行位置表，
    # 以便马上取出要给 LLM 读的段落原文。
    passage_ids = load_passage_ids(args.corpus_path, cache_path=args.passage_ids_cache)
    encoder = ContrieverEncoder(
        args.retriever_model, max_length=config.QUERY_MAX_LENGTH, use_fp16=config.QUERY_USE_FP16
    )
    logger.info("Loading index %s", args.index_path)
    dense = DenseRetriever(args.index_path, passage_ids, encoder)
    logger.info("Index vectors: %d", dense.index.ntotal)
    corpus = CorpusReader(args.corpus_path, offsets_cache=args.offsets_cache)
    retriever = StepRetriever(dense, corpus, topk=args.topk, eval_k=args.eval_k)
    react = ReAct(llm, retriever, max_iter=args.max_iter)
    # Relevant passages per question, for entity recall of the merged ranking (as in the official code).
    # 每题的相关段落，用于计算合并排序的实体召回（与官方一致）。
    qrels = load_qrels(args.qrels)

    def answer(q):
        # The whole ReAct loop for one question. If no round has a valid action this raises an error and the run
        # stops, as in the official code; submitting again continues after the finished questions.
        # 一道题的完整 ReAct 循环。如果所有轮次都没有有效动作，这里会报错、运行停止，与官方一致；
        # 重新提交会接着已完成的题往下跑。
        reasoning_path, prediction = react.inference(q["question"], generation_temp=args.temperature)

        # Rankings of the Search steps -> merged ranking (official) -> entity recall and TREC lines.
        # 各 Search 步骤的排序 -> 合并排序（官方方式）-> 实体召回和 TREC 行。
        searches = [s for s in reasoning_path if s.get("action_type") == "search"]
        step_rankings = [s["ranked_list"] for s in searches]
        merged = interleave_rankings(step_rankings)
        fields = {
            "reasoning_path": reasoning_path_for_output(reasoning_path),
            "llm_outputs": react.llm_outputs,
            "num_searches": len(searches),
            "forced_finish": react.forced_finish,
            **react.usage,
        }
        # Official: entity recall only when there was a search and the question has qrels; in the summary,
        # questions without it count as 0. "entity_recall" uses the whole merged ranking (no cut-off).
        # 官方做法：只有检索过、且这道题有 qrels 时才算实体召回；汇总时没有的按 0 算。
        # "entity_recall" 用整个合并排序（不截断）。
        gold = qrels.get(q["id"])
        if merged and gold:
            merged_ids = [pid for pid, *_ in merged]
            for k in config.RECALL_KS:
                fields[f"entity_recall@{k}"] = entity_recall_at_k(merged_ids, gold, k)
            fields["entity_recall"] = entity_recall_at_k(merged_ids, gold, None)
        trec_lines = [
            f"{q['id']} Q0 {pid} {rank} {score:.4f} contriever_sq{step}_rank{rank_in_step}\n"
            for rank, (pid, score, step, rank_in_step) in enumerate(merged[: args.eval_k], start=1)
        ]
        step_lines = [
            json.dumps(
                {
                    "query_id": q["id"],
                    "step": n,
                    "search_query": s["search_query"],
                    "ranked": [[pid, round(score, 4)] for pid, score in s["ranked_list"]],
                }
            )
            + "\n"
            for n, s in enumerate(searches, start=1)
        ]
        return {"prediction": prediction, "fields": fields, "trec_lines": trec_lines, "step_lines": step_lines}

    return answer


# ---------------------------------------------------------------------------
# Summary / 汇总
# ---------------------------------------------------------------------------


def summarize(records, method, tolerances):
    """
    Scores and counts over all records of the results file (also those of earlier, resumed runs).
    结果文件里所有记录的分数和统计（包括之前被续跑的那些运行写下的记录）。
    """
    # json stores the tolerance keys as strings ("10.0"); turn them back into numbers.
    # json 把容差键存成了字符串（"10.0"）；转回数字。
    scores = [
        {"exact_match": r["exact_match"], "soft_matches": {float(k): v for k, v in r["soft_matches"].items()}}
        for r in records
    ]
    summary = summarize_answer_scores(scores, tolerances)
    n = max(len(records), 1)
    summary["seconds_per_question"] = sum(r["seconds"] for r in records) / n
    if method == "single_retrieval":
        summary["no_answer_tag"] = sum(r["prediction"] is None for r in records)
        summary["hit_max_new_tokens"] = sum(r["hit_max_new_tokens"] for r in records)
    else:
        summary["forced_finish"] = sum(r["forced_finish"] for r in records)
        summary["mean_num_searches"] = sum(r["num_searches"] for r in records) / n
        summary["mean_llm_calls"] = sum(r["llm_calls"] for r in records) / n
        # Entity recall of the merged ranking in %, questions without it counted as 0 (as in the official code).
        # 合并排序的实体召回（%），没有的题按 0 算（与官方一致）。
        for key in [f"entity_recall@{k}" for k in config.RECALL_KS] + ["entity_recall"]:
            summary[key] = 100 * sum(r.get(key, 0.0) for r in records) / n
    return summary


def main():
    args = parse_args()

    # Start logging (terminal + results/logs/05_generate_answers_<time>.log), record all arguments,
    # and fix the random seed (temperature > 0 makes generation random).
    # 开启日志并记录全部参数；固定随机种子（temperature > 0 时生成带随机性）。
    logger = setup_logging(Path(__file__).stem, config.LOG_DIR)
    log_args(logger, args)
    set_seed(args.seed)

    # 1. Questions, minus the ones already in the results file (resuming); --limit applies to the rest.
    #    题目，去掉结果文件里已有的（续跑）；--limit 作用于剩下的题。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    finished = load_finished(args.output, logger)
    finished_ids = {r["query_id"] for r in finished}
    queries = [q for q in load_queries(args.queries) if q["id"] not in finished_ids][: args.limit]
    logger.info("Already in %s: %d, to do now: %d", args.output, len(finished), len(queries))

    # React run files: drop lines of questions without a results line (see keep_finished_lines).
    # ReAct 的检索排序文件：删掉没有结果行的题（见 keep_finished_lines）。
    trec_path = args.runs_dir / f"{args.output.stem}.trec"
    steps_path = args.runs_dir / f"{args.output.stem}_steps.jsonl"
    if args.method == "react":
        args.runs_dir.mkdir(parents=True, exist_ok=True)
        keep_finished_lines(trec_path, finished_ids, lambda line: line.split()[0])
        keep_finished_lines(steps_path, finished_ids, lambda line: json.loads(line)["query_id"])

    if not queries:
        logger.info("Nothing to do: every question is already in %s", args.output)
        return

    # 2. The reader: LLM loads the model onto the GPU(s) (a too-large model fails here).
    #    阅读器：LLM 把模型加载到 GPU 上（模型太大会在这里报错）。
    logger.info("Loading reader %s (%s)", args.model, args.dtype)
    llm = LLM(args.model, dtype=resolve_dtype(args.dtype))
    # Where the weights ended up: number of weight tensors per device, e.g. {"cuda:0": 169, "cuda:1": 170}
    # means the model is split over two GPUs, {"cpu": ...} means no GPU was used.
    # 权重被放在了哪里：每个设备上的权重张量个数，例如 {"cuda:0": 169, "cuda:1": 170} 表示模型分在两块 GPU 上，
    # {"cpu": ...} 表示没有用到 GPU。
    params_per_device = Counter(str(p.device) for p in llm.model.parameters())
    logger.info("Weights per device: %s", dict(params_per_device))

    # 3. The method: answer(question) -> prediction and fields (see the functions above).
    #    方法：answer(题目) -> 预测和字段（见上面的函数）。
    if args.method == "single_retrieval":
        answer = make_single_retrieval(args, llm, queries, logger)
    else:
        answer = make_react(args, llm, logger)

    # Official scoring function from external/total-recall-qa, so scores match the paper.
    # 官方判分函数（来自 external/total-recall-qa），保证判分与论文一致。
    soft_exact_match = load_official_answer_metric(config.OFFICIAL_GEN_METRICS)
    tolerances = config.SOFT_MATCH_TOLERANCES

    new_records = []
    start = time.time()
    # "a" = append: new lines go after the finished ones.   "a" = 追加：新行写在已有的行后面。
    with open(args.output, "a", encoding="utf-8") as f:
        # 4. Answer the questions one by one. tqdm shows a progress bar; enumerate(start=1) gives i = 1, 2, ...
        #    逐题作答。tqdm 显示进度条；enumerate(start=1) 给出题号 i = 1, 2, ...
        for i, q in enumerate(tqdm(queries, desc="generate"), start=1):
            # a. The method answers. The slowest step.   方法作答。最慢的一步。
            q_start = time.time()
            out = answer(q)
            pred = out["prediction"]

            # b. Score against the gold answer, as in the official code:
            #    {"exact_match": bool, "soft_matches": {1.0: bool, 5.0: bool, ..., 90.0: bool}}.
            #    和标准答案对比，与官方一致：
            #    {"exact_match": bool, "soft_matches": {1.0: bool, 5.0: bool, ..., 90.0: bool}}。
            score = score_answer(soft_exact_match, pred, q["answer"], tolerances, decimals=config.ANSWER_DECIMALS)

            # c. React: write this question's rankings first, its results line last. The results line marks the
            #    question as finished; if the job stops in between, resuming removes these lines again.
            #    ReAct：先写这道题的检索排序，最后写结果行。结果行表示这道题已完成；
            #    如果作业在中间停止，续跑时会把这些行删掉。
            if args.method == "react":
                with open(trec_path, "a", encoding="utf-8") as tf:
                    tf.writelines(out["trec_lines"])
                with open(steps_path, "a", encoding="utf-8") as sf:
                    sf.writelines(out["step_lines"])

            # d. Write one line for this question. **score unpacks it into "exact_match" and "soft_matches" keys
            #    (json writes the tolerance keys as strings, e.g. "10.0", like the official results);
            #    ensure_ascii=False keeps non-English characters readable; flush() writes to disk immediately,
            #    so finished questions are kept if the run crashes.
            #    写入这道题的一行。**score 展开成 "exact_match"、"soft_matches" 两个键（json 把容差键写成字符串，
            #    如 "10.0"，与官方结果相同）；ensure_ascii=False 让非英文字符保持可读；
            #    flush() 立即写入硬盘，崩溃时已完成的题不会丢失。
            record = {
                "query_id": q["id"],
                "question": q["question"],
                "gold": q["answer"],
                "prediction": pred,
                **score,
                **out["fields"],
                "seconds": round(time.time() - q_start, 2),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            new_records.append(record)

            # e. Log progress every log_every questions, with the average time per question (on the cluster this
            #    tells whether all questions fit in the job's time limit).
            #    每 log_every 题在日志里记一次进度和每题平均耗时（在集群上可以判断作业时间够不够跑完）。
            if i % args.log_every == 0:
                em = 100 * sum(r["exact_match"] for r in new_records) / i
                logger.info(
                    "[%d/%d] EM so far (this run): %.2f%%, %.1f s/question",
                    i,
                    len(queries),
                    em,
                    (time.time() - start) / i,
                )

    # 5. Summary over all questions in the results file: average the scores, log them and write them next to it.
    #    汇总结果文件里的所有题：把判分平均，记进日志，并写到逐题结果旁边。
    records = load_finished(args.output, logger)
    summary = summarize(records, args.method, tolerances)
    logger.info("Questions in %s: %d (this run: %d)", args.output, len(records), len(new_records))
    logger.info("Exact match: %.2f%%", summary["exact_match"])
    for pct in tolerances:
        logger.info("Soft match %4.0f%%: %.2f%%", pct, summary["soft_match"][pct])
    for key, value in summary.items():
        if key not in ("exact_match", "soft_match", "num_queries"):
            logger.info("%s: %s", key, round(value, 4) if isinstance(value, float) else value)

    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Predictions written to %s, summary to %s", args.output, summary_path)
    if args.method == "react":
        logger.info("Rankings written to %s and %s", trec_path, steps_path)


if __name__ == "__main__":
    main()
