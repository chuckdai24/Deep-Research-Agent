"""
数据读写。

包括四类数据：
  1. queries   —— 测试问题与标准答案（jsonl）
  2. qrels     —— 每个问题的相关段落标注（TREC qrels 格式）
  3. corpus    —— 维基段落语料（jsonl），以及"索引位置 -> doc_id"、"doc_id -> 原文"两张映射表
  4. run 文件  —— 检索结果（TREC run 格式），把检索和后续评估 / 生成解耦
"""

import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# queries / qrels
# ---------------------------------------------------------------------------


def load_queries(path):
    """
    读取 queries jsonl。

    每行形如：
        {"id": "13_p569", "question": "...", "answer": 1886, "src": "qald10"}
    返回保持文件顺序的列表。
    """
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_qrels(path):
    """
    读取 TREC qrels，每行为 "query_id iteration doc_id relevance"。

    返回 {query_id: 相关 passage id 的集合}，只保留 relevance > 0 的行。
    """
    qrels = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            qid, _iteration, doc_id, relevance = line.split()
            if int(relevance) > 0:
                qrels[qid].add(doc_id)
    return dict(qrels)


# ---------------------------------------------------------------------------
# 语料
# ---------------------------------------------------------------------------


def export_corpus(output_path, split="wiki_partial"):
    """
    从 HuggingFace (mahtaa/trqa) 下载语料，转换成官方 index_builder 需要的格式。

    输出每行：{"id": ..., "contents": "title\\ncontents"}
    之所以把标题拼在正文第一行：index_builder 和官方 passages2string 都约定
    "contents 的第一行是标题"，按这个格式存，后面才能正确还原标题。
    """
    from datasets import load_dataset

    corpus = load_dataset("mahtaa/trqa", "corpus", split=split)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in tqdm(corpus, desc="export corpus"):
            record = {"id": doc["id"], "contents": f"{doc['title']}\n{doc['contents']}"}
            f.write(json.dumps(record) + "\n")
    return len(corpus)


def write_sample(corpus_path, output_path, n):
    """取语料前 n 行另存为小样本，用于快速调试建索引流程。"""
    with open(corpus_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= n:
                break
            fout.write(line)


def iter_corpus(corpus_path, desc="read corpus"):
    """
    Read the corpus jsonl line by line, from the first line to the last; yields {"id", "contents"} dicts.
    从第一行到最后一行逐行读取语料 jsonl，每次产出一个 {"id", "contents"} 字典。
    """
    with open(corpus_path, encoding="utf-8") as f:
        for line in tqdm(f, desc=desc, unit=" docs"):
            yield json.loads(line)


def load_passage_ids(corpus_path, cache_path=None):
    """
    Lookup table 1: index position -> passage id (a list; the list index is the position).
    映射表 1：索引位置 -> 段落 id（返回列表，列表下标就是位置）。

    Why: FAISS only stores vectors, so a search returns "the i-th vector", which has to be mapped
    to a passage id before it can be compared with the qrels.
    为什么需要：FAISS 只存向量，检索返回的是"第 i 个向量"，必须翻译成段落 id 才能和 qrels 对比。

    Why the order matches: the official index_builder.py (external/total-recall-qa/c2_corpus_creation/)
    loads the corpus in file order (load_corpus, line 145), encodes it in consecutive batches and
    concatenates them in order (encode_all, lines 487-557), then adds the whole matrix at once
    (faiss_index.add, line 609). So index position i = corpus line i. Here the same corpus file is read
    line by line, so passage_ids[i] = id of corpus line i as well.
    为什么顺序能对上：官方 index_builder.py 按文件顺序读语料（load_corpus，第 145 行），
    按顺序分批编码并依次拼接（encode_all，第 487-557 行），再把整个矩阵一次加入索引
    （faiss_index.add，第 609 行），所以索引位置 i = 语料第 i 行。这里逐行读取同一个语料文件，
    所以 passage_ids[i] 也 = 语料第 i 行的 id。

    Therefore the corpus file must be the one the index was built from, in the same line order.
    因此语料必须是建索引时用的那一个，行顺序也不能变。

    cache_path: read it if it exists; otherwise scan the corpus (~15 s) and write it for next time.
    cache_path：存在就直接读；不存在就扫描语料（约 15 秒）后写入，下次直接读。
    """
    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path, encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]

    # One id per corpus line, in order / 每行语料取一个 id，按顺序
    passage_ids = [doc["id"] for doc in iter_corpus(corpus_path, desc="build passage_ids")]

    if cache_path is not None:
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write("\n".join(passage_ids) + "\n")
    return passage_ids


def load_passages(corpus_path, wanted_ids=None):
    """
    Lookup table 2: passage id -> passage record {"id": ..., "contents": "title\\ntext"}.
    映射表 2：段落 id -> 段落记录 {"id": ..., "contents": "标题\\n正文"}。

    Why: the LLM reads text, not vectors, so the passages are fetched by id.
    为什么需要：生成阶段 LLM 读的是文字而不是向量，要用段落 id 取回原文。

    Each record is the corpus line as is, i.e. the same dict the official retriever returns
    (load_docs in external/total-recall-qa/c5_task_evaluation/src/retrieval_models_local.py),
    so it can be passed to the prompt exactly like in the official code.
    每条记录就是语料里的那一行原样，与官方检索器返回的字典相同（官方 load_docs），
    所以放进 prompt 的方式和官方代码完全一致。

    With wanted_ids only those passages are kept (generation needs only the top-k), instead of
    holding all 5.9M passages in memory; scanning stops once all of them are found.
    给了 wanted_ids 时只保留这些段落（生成只需要前 k 个），不必把 591 万段落都放进内存；
    全部找齐后提前停止扫描。
    """
    wanted = set(wanted_ids) if wanted_ids is not None else None
    passages = {}
    for doc in iter_corpus(corpus_path, desc="load passages"):
        if wanted is None or doc["id"] in wanted:
            passages[doc["id"]] = doc
            if wanted is not None and len(passages) == len(wanted):
                break  # all found, stop scanning / 需要的都找到了，提前结束扫描
    return passages


# ---------------------------------------------------------------------------
# 检索结果（TREC run 文件）
# ---------------------------------------------------------------------------


def write_trec_run(path, results, run_name="contriever"):
    """
    把检索结果写成 TREC run 格式，每行 "qid Q0 doc_id rank score run_name"。

    results: {qid: [(doc_id, score), ...]}，已按分数从高到低排序。
    官方 run_evalution.py 的 --retrieval_results_file 也接受这个格式。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, ranked in results.items():
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")


def read_trec_run(path):
    """读取 TREC run 文件，返回 {qid: [doc_id, ...]}（按 rank 升序）。"""
    ranked = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            qid, _q0, doc_id, rank, _score, _name = line.split()
            ranked[qid].append((int(rank), doc_id))
    return {qid: [doc_id for _, doc_id in sorted(pairs)] for qid, pairs in ranked.items()}
