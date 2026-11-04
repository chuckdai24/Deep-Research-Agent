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
    """逐行读取语料 jsonl，每次产出一个 {"id", "contents"} 字典。"""
    with open(corpus_path, encoding="utf-8") as f:
        for line in tqdm(f, desc=desc, unit=" docs"):
            yield json.loads(line)


def load_passage_ids(corpus_path, cache_path=None):
    """
    映射表 1：索引位置 -> doc_id（返回列表，下标即位置）。

    为什么需要：FAISS 只存向量，检索返回的是"第几个向量"，必须翻译成 doc_id
    才能和 qrels 对比。index_builder 是按语料文件的行顺序依次把向量加入索引的，
    所以按同样顺序读语料，列表下标 i 就对应索引里第 i 个向量。

    若给了 cache_path：存在则直接读取；不存在则扫描语料后写入，下次就快了。
    """
    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path, encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]

    passage_ids = [doc["id"] for doc in iter_corpus(corpus_path, desc="build passage_ids")]

    if cache_path is not None:
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write("\n".join(passage_ids) + "\n")
    return passage_ids


def load_passage_texts(corpus_path, wanted_ids=None):
    """
    映射表 2：doc_id -> 原文 contents。

    为什么需要：生成阶段 LLM 读的是文字而不是向量，要用 doc_id 取回原文。
    wanted_ids 不为空时只保留这些 id 的原文（生成只需要 Top-k 段落），
    避免把 591 万段落全部放进内存。
    """
    wanted = set(wanted_ids) if wanted_ids is not None else None
    texts = {}
    for doc in iter_corpus(corpus_path, desc="load passage texts"):
        if wanted is None or doc["id"] in wanted:
            texts[doc["id"]] = doc["contents"]
            if wanted is not None and len(texts) == len(wanted):
                break  # 需要的都找到了，提前结束扫描
    return texts


def split_title(contents):
    """把 "title\\ncontents" 拆回 (title, text)。"""
    title, _, text = contents.partition("\n")
    return title, text


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
