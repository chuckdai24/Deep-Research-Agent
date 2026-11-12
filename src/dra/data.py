"""
Reading and writing data.
数据读写。

Four kinds of data / 四类数据：
  1. queries   test questions with gold answers (jsonl)
               测试问题与标准答案（jsonl）
  2. qrels     relevant passages per question (TREC qrels format)
               每个问题的相关段落标注（TREC qrels 格式）
  3. corpus    Wikipedia passages (jsonl), plus the two lookup tables
               "index position -> passage id" and "passage id -> passage record",
               and CorpusReader, which reads single lines by line number (for retrieval during ReAct)
               维基段落语料（jsonl），以及"索引位置 -> 段落 id"、"段落 id -> 段落记录"两张映射表，
               和按行号读取单行的 CorpusReader（ReAct 运行中检索时用）
  4. run file  retrieval results (TREC run format), which decouples retrieval from evaluation / generation
               检索结果（TREC run 格式），把检索和后续的评估 / 生成分开
"""

import json
from array import array
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# queries / qrels
# ---------------------------------------------------------------------------


def load_queries(path):
    """
    Read the queries jsonl. Each line looks like:
    读取 queries jsonl，每行形如：
        {"id": "13_p569", "question": "...", "answer": 1886, "src": "qald10"}
    Returns a list in file order; empty lines are skipped.
    返回保持文件顺序的列表；空行跳过。
    """
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_qrels(path):
    """
    Read TREC qrels; each line is "query_id iteration passage_id relevance".
    读取 TREC qrels，每行为 "query_id iteration passage_id relevance"。

    Returns {query_id: set of relevant passage ids}, keeping only lines with relevance > 0.
    A set, because evaluation only asks whether a passage is relevant, not in which order.
    返回 {query_id: 相关段落 id 的集合}，只保留 relevance > 0 的行。
    用集合，是因为评估只关心"是否相关"，不关心顺序。
    """
    # defaultdict(set): a dict that creates an empty set the first time a new key is used.
    # defaultdict(set)：第一次用到某个新键时，自动为它创建一个空集合。
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
# Corpus / 语料
# ---------------------------------------------------------------------------


def export_corpus(output_path, split="wiki_partial"):
    """
    Download the corpus from HuggingFace (mahtaa/trqa) and save it as jsonl, keeping the original format.
    从 HuggingFace（mahtaa/trqa）下载语料并存成 jsonl，保持原始格式不变。

    Output, one record per line, exactly as published by the authors:
        {"id": ..., "title": ..., "section": [...], "contents": ...}
    输出每行一条记录，与作者发布的原始格式完全相同（字段见上）。

    Why the original format: with a separate "title" field, the official index_builder.py encodes each
    passage as '"title"\\ntext' (lines 489-492), like the authors' own index; and our prompt code
    (prompts.format_passages) uses the "title" field exactly like the official passages2string.
    Earlier versions merged the title into "contents" ("title\\ncontents"); both formats work with
    our code, but only the original one encodes passages the same way as the official index builder.
    为什么保持原始格式：有单独的 "title" 字段时，官方 index_builder.py 会把段落编码成 '"标题"\\n正文'
    （第 489-492 行），与作者自己建的索引一致；我们的 prompt 代码（prompts.format_passages）也像官方
    passages2string 一样直接使用 "title" 字段。旧版本曾把标题拼进 "contents"（"标题\\n正文"），
    两种格式我们的代码都能处理，但只有原始格式与官方建索引的编码方式一致。

    Returns the number of passages written. / 返回写出的段落数。
    """
    # Imported here rather than at the top: only step 1 needs the datasets library.
    # 在这里而不是文件开头导入：只有步骤 1 需要 datasets 库。
    from datasets import load_dataset

    corpus = load_dataset("mahtaa/trqa", "corpus", split=split)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        # Write each record unchanged, in the original order (index position i = line i).
        # 每条记录原样写出，保持原始顺序（索引位置 i = 第 i 行）。
        for doc in tqdm(corpus, desc="export corpus"):
            f.write(json.dumps(doc) + "\n")
    return len(corpus)


def write_sample(corpus_path, output_path, n):
    """
    Copy the first n lines of the corpus into a small sample file, for quickly testing index building.
    取语料前 n 行另存为小样本，用于快速调试建索引流程。
    """
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


def build_line_offsets(corpus_path):
    """
    Scan the corpus once and return the byte position where every line starts (int64 array; entry i = line i).
    把语料扫描一遍，返回每一行在文件里从第几个字节开始（int64 数组；第 i 个元素 = 第 i 行）。

    The file is read in binary mode, so the positions are exact byte counts that f.seek() can jump to.
    用二进制方式读文件，所以得到的是准确的字节数，可以直接交给 f.seek() 跳过去。
    """
    # array("q") = a compact list of 64-bit integers (a Python list of 5.9M ints would take ~5x more memory).
    # array("q") 是紧凑的 64 位整数列表（同样 590 万个数，用 Python 列表要多占约 5 倍内存）。
    offsets = array("q")
    position = 0
    with open(corpus_path, "rb") as f:
        for line in tqdm(f, desc="build line offsets", unit=" lines"):
            offsets.append(position)
            position += len(line)
    return np.frombuffer(offsets, dtype=np.int64).copy()


class CorpusReader:
    """
    Read single corpus lines by line number, without loading the whole corpus into memory.
    按行号读取语料中的某一行，不必把整个语料读进内存。

    Why: ReAct searches with queries the model writes while it runs, so which passages are needed is only
    known at that moment; load_passages would have to scan the 3.8 GB file for every search (~2 min).
    Instead the start position of every line is recorded once (build_line_offsets, ~47 MB for 5.9M lines),
    and a line is read by jumping straight to it (milliseconds).
    为什么需要：ReAct 用模型运行中写出的查询去检索，需要哪些段落要到那一刻才知道；用 load_passages
    就得每次检索都把 3.8 GB 的文件扫一遍（约 2 分钟）。所以先记下每一行的起始位置（build_line_offsets，
    590 万行约 47 MB），之后直接跳到那一行去读（几毫秒）。

    Line number = index position (see load_passage_ids), so a FAISS result can be read directly.
    行号 = 索引位置（见 load_passage_ids），所以 FAISS 返回的位置可以直接拿来读。

    offsets_cache: .npy file; read it if it exists, otherwise scan the corpus and write it for next time.
    offsets_cache：.npy 文件；存在就直接读，不存在就扫描语料后写入，下次直接读。
    """

    def __init__(self, corpus_path, offsets_cache=None):
        self.corpus_path = Path(corpus_path)
        if offsets_cache is not None and Path(offsets_cache).exists():
            self.offsets = np.load(offsets_cache)
        else:
            self.offsets = build_line_offsets(self.corpus_path)
            if offsets_cache is not None:
                Path(offsets_cache).parent.mkdir(parents=True, exist_ok=True)
                np.save(offsets_cache, self.offsets)
        # Kept open for the whole run: opening the file for every read would be slower.
        # 整个运行期间保持打开：每读一次就打开一次文件会更慢。
        self._file = open(self.corpus_path, "rb")  # noqa: SIM115

    def __len__(self):
        return len(self.offsets)

    def get(self, line_number):
        """
        Return line line_number of the corpus as a dict (e.g. {"id", "title", "section", "contents"}).
        返回语料第 line_number 行，解析成字典（例如 {"id", "title", "section", "contents"}）。
        """
        self._file.seek(int(self.offsets[line_number]))
        return json.loads(self._file.readline())

    def close(self):
        self._file.close()


# ---------------------------------------------------------------------------
# Retrieval results (TREC run files) / 检索结果（TREC run 文件）
# ---------------------------------------------------------------------------


def write_trec_run(path, results, run_name="contriever"):
    """
    Write retrieval results in TREC run format, one line per (question, passage):
    把检索结果写成 TREC run 格式，每行一个（题目, 段落）：
        "qid Q0 passage_id rank score run_name"
    ("Q0" is a fixed placeholder column of the format / "Q0" 是格式里固定不变的占位列)

    results: {qid: [(passage_id, score), ...]}, each list already sorted by score, highest first.
    results：{qid: [(段落 id, 分数), ...]}，每个列表已按分数从高到低排序。
    The official run_evalution.py (--retrieval_results_file) accepts this format as well.
    官方 run_evalution.py 的 --retrieval_results_file 也接受这个格式。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # create runs/ if missing / 文件夹不存在就创建
    with open(path, "w", encoding="utf-8") as f:
        for qid, ranked in results.items():
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")


def read_trec_run(path):
    """
    Read a TREC run file; returns {qid: [passage_id, ...]} in rank order (rank 1 first).
    Scores are dropped: evaluation and generation only need the order.
    读取 TREC run 文件，返回 {qid: [段落 id, ...]}，按排名从 1 开始排序。
    分数不保留：评估和生成只需要顺序。
    """
    # Collect (rank, passage_id) pairs, then sort by rank, so the result is correct even if
    # the lines in the file are not in rank order.
    # 先收集 (排名, 段落 id)，再按排名排序；即使文件里的行没按排名排列，结果也是对的。
    ranked = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            qid, _q0, doc_id, rank, _score, _name = line.split()
            ranked[qid].append((int(rank), doc_id))
    return {qid: [doc_id for _, doc_id in sorted(pairs)] for qid, pairs in ranked.items()}
