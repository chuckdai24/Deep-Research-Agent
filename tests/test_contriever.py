"""
Unit tests for retrieval/contriever.py: pooling and FAISS search logic
(tiny index and fake encoder; Contriever itself is not loaded).
retrieval/contriever.py 的单元测试：池化与 FAISS 检索逻辑（用小索引和假编码器，不加载 Contriever）。
"""

import json

import faiss
import numpy as np
import pytest
import torch

from dra.data import CorpusReader
from dra.retrieval import DenseRetriever, StepRetriever, mean_pooling


def test_mean_pooling_ignores_padding():
    # 3 tokens; the 3rd is padding (mask 0), so it must not affect the average.
    # 3 个 token，第 3 个是补齐的空位（mask 为 0），不能影响平均值。
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(mean_pooling(hidden, mask), torch.tensor([[2.0, 3.0]]))


class FakeEncoder:
    """
    Returns a fixed vector per query text, standing in for ContrieverEncoder.
    按查询文本返回固定向量，代替 ContrieverEncoder。
    """

    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts, batch_size=32):
        return np.array([self.vectors[t] for t in texts], dtype="float32")


@pytest.fixture
def index_path(tmp_path):
    # 3 passage vectors, at index positions 0, 1, 2 / 3 个段落向量，在索引里的位置依次是 0、1、2
    doc_vectors = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype="float32")
    index = faiss.IndexFlatIP(2)
    index.add(doc_vectors)
    path = tmp_path / "test.index"
    faiss.write_index(index, str(path))
    return path


def test_search_maps_positions_to_doc_ids(index_path):
    encoder = FakeEncoder({"q": [0, 1]})
    retriever = DenseRetriever(index_path, ["a-1", "b-1", "c-1"], encoder)

    [ranked] = retriever.search(["q"], k=2)

    assert [doc_id for doc_id, _ in ranked] == ["b-1", "c-1"]
    assert [score for _, score in ranked] == pytest.approx([1.0, 0.8])


def test_search_drops_missing_results_when_k_exceeds_index(index_path):
    encoder = FakeEncoder({"q": [1, 0]})
    retriever = DenseRetriever(index_path, ["a-1", "b-1", "c-1"], encoder)

    [ranked] = retriever.search(["q"], k=5)  # only 3 vectors, FAISS pads with -1 / 索引只有 3 个向量，FAISS 用 -1 补位
    assert len(ranked) == 3


def test_passage_ids_must_match_index_size(index_path):
    with pytest.raises(AssertionError):
        DenseRetriever(index_path, ["a-1", "b-1"], FakeEncoder({}))


def test_search_positions_returns_index_positions(index_path):
    encoder = FakeEncoder({"q": [0, 1]})
    retriever = DenseRetriever(index_path, ["a-1", "b-1", "c-1"], encoder)

    [hits] = retriever.search_positions(["q"], k=2)

    assert [pos for pos, _ in hits] == [1, 2]


def write_corpus(path, ids):
    with open(path, "w", encoding="utf-8") as f:
        for doc_id in ids:
            f.write(json.dumps({"id": doc_id, "title": doc_id.upper(), "contents": f"text of {doc_id}"}) + "\n")


def test_step_retriever_returns_top_docs_and_full_ranking(index_path, tmp_path):
    write_corpus(tmp_path / "corpus.jsonl", ["a-1", "b-1", "c-1"])
    dense = DenseRetriever(index_path, ["a-1", "b-1", "c-1"], FakeEncoder({"q": [0, 1]}))
    retriever = StepRetriever(dense, CorpusReader(tmp_path / "corpus.jsonl"), topk=1, eval_k=3)

    docs, ranked = retriever.search_for_step("q")

    # Only the top 1 passage with its text, but the ranking of all 3.
    # 只有第 1 名带原文，但排序包含全部 3 个。
    assert docs == [{"id": "b-1", "title": "B-1", "contents": "text of b-1"}]
    assert [doc_id for doc_id, _ in ranked] == ["b-1", "c-1", "a-1"]


def test_step_retriever_detects_wrong_corpus_order(index_path, tmp_path):
    # Same passages, different line order: the line read is not the passage the index points to.
    # 同样的段落、不同的行顺序：读出的那一行不是索引指向的段落。
    write_corpus(tmp_path / "corpus.jsonl", ["c-1", "b-1", "a-1"])
    dense = DenseRetriever(index_path, ["a-1", "b-1", "c-1"], FakeEncoder({"q": [1, 0]}))
    retriever = StepRetriever(dense, CorpusReader(tmp_path / "corpus.jsonl"), topk=1, eval_k=3)

    with pytest.raises(ValueError):
        retriever.search_for_step("q")


def test_step_retriever_corpus_size_must_match_index(index_path, tmp_path):
    write_corpus(tmp_path / "corpus.jsonl", ["a-1", "b-1"])
    dense = DenseRetriever(index_path, ["a-1", "b-1", "c-1"], FakeEncoder({}))
    with pytest.raises(AssertionError):
        StepRetriever(dense, CorpusReader(tmp_path / "corpus.jsonl"))
