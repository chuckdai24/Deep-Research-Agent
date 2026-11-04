"""retrieval/contriever.py 的单元测试：池化与 FAISS 检索逻辑（用小索引和假编码器，不加载 Contriever）。"""

import faiss
import numpy as np
import pytest
import torch

from dra.retrieval import DenseRetriever, mean_pooling


def test_mean_pooling_ignores_padding():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])  # 第 3 个 token 是 padding
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(mean_pooling(hidden, mask), torch.tensor([[2.0, 3.0]]))


class FakeEncoder:
    """按查询文本返回固定向量，代替 ContrieverEncoder。"""

    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts, batch_size=32):
        return np.array([self.vectors[t] for t in texts], dtype="float32")


@pytest.fixture
def index_path(tmp_path):
    # 3 个文档向量，在索引里的位置依次是 0、1、2
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

    [ranked] = retriever.search(["q"], k=5)  # 索引只有 3 个向量，FAISS 用 -1 补位
    assert len(ranked) == 3


def test_passage_ids_must_match_index_size(index_path):
    with pytest.raises(AssertionError):
        DenseRetriever(index_path, ["a-1", "b-1"], FakeEncoder({}))
