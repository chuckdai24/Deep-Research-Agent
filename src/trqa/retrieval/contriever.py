"""
稠密检索：Contriever 编码 + FAISS 相似度搜索。

流程：
    query 文本 --(ContrieverEncoder)--> query 向量
               --(FAISS 索引内积搜索)--> 索引位置
               --(passage_ids 映射)-----> doc_id 排名列表
"""

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def mean_pooling(last_hidden_state, attention_mask):
    """
    对每个 token 的隐藏向量取平均，得到整句向量（Contriever 的池化方式）。
    用 attention_mask 屏蔽 padding，只对真实 token 求平均。
    """
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1)


class ContrieverEncoder:
    """
    Contriever 文本编码器。

    关键约束：query 必须和建索引时的文档使用同一模型、同一池化方式、
    同样做 L2 归一化，否则两边向量不在同一空间，相似度没有意义。
    归一化之后，FAISS Flat 索引的内积就等于余弦相似度。
    """

    def __init__(self, model_name, device=None, max_length=512):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def encode(self, texts, batch_size=32):
        """把文本列表编码成 (N, 768) 的 float32 矩阵，每行已 L2 归一化。"""
        all_embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="encode"):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)
            emb = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            emb = torch.nn.functional.normalize(emb, dim=-1)
            all_embs.append(emb.cpu().numpy())
        return np.concatenate(all_embs, axis=0).astype("float32")


class DenseRetriever:
    """
    FAISS 索引 + "索引位置 -> doc_id" 映射表，对外提供按文本检索的接口。
    """

    def __init__(self, index_path, passage_ids, encoder):
        import faiss

        self.index = faiss.read_index(str(index_path))
        self.passage_ids = passage_ids
        self.encoder = encoder
        # 映射表必须与索引一一对应，否则检索结果会整体错位
        assert len(passage_ids) == self.index.ntotal, (
            f"passage_ids ({len(passage_ids)}) 与索引向量数 ({self.index.ntotal}) 不一致"
        )

    def search(self, queries, k=1000, batch_size=32):
        """
        检索每个 query 最相似的 k 个段落。

        返回与 queries 等长的列表，每个元素是 [(doc_id, score), ...]，按分数降序。
        """
        query_embs = self.encoder.encode(queries, batch_size=batch_size)
        scores, indices = self.index.search(query_embs, k)
        return [
            [(self.passage_ids[idx], float(s)) for idx, s in zip(row_idx, row_scores, strict=True) if idx >= 0]
            for row_idx, row_scores in zip(indices, scores, strict=True)
        ]
