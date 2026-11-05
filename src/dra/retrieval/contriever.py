"""
Dense retrieval: Contriever encoding + FAISS similarity search.
稠密检索：Contriever 编码 + FAISS 相似度搜索。

Flow / 流程：
    question text --(ContrieverEncoder)--> question vector
                  --(FAISS inner-product search)--> index positions
                  --(passage_ids lookup)--> ranked passage ids
    问题文字 --(ContrieverEncoder)--> 问题向量 --(FAISS 内积搜索)--> 索引位置 --(查 passage_ids)--> 段落 id 排名

A small step-by-step demo with 1 question and 4 passages: Project/demo_dense_retrieval.py (outside the repo).
1 个问题 + 4 个段落的逐步演示：Project/demo_dense_retrieval.py（在仓库外）。
"""

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def mean_pooling(last_hidden_state, attention_mask):
    """
    Average the token vectors of each text into one vector (Contriever's pooling).
    把每段文字所有 token 的向量取平均，得到一个向量（Contriever 的池化方式）。

    attention_mask is 1 for real tokens and 0 for padding, so padding is left out of the average.
    Without it, a short question padded to the length of a long one in the same batch would get a
    different vector (in the demo, similarity to the right passage dropped from 0.81 to 0.25).
    attention_mask 真实 token 为 1、补齐的空位为 0，所以补齐的位置不参与平均。
    不用它的话，同一批里短问题被补齐到长问题的长度，向量就会变
    （演示中，与正确段落的相似度从 0.81 降到 0.25）。

    last_hidden_state: (batch, tokens, 768)    attention_mask: (batch, tokens)    返回 / returns: (batch, 768)
    """
    # unsqueeze(-1) ADDS a last dimension of size 1: (batch, tokens) -> (batch, tokens, 1).
    # When multiplied with (batch, tokens, 768), that 1 is broadcast (copied) to all 768 values,
    # so every value of a padding token becomes 0. .float() turns the 0/1 integers into 0.0/1.0.
    # unsqueeze(-1) 在最后【增加】一个大小为 1 的维度：(batch, tokens) -> (batch, tokens, 1)。
    # 和 (batch, tokens, 768) 相乘时，这个 1 会被广播（复制）到全部 768 个数，补齐位置的值都变成 0。
    # .float() 把整数 0/1 变成小数 0.0/1.0。
    mask = attention_mask.unsqueeze(-1).float()
    # Sum over the tokens (dim 1), then divide by the number of real tokens.
    # 在 token 维度（第 1 维）上求和，再除以真实 token 的个数。
    return (last_hidden_state * mask).sum(1) / mask.sum(1)


class ContrieverEncoder:
    """
    Contriever text encoder: text -> 768-dim vector.
    Contriever 文本编码器：文字 -> 768 维向量。

    Questions must be encoded exactly like the passages were when the index was built (same model,
    same pooling, same L2 normalisation); otherwise the two kinds of vectors are not comparable.
    After normalisation every vector has length 1, so the inner product used by the FAISS Flat index
    equals the cosine similarity.
    问题必须和建索引时的段落用完全相同的方式编码（同一模型、同样的池化、同样的 L2 归一化），
    否则两边的向量没法比较。归一化后每个向量长度为 1，FAISS Flat 索引的内积就等于余弦相似度。
    """

    def __init__(self, model_name, device=None, max_length=512):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()  # inference only, no gradients: faster and less memory / 只推理不训练：更快、更省显存
    def encode(self, texts, batch_size=32):
        """
        Encode a list of texts into an (N, 768) float32 matrix; every row is L2-normalised.
        把文字列表编码成 (N, 768) 的 float32 矩阵，每行已 L2 归一化。
        """
        all_embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="encode"):
            batch = texts[i : i + batch_size]
            # a. Tokenise: text -> token ids. padding=True pads shorter texts so the batch forms one matrix,
            #    and also returns attention_mask (1 = real token, 0 = padding).
            #    分词：文字 -> token id。padding=True 把短的补齐成一个矩阵，同时返回 attention_mask。
            inputs = self.tokenizer(
                batch,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            # b. Model: one 768-dim vector per token; attention_mask tells it to ignore padding.
            #    过模型：每个 token 一个 768 维向量；attention_mask 让模型忽略补齐的空位。
            outputs = self.model(**inputs)
            # c. Mean pooling: one vector per text.   平均池化：每段文字一个向量。
            emb = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            # d. Normalise to length 1.               归一化：长度缩放为 1。
            emb = torch.nn.functional.normalize(emb, dim=-1)
            all_embs.append(emb.cpu().numpy())
        # Batches are appended in order, so row i of the result belongs to texts[i].
        # 各批按顺序追加，所以结果第 i 行对应 texts[i]。
        return np.concatenate(all_embs, axis=0).astype("float32")


class DenseRetriever:
    """
    FAISS index + position -> passage id table, searchable with plain text.
    FAISS 索引 + "位置 -> 段落 id"映射表，可以直接用文字检索。
    """

    def __init__(self, index_path, passage_ids, encoder):
        import faiss

        self.index = faiss.read_index(str(index_path))  # reads the whole index into memory (18 GB) / 整个读进内存
        self.passage_ids = passage_ids
        self.encoder = encoder
        # The table must match the index one-to-one, otherwise every result is mapped to the wrong passage.
        # Note: this only checks the count; a different line order would not be detected here.
        # 映射表必须与索引一一对应，否则每个结果都会翻译错。注意：这里只能检查数量，检查不出顺序错乱。
        assert len(passage_ids) == self.index.ntotal, (
            f"passage_ids ({len(passage_ids)}) does not match index size ({self.index.ntotal})"
        )

    def search(self, queries, k=1000, batch_size=32):
        """
        Return the k most similar passages for every query.
        检索每个问题最相似的 k 个段落。

        Returns a list as long as queries; element i is [(passage_id, score), ...] for queries[i],
        sorted by score, highest first.
        返回与 queries 等长的列表，第 i 个元素是 queries[i] 的 [(段落 id, 分数), ...]，按分数降序。
        """
        # ① Encode: (n_queries, 768)                                     编码
        query_embs = self.encoder.encode(queries, batch_size=batch_size)
        # ② Search: inner product with every passage vector, keep the top k.
        #    Both results are (n_queries, k): positions in the index and their scores.
        #    搜索：与每个段落向量算内积，保留前 k 个。两个结果都是 (问题数, k)：索引位置和对应分数。
        scores, indices = self.index.search(query_embs, k)
        # ③ Map positions to passage ids. FAISS pads with -1 when it has fewer than k results; skip those.
        #    把位置翻译成段落 id。结果不足 k 个时 FAISS 用 -1 补位，跳过。
        return [
            [(self.passage_ids[idx], float(s)) for idx, s in zip(row_idx, row_scores, strict=True) if idx >= 0]
            for row_idx, row_scores in zip(indices, scores, strict=True)
        ]
