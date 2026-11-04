"""
检索器，每种检索模型一个模块：

    contriever   Contriever 编码 + FAISS 稠密检索
"""

from .contriever import ContrieverEncoder, DenseRetriever, mean_pooling

__all__ = ["ContrieverEncoder", "DenseRetriever", "mean_pooling"]
