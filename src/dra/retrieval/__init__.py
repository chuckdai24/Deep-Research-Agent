"""
Retrievers, one module per retrieval model.
检索器，每种检索模型一个模块。

    contriever   Contriever encoding + FAISS dense retrieval
                 Contriever 编码 + FAISS 稠密检索

The import below lets other code write "from dra.retrieval import DenseRetriever"
instead of "from dra.retrieval.contriever import DenseRetriever".
下面的 import 让其他代码可以直接写 "from dra.retrieval import DenseRetriever"，
不用写完整的 "from dra.retrieval.contriever import DenseRetriever"。
"""

from .contriever import ContrieverEncoder, DenseRetriever, mean_pooling

# Names exported by "from dra.retrieval import *" / "from dra.retrieval import *" 会导出的名字
__all__ = ["ContrieverEncoder", "DenseRetriever", "mean_pooling"]
