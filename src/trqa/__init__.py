"""
TRQA (Total Recall QA) 实验代码：复现基线（Phase 1）与我们自己的方法（Phase 2）。

模块划分：
    config      路径与默认超参数
    data        queries / qrels / 语料 / TREC run 文件的读写，以及两张映射表
    evaluation  实体级召回 Entity Recall@k；答案判分（复用官方实现）
    retrieval/  检索器（Contriever + FAISS）
    prompts     prompt 模板与段落拼接格式
    llm         LLM 加载、生成、输出解析
    methods/    回答问题的方法（单次检索基线等）
"""
