"""
Code for our TRQA (Total Recall QA) experiments: reproducing the baselines (Phase 1) and our own method (Phase 2).
TRQA (Total Recall QA) 实验代码：复现基线（Phase 1）与我们自己的方法（Phase 2）。

Modules / 模块划分：
    config      paths and default hyperparameters
                路径与默认超参数
    data        reading/writing queries, qrels, corpus and TREC run files; the two lookup tables
                queries / qrels / 语料 / TREC run 文件的读写，以及两张映射表
    evaluation  Entity Recall@k; answer scoring (reuses the official implementation)
                实体级召回 Entity Recall@k；答案判分（复用官方实现）
    retrieval/  retrievers (Contriever + FAISS)
                检索器（Contriever + FAISS）
    prompts     prompt templates and the passage format
                prompt 模板与段落拼接格式
    llm         loading the LLM, generating, parsing its output
                LLM 加载、生成、输出解析
    methods/    ways to answer a question (single-retrieval baseline, ...)
                回答问题的方法（单次检索基线等）
    utils       random seed
                随机种子
    logging_utils  logging to the terminal and a log file
                   日志：同时输出到终端和日志文件
"""
