# TRQA Single-Retrieval RAG Baseline Reproduction

[English](#english) | [中文](#中文)

---

## English

Reproduces the **Contriever dense retrieval + LLM reader** (single retrieval) baseline of
[Total Recall QA](https://github.com/mahta-r/total-recall-qa) on the **wiki2 test set**,
using the partial Wikipedia corpus.

TRQA questions require finding *all* relevant entities and then aggregating over them, e.g.:

> What is the average year of birth of the General Secretaries of the CPSU who served before Leonid Brezhnev?  → 1886

Retrieval is therefore evaluated with **entity-level recall**: how many of the relevant entities are retrieved.

### Repository layout

```
.
├── external/
│   └── total-recall-qa/   Official repo (git submodule): dataset, index builder, official answer metric
├── src/trqa/
│   ├── config.py          Paths and default hyperparameters (the only file to edit when moving machines)
│   ├── data.py            Read/write queries, qrels, corpus and TREC run files; the two lookup tables
│   ├── evaluation.py      Entity Recall@k; loads the official answer metric
│   ├── retrieval/
│   │   └── contriever.py  ContrieverEncoder (mean pooling + L2 norm), DenseRetriever (FAISS search)
│   ├── prompts.py         Prompt templates and the `Doc i (Title: ...)` passage format
│   ├── llm.py             LLM loading and generation; <answer>/<tag> parsing
│   └── methods/           One module per QA method
│       └── single_retrieval.py   Single-retrieval baseline (Phase 1)
├── scripts/
│   ├── 01_prepare_corpus.py      Download and convert the corpus (optionally write a small sample)
│   ├── 02_build_index.py         Build the Contriever index with the official index_builder
│   ├── 03_retrieve.py            Retrieve top-K and write a TREC run file
│   ├── 04_evaluate_retrieval.py  Compute entity-level R@k
│   └── 05_generate_answers.py    Generate answers with RAG and score them
├── results/               Experiment outputs: runs/, eval/, generation/
├── data/                  Corpus and index (not tracked by git, see below)
└── pyproject.toml
```

`src/` holds reusable code and is never run directly; `scripts/` are the entry points, one per pipeline step.
Data, retrieval, evaluation, prompts and the LLM are shared by all methods; a new method
(e.g. ReAct, or our own in Phase 2) is added as one new module in `methods/`.
Each step writes its output to a file and the next step reads it, so expensive steps
(index building, retrieval) do not have to be re-run when changing a later step.

### Installation

```bash
git clone --recurse-submodules <repo-url>
# If already cloned: git submodule update --init

pip install -e .
pip install -r external/total-recall-qa/evaluation_requirements.txt
```

`pip install -e .` installs `trqa` in editable mode, so changes in `src/` take effect without reinstalling.

### Data

The corpus (~3.6 GB) and index (~36 GB) are not tracked by git. By default they live under `data/`:

| Content | Path |
|---|---|
| Corpus | `data/wiki_partial_corpus.jsonl` |
| Index | `data/my_indices/contriever_test/contriever_Flat.index` |

If the data is somewhere else, point `TRQA_DATA_DIR` to it instead of copying:

```bash
export TRQA_DATA_DIR="/path/to/TRQA_project"        # bash
$env:TRQA_DATA_DIR = "D:\path\to\TRQA_project"       # PowerShell
set TRQA_DATA_DIR=D:\path\to\TRQA_project            # cmd
```

### Usage

```bash
python scripts/01_prepare_corpus.py                 # skipped if the corpus already exists
python scripts/02_build_index.py                    # slowest step, run once
python scripts/03_retrieve.py                       # -> results/runs/contriever_wiki2_test.trec
python scripts/04_evaluate_retrieval.py             # -> results/eval/
python scripts/05_generate_answers.py --limit 5     # -> results/generation/, try a few queries first
```

Every script supports `-h`; defaults come from `src/trqa/config.py`.

### Development

```bash
pip install -e ".[dev]"     # pytest + ruff

pytest                      # unit tests in tests/ (no corpus, index or GPU needed, ~10 s)
ruff format .               # auto-format (black-compatible style)
ruff check . --fix          # lint and fix what can be fixed automatically
```

Run all three before every commit. Configuration is in `pyproject.toml`; `external/` is excluded from ruff.

### Pipeline

```
 HF corpus ──01──> wiki_partial_corpus.jsonl ──02──> FAISS index (5.91M vectors, 18 GB)
                                                       │
 queries ──03── Contriever encoding ──> inner-product top-1000 ──passage_ids──> doc_id ranking (TREC run)
                                                                                  │
                                     04 entity-level R@k (vs. qrels) ◄────────────┤
                                     05 top-50 texts ─id_to_text─> LLM ─> <answer> ─> scoring
```

Two lookup tables:
- `passage_ids` (index position → doc_id): FAISS only returns vector positions, which must be mapped to doc_ids to compare with qrels.
- `id_to_text` (doc_id → text): the LLM reads text, not vectors, so the passages are fetched by doc_id.

### Current results

Retrieval (Contriever, wiki2 test, 1258 queries), entity-level recall (%):

| | R@3 | R@10 | R@100 | R@1000 |
|---|---|---|---|---|
| Paper, Table 4 (full corpus) | 0.80 | 2.32 | 8.65 | 26.82 |
| Ours (partial corpus) | 0.93 | 2.18 | 8.40 | 26.31 |

Generation: not run yet at scale. Qwen2.5-1.5B-Instruct with top-50 passages was only tried on the first query.

### Alignment with the official implementation

- Retriever `facebook/contriever-msmarco`, mean pooling, L2 normalisation: same as the official `index_builder.py`.
- Entity Recall@k: same as the official `extract_entity_id` / `entity_recall_at_k`.
- Prompt: system prompt, `<information>` template and `Doc i (Title: ...)` passage format match the official `SingleRetrieval`.
- Answer scoring: loads the official `soft_exact_match` directly; reports exact match and soft match with 5% tolerance.
- `--answer_fallback`: if the model gives no `<answer>` tag, the last line is used as the answer. This is **not** the official setting and must be stated when reporting results.
- The corpus is **wiki_partial**, so numbers are not directly comparable with the paper's full-corpus results.

---

## 中文

复现 [Total Recall QA](https://github.com/mahta-r/total-recall-qa) 在 **wiki2 测试集**上的
**Contriever 稠密检索 + LLM 阅读器**（single retrieval）基线，使用 partial 维基语料。

TRQA 的问题需要"找全"所有相关实体后再做聚合计算，例如：

> What is the average year of birth of the General Secretaries of the CPSU who served before Leonid Brezhnev?  → 1886

因此检索的评估重点是**实体级召回**：能把多少相关实体找回来。

### 目录结构

```
.
├── external/
│   └── total-recall-qa/   官方仓库（git submodule）：数据集、建索引脚本、官方判分函数
├── src/trqa/
│   ├── config.py          路径与默认超参数（换机器只改这里）
│   ├── data.py            queries / qrels / 语料 / TREC run 读写，两张映射表
│   ├── evaluation.py      Entity Recall@k；加载官方答案判分函数
│   ├── retrieval/
│   │   └── contriever.py  ContrieverEncoder（mean pooling + L2 归一化）、DenseRetriever（FAISS 检索）
│   ├── prompts.py         prompt 模板与 `Doc i (Title: ...)` 段落格式
│   ├── llm.py             LLM 加载与生成；<answer> 等标签的解析
│   └── methods/           回答问题的方法，每种方法一个模块
│       └── single_retrieval.py   单次检索基线（Phase 1）
├── scripts/
│   ├── 01_prepare_corpus.py      下载并转换语料（可选切小样本）
│   ├── 02_build_index.py         调用官方 index_builder 建 Contriever 索引
│   ├── 03_retrieve.py            检索 Top-K，写 TREC run 文件
│   ├── 04_evaluate_retrieval.py  计算实体级 R@k
│   └── 05_generate_answers.py    RAG 生成答案并判分
├── results/               实验输出：runs/、eval/、generation/
├── data/                  语料与索引（不进 git，见下文）
└── pyproject.toml
```

`src/` 放可复用的代码，本身不直接运行；`scripts/` 是运行入口，每个脚本对应流程中的一步。
数据、检索、评估、prompt 和 LLM 是所有方法共用的；新增一种方法（比如 ReAct，或 Phase 2 我们自己的方法）
只需要在 `methods/` 里加一个模块。
每一步把结果写进文件，下一步读取文件，所以改后面的步骤时不必重跑建索引、检索这些耗时的步骤。

### 安装

```bash
git clone --recurse-submodules <本仓库地址>
# 已经 clone 过的话：git submodule update --init

pip install -e .
pip install -r external/total-recall-qa/evaluation_requirements.txt
```

`pip install -e .` 以可编辑模式安装 `trqa`，修改 `src/` 后立即生效，不用重新安装。

### 数据

语料（约 3.6 GB）和索引（约 36 GB）不进 git，默认放在 `data/` 下：

| 内容 | 路径 |
|---|---|
| 语料 | `data/wiki_partial_corpus.jsonl` |
| 索引 | `data/my_indices/contriever_test/contriever_Flat.index` |

如果数据已经在别的位置，设置环境变量 `TRQA_DATA_DIR` 指过去即可，不用拷贝：

```bash
export TRQA_DATA_DIR="/path/to/TRQA_project"        # bash
$env:TRQA_DATA_DIR = "D:\path\to\TRQA_project"       # PowerShell
set TRQA_DATA_DIR=D:\path\to\TRQA_project            # cmd
```

### 运行

```bash
python scripts/01_prepare_corpus.py                 # 已有语料会自动跳过
python scripts/02_build_index.py                    # 耗时最长，只需一次
python scripts/03_retrieve.py                       # -> results/runs/contriever_wiki2_test.trec
python scripts/04_evaluate_retrieval.py             # -> results/eval/
python scripts/05_generate_answers.py --limit 5     # -> results/generation/，先小规模试跑
```

所有脚本都支持 `-h` 查看参数，默认值来自 `src/trqa/config.py`。

### 开发

```bash
pip install -e ".[dev]"     # 安装 pytest 和 ruff

pytest                      # 运行 tests/ 里的单元测试（不需要语料、索引或 GPU，约 10 秒）
ruff format .               # 自动格式化（与 black 风格兼容）
ruff check . --fix          # 代码检查，并自动修复能修的问题
```

每次提交前把这三条都跑一遍。配置在 `pyproject.toml` 里，`external/` 不在 ruff 的检查范围内。

### 流程

```
 HF 语料 ──01──> wiki_partial_corpus.jsonl ──02──> FAISS 索引 (591 万向量, 18 GB)
                                                    │
 queries ──03── Contriever 编码 ──> 内积搜索 Top-1000 ──passage_ids──> doc_id 排名 (TREC run)
                                                                          │
                                       04 实体级 R@k (对比 qrels) ◄────────┤
                                       05 Top-50 原文 ─id_to_text─> LLM ─> <answer> ─> 判分
```

两张映射表：
- `passage_ids`（索引位置 → doc_id）：FAISS 只返回"第几个向量"，需要翻译成 doc_id 才能和 qrels 对比。
- `id_to_text`（doc_id → 原文）：LLM 读的是文字而不是向量，要用 doc_id 取回原文。

### 当前结果

检索（Contriever，wiki2 test，1258 题），实体级召回（%）：

| | R@3 | R@10 | R@100 | R@1000 |
|---|---|---|---|---|
| 论文 Table 4（完整语料） | 0.80 | 2.32 | 8.65 | 26.82 |
| 本仓库（partial 语料） | 0.93 | 2.18 | 8.40 | 26.31 |

生成：尚未大规模运行。Qwen2.5-1.5B-Instruct + Top-50 段落目前只在第 1 题上试过。

### 与官方实现的对齐说明

- 检索模型 `facebook/contriever-msmarco`、mean pooling、L2 归一化：与官方 `index_builder.py` 一致。
- Entity Recall@k：与官方 `extract_entity_id` / `entity_recall_at_k` 一致。
- prompt：system prompt、`<information>` 模板、`Doc i (Title: ...)` 段落格式都与官方 `SingleRetrieval` 一致。
- 答案判分：直接加载官方 `soft_exact_match`，同时输出 exact match 和 5% 容差的 soft match。
- `--answer_fallback`：模型没有输出 `<answer>` 标签时，取最后一行作为答案。这**不是**官方设定，报告结果时需注明。
- 语料用的是 **wiki_partial**（部分语料），与论文在完整语料上的数字不能直接比较。
