# Group 18 — Deep Research Agents on Total Recall QA

[English](#english) | [中文](#中文)

---

## English

Course project for Information Retrieval (NWI-I00041, Radboud University), based on
[Total Recall QA](https://github.com/mahta-r/total-recall-qa) (TRQA).

| | |
|---|---|
| Dataset | TRQA **Wiki2**, test split (1258 queries) |
| Corpus | Partial Wikipedia corpus (5,910,177 passages) |
| Retriever | Contriever (`facebook/contriever-msmarco`) |
| Reader | Qwen2.5-7B-Instruct, top-50 passages |

### Status

| Step | Description | Status |
|---|---|---|
| 1.1 | Retrieval reproduction (Table 4: R@3/10/100/1000) | Done |
| 1.2 | Single retrieval + reader, and one DRA (Table 8: EM, SM 10/20/50/90%) | Done (single retrieval and ReAct) |
| 2 | Own DRA method | Not started |

### Results

**Step 1.1** — Contriever on Wiki2 test, entity-level recall (%):

| Run | Corpus | Index | R@3 | R@10 | R@100 | R@1000 |
|---|---|---|---|---|---|---|
| Paper, Table 4 | full | — | 0.80 | 2.32 | 8.65 | 26.82 |
| Ours, csedu | partial | official (HF `mahtaa/trqa_experiments`) | 0.93 | 2.18 | 8.39 | 26.30 |
| Ours, local | partial | built with the official `index_builder.py` | 0.93 | 2.18 | 8.39 | 26.31 |

**Step 1.2** — Qwen2.5-7B-Instruct reader on Wiki2 test, partial corpus (%). Single retrieval: one search,
top-50 passages in the prompt. ReAct: the model searches while it reasons, top-3 passages per Search step.

| Method | Run | Retriever | Reader dtype | EM | SM1% | SM5% | SM10% | SM20% | SM50% | SM90% |
|---|---|---|---|---|---|---|---|---|---|---|
| Single retrieval | Paper, Table 8 | E5 | bf16 | 3.42 | – | – | 7.00 | 8.82 | 14.47 | 24.32 |
| Single retrieval | Ours, csedu | Contriever | fp16 | 2.07 | 3.74 | 4.53 | 5.09 | 6.20 | 11.13 | 18.52 |
| ReAct | Paper, Table 8 | E5 | bf16 | 4.53 | – | – | 15.58 | 17.01 | 24.32 | 38.79 |
| ReAct | Ours, csedu | Contriever | fp16 | 4.29 | 9.06 | 12.64 | 13.28 | 14.31 | 19.24 | 29.33 |

Ours, both methods: 1258 queries, 2× RTX 2080 Ti, temperature 0.7, one run each.

- Single retrieval: 581 outputs without an `<answer>` tag (scored as wrong, as in the official code);
  1 output cut off at `max_new_tokens=1024`; 3.3 s per query.
- ReAct: 2.67 searches and 4.59 LLM calls per query on average; 6 queries finished without searching;
  0 queries reached the 20-step limit (forced finish); 23.0 s per query; the full run took 8 h 08 min.

Entity-level recall (%) of the passages retrieved by ReAct. The rankings of all Search steps of a query are
merged by rank (rank 1 of every step, then rank 2, ...; duplicates skipped), as in the official
`_interleave_rank_lists_with_dedup`; queries without a search count as 0. "All" is recall over the whole merged list.

| Run | R@3 | R@10 | R@100 | R@1000 | All |
|---|---|---|---|---|---|
| ReAct, ours, csedu | 1.05 | 2.68 | 11.39 | 33.77 | 39.85 |

### Quick start (csedu cluster)

```bash
# Once, on the login node
git clone --recurse-submodules https://github.com/diegoruedaa/group18-project.git ~/group18-project
cd ~/group18-project
bash cluster/setup.sh                       # venv, dependencies, Contriever, official index (18 GB), Qwen2.5-7B (15 GB)

# Step 1.1 (runs scripts 03 + 04 on a GPU node)
sbatch cluster/step1_1_retrieval.sbatch
squeue -u $USER                             # PD = waiting, R = running
cat results/eval_official/retrieval_summary.json

# Step 1.2 (runs script 05 on 2 GPUs; needs the Step 1.1 run file)
sbatch --time=01:00:00 cluster/step1_2_single_retrieval.sbatch --limit 5   # test run
sbatch cluster/step1_2_single_retrieval.sbatch                             # all queries
cat results/generation/single_retrieval_qwen7b_fp16_summary.json

# Step 1.2 ReAct (script 05 --method react on 2 GPUs; loads the index itself, no run file needed)
sbatch --time=01:00:00 cluster/step1_2_react.sbatch --limit 3 --output results/generation/react_test.jsonl
sbatch cluster/step1_2_react.sbatch                                         # all queries
cat results/generation/react_qwen7b_fp16_summary.json
```

Script 05 resumes like the official code: questions already in the output file are skipped, so a stopped job
can be submitted again with the same command.

`setup.sh` is safe to re-run; finished steps are skipped and an interrupted download resumes.
Optionally add `source ~/group18-project/cluster/env.sh` to `~/.bashrc` to load the environment at login.

### Cluster configuration

All paths are defined in `cluster/env.sh` and derived from `$USER`, so the scripts work unchanged for every
group member. Home is limited to 10 GB; everything large is stored on the course volume.

| Variable | Default | Content |
|---|---|---|
| `MYDATA` | `/vol/csedu-nobackup/course/I00041_informationretrieval/users/$USER` | Personal data folder |
| `HF_HOME` | `$MYDATA/hf_cache` | HuggingFace model cache |
| `PIP_CACHE_DIR` | `$MYDATA/pip_cache` | pip download cache |
| `TRQA_CONTRIEVER_INDEX` | `$MYDATA/trqa_experiments/indices/wiki_partial/contriever_partial/contriever_Flat.index` | Official index (18 GB, fixed HF revision) |
| `TRQA_CORPUS` | `$COURSE/shared/trqa/data/shared/corpus_wikipedia_en-20251001_rewritten_partial.jsonl` | Course copy of the corpus (read-only) |

The venv is `$MYDATA/venv`; the passage id table created by the first run of 03 is `$MYDATA/passage_ids_official.txt`.
To use other locations, export the variable before running `setup.sh` or sourcing `env.sh`
(e.g. `export TRQA_CONTRIEVER_INDEX=/path/to/existing.index` skips the download).

**SLURM limits.** Account `csedui00041`, partition `csedu`. Each QOS caps the resources of a single job:

| QOS | Max per job | Max time |
|---|---|---|
| `csedu-small` | 1 GPU, 6 CPUs, 15 GB | 4 h |
| `csedu-normal` | 2 GPUs, 12 CPUs, 31 GB | 12 h |
| `csedu-large` | — | 2 days |

Step 03 loads the full index into memory (about 21 GB in total), so the Step 1.1 job uses `csedu-normal` with 30 GB.
Jobs run with `HF_HUB_OFFLINE=1`; models must be downloaded on the login node first.

### Running locally

```bash
git clone --recurse-submodules https://github.com/diegoruedaa/group18-project.git
pip install -e ".[dev]"
export TRQA_DATA_DIR=/path/to/data          # folder with the corpus and index, default: ./data

python scripts/01_prepare_corpus.py         # download the corpus (not needed on csedu)
python scripts/02_build_index.py            # build the index, several hours (not needed on csedu)
python scripts/03_retrieve.py               # top-1000 per query -> runs/*.trec
python scripts/04_evaluate_retrieval.py     # entity-level R@k -> results/eval/
python scripts/05_generate_answers.py       # reader + scoring -> results/generation/
python scripts/05_generate_answers.py --method react   # ReAct, searches while running
```

Every script supports `-h`. Defaults are in `src/dra/config.py`; every run writes a log to `results/logs/`.

### Repository layout

```
├── src/dra/                 Library code (installed as package `dra`)
│   ├── config.py            Paths and default hyperparameters
│   ├── data.py              Queries, qrels, corpus, TREC run files
│   ├── retrieval/           Contriever encoder and FAISS retriever
│   ├── evaluation.py        Entity-level Recall@k, official answer metric
│   ├── prompts.py           Prompt templates (copied from the official code)
│   ├── llm.py               Reader loading, generation, answer parsing
│   └── methods/             One module per QA method (single_retrieval.py, react.py)
├── scripts/01–05            One entry point per pipeline step
├── cluster/                 csedu environment, setup and SLURM job scripts
├── tests/                   Unit tests (no corpus, index or GPU needed)
├── external/total-recall-qa Official repository (git submodule, not modified)
├── runs/                    Retrieval run files (intermediate, not tracked by git)
└── results/                 Evaluation, generated answers and logs (tracked by git)
```

Steps exchange data through files in `runs/` and `results/` (retrieval results are TREC run files in `runs/`),
so later steps can be changed and re-run without repeating retrieval.

### Consistency with the official implementation

- Index, query encoding (max length 64, fp16, no prefix), mean pooling and L2 normalisation follow the official code.
- Entity-level Recall@k follows the official `extract_entity_id` / `entity_recall_at_k`.
- System prompt, `<information>` template and passage format match the official `SingleRetrieval`.
- Answers are scored with the official `soft_exact_match`: EM and SM at 1/5/10/20/50/90%, `decimals=3`, as in `run_evalution.py`.
- The reader runs in fp16 on csedu (`--dtype float16`, split over 2 GPUs); the official setting is bf16, which the RTX 2080 Ti does not support.
- As in the official `get_answer`, only the `<answer>` tag is parsed; an output without it has no answer and is scored as wrong.
- ReAct follows the official `ReAct_Model`: same instruction and 6 examples, stop strings, action parsing, 20-step limit,
  forced finish, top-3 passages per step out of 1000 retrieved; a unit test runs the official class and ours on the same
  model outputs and compares every call.

### Development

```bash
pytest                  # unit tests, ~10 s
ruff format .           # format
ruff check . --fix      # lint
```

Run all three before committing. Settings are in `pyproject.toml`.

---

## 中文

信息检索课程（NWI-I00041，Radboud University）项目，基于
[Total Recall QA](https://github.com/mahta-r/total-recall-qa)（TRQA）。

| | |
|---|---|
| 数据集 | TRQA **Wiki2** 测试集（1258 题） |
| 语料 | partial 维基语料（5,910,177 个段落） |
| 检索模型 | Contriever（`facebook/contriever-msmarco`） |
| 阅读器 | Qwen2.5-7B-Instruct，Top-50 段落 |

### 进度

| 步骤 | 内容 | 状态 |
|---|---|---|
| 1.1 | 检索复现（Table 4：R@3/10/100/1000） | 完成 |
| 1.2 | 单次检索 + 阅读器，以及一种 DRA（Table 8：EM、SM 10/20/50/90%） | 完成（单次检索和 ReAct） |
| 2 | 自己的 DRA 方法 | 未开始 |

### 结果

**Step 1.1**：Contriever，Wiki2 测试集，实体级召回（%）：

| 运行 | 语料 | 索引 | R@3 | R@10 | R@100 | R@1000 |
|---|---|---|---|---|---|---|
| 论文 Table 4 | 完整 | — | 0.80 | 2.32 | 8.65 | 26.82 |
| 本组，csedu | partial | 官方（HF `mahtaa/trqa_experiments`） | 0.93 | 2.18 | 8.39 | 26.30 |
| 本组，本地 | partial | 用官方 `index_builder.py` 自建 | 0.93 | 2.18 | 8.39 | 26.31 |

**Step 1.2**：阅读器 Qwen2.5-7B-Instruct，Wiki2 测试集，partial 语料（%）。单次检索：检索一次，
Top-50 段落放进 prompt。ReAct：模型边推理边检索，每个 Search 步骤给 Top-3 段落。

| 方法 | 运行 | 检索模型 | 阅读器精度 | EM | SM1% | SM5% | SM10% | SM20% | SM50% | SM90% |
|---|---|---|---|---|---|---|---|---|---|---|
| 单次检索 | 论文 Table 8 | E5 | bf16 | 3.42 | – | – | 7.00 | 8.82 | 14.47 | 24.32 |
| 单次检索 | 本组，csedu | Contriever | fp16 | 2.07 | 3.74 | 4.53 | 5.09 | 6.20 | 11.13 | 18.52 |
| ReAct | 论文 Table 8 | E5 | bf16 | 4.53 | – | – | 15.58 | 17.01 | 24.32 | 38.79 |
| ReAct | 本组，csedu | Contriever | fp16 | 4.29 | 9.06 | 12.64 | 13.28 | 14.31 | 19.24 | 29.33 |

本组（两种方法相同）：1258 题，2 块 RTX 2080 Ti，温度 0.7，各运行一次。

- 单次检索：581 题输出中没有 `<answer>` 标签（与官方代码一致，按答错计）；
  1 题达到 `max_new_tokens=1024` 被截断；每题 3.3 秒。
- ReAct：平均每题检索 2.67 次、调用 LLM 4.59 次；6 题没有检索就结束；
  0 题达到 20 步上限（强制作答）；每题 23.0 秒；全量运行用时 8 小时 08 分。

ReAct 检索到的段落的实体级召回（%）。每题所有 Search 步骤的排名按名次交错合并（各步第 1 名，再各步第 2 名……；
重复的跳过），与官方 `_interleave_rank_lists_with_dedup` 一致；没有检索的题按 0 计。"全部"是合并列表全部结果上的召回。

| 运行 | R@3 | R@10 | R@100 | R@1000 | 全部 |
|---|---|---|---|---|---|
| ReAct，本组，csedu | 1.05 | 2.68 | 11.39 | 33.77 | 39.85 |

### 快速开始（csedu 集群）

```bash
# 一次性，在登录节点上
git clone --recurse-submodules https://github.com/diegoruedaa/group18-project.git ~/group18-project
cd ~/group18-project
bash cluster/setup.sh                       # venv、依赖、Contriever、官方索引（18 GB）、Qwen2.5-7B（15 GB）

# Step 1.1（在 GPU 节点上运行脚本 03 + 04）
sbatch cluster/step1_1_retrieval.sbatch
squeue -u $USER                             # PD = 排队中，R = 运行中
cat results/eval_official/retrieval_summary.json

# Step 1.2（在 2 块 GPU 上运行脚本 05；需要 Step 1.1 的 run 文件）
sbatch --time=01:00:00 cluster/step1_2_single_retrieval.sbatch --limit 5   # 先测 5 题
sbatch cluster/step1_2_single_retrieval.sbatch                             # 全部题目
cat results/generation/single_retrieval_qwen7b_fp16_summary.json

# Step 1.2 ReAct（在 2 块 GPU 上运行脚本 05 --method react；自己加载索引，不需要 run 文件）
sbatch --time=01:00:00 cluster/step1_2_react.sbatch --limit 3 --output results/generation/react_test.jsonl
sbatch cluster/step1_2_react.sbatch                                         # 全部题目
cat results/generation/react_qwen7b_fp16_summary.json
```

脚本 05 和官方代码一样支持断点续跑：输出文件里已有的题会被跳过，所以作业停止后用同样的命令再提交一次即可。

`setup.sh` 可以重复运行：已完成的步骤会跳过，中断的下载会从断点继续。
可选：把 `source ~/group18-project/cluster/env.sh` 加进 `~/.bashrc`，登录时自动加载环境。

### 集群配置

所有路径都在 `cluster/env.sh` 里定义，并根据 `$USER` 自动确定，所以每个组员都能直接使用。
home 只有 10 GB，大文件都放在课程存储卷上。

| 变量 | 默认值 | 内容 |
|---|---|---|
| `MYDATA` | `/vol/csedu-nobackup/course/I00041_informationretrieval/users/$USER` | 个人数据文件夹 |
| `HF_HOME` | `$MYDATA/hf_cache` | HuggingFace 模型缓存 |
| `PIP_CACHE_DIR` | `$MYDATA/pip_cache` | pip 下载缓存 |
| `TRQA_CONTRIEVER_INDEX` | `$MYDATA/trqa_experiments/indices/wiki_partial/contriever_partial/contriever_Flat.index` | 官方索引（18 GB，固定 HF 版本） |
| `TRQA_CORPUS` | `$COURSE/shared/trqa/data/shared/corpus_wikipedia_en-20251001_rewritten_partial.jsonl` | 课程提供的语料（只读） |

venv 在 `$MYDATA/venv`；第一次运行 03 时生成的段落 id 映射表在 `$MYDATA/passage_ids_official.txt`。
想用别的位置，在运行 `setup.sh` 或 source `env.sh` 之前 export 对应的变量
（例如 `export TRQA_CONTRIEVER_INDEX=/path/to/existing.index` 会跳过下载）。

**SLURM 资源上限。** 账户 `csedui00041`，分区 `csedu`。每个 QOS 限制单个作业能申请的资源：

| QOS | 每个作业最多 | 最长时间 |
|---|---|---|
| `csedu-small` | 1 块 GPU、6 个 CPU、15 GB | 4 小时 |
| `csedu-normal` | 2 块 GPU、12 个 CPU、31 GB | 12 小时 |
| `csedu-large` | — | 2 天 |

03 会把整个索引读进内存（总共约 21 GB），所以 Step 1.1 的作业用 `csedu-normal`、30 GB。
作业设置了 `HF_HUB_OFFLINE=1`，模型要先在登录节点上下载好。

### 本地运行

```bash
git clone --recurse-submodules https://github.com/diegoruedaa/group18-project.git
pip install -e ".[dev]"
export TRQA_DATA_DIR=/path/to/data          # 语料和索引所在的文件夹，默认 ./data

python scripts/01_prepare_corpus.py         # 下载语料（csedu 上不需要）
python scripts/02_build_index.py            # 建索引，需要几个小时（csedu 上不需要）
python scripts/03_retrieve.py               # 每题检索 Top-1000 -> runs/*.trec
python scripts/04_evaluate_retrieval.py     # 实体级 R@k -> results/eval/
python scripts/05_generate_answers.py       # 阅读器 + 判分 -> results/generation/
python scripts/05_generate_answers.py --method react   # ReAct，运行中检索
```

所有脚本都支持 `-h`。默认值在 `src/dra/config.py`；每次运行都会在 `results/logs/` 写一份日志。

### 目录结构

```
├── src/dra/                 库代码（安装为 `dra` 包）
│   ├── config.py            路径与默认超参数
│   ├── data.py              题目、qrels、语料、TREC run 文件
│   ├── retrieval/           Contriever 编码器与 FAISS 检索
│   ├── evaluation.py        实体级 Recall@k、官方答案判分
│   ├── prompts.py           prompt 模板（取自官方代码）
│   ├── llm.py               阅读器加载、生成、答案解析
│   └── methods/             每种问答方法一个模块（single_retrieval.py、react.py）
├── scripts/01–05            每个流程步骤一个入口脚本
├── cluster/                 csedu 环境、安装与 SLURM 作业脚本
├── tests/                   单元测试（不需要语料、索引或 GPU）
├── external/total-recall-qa 官方仓库（git submodule，不修改）
├── runs/                    检索 run 文件（中间数据，不进 git）
└── results/                 评估结果、生成答案和日志（进 git）
```

各步骤通过 `runs/` 和 `results/` 里的文件衔接（检索结果是 `runs/` 里的 TREC run 文件），
所以修改并重跑后面的步骤时不需要重新检索。

### 与官方实现的一致性

- 索引、问题编码（最长 64 token、fp16、不加前缀）、mean pooling 和 L2 归一化都按官方代码。
- 实体级 Recall@k 按官方 `extract_entity_id` / `entity_recall_at_k`。
- system prompt、`<information>` 模板和段落格式与官方 `SingleRetrieval` 一致。
- 答案用官方 `soft_exact_match` 判分：EM 和 SM 1/5/10/20/50/90%，`decimals=3`，与 `run_evalution.py` 一致。
- 阅读器在 csedu 上用 fp16 运行（`--dtype float16`，分在 2 块 GPU 上）；官方设定是 bf16，RTX 2080 Ti 不支持。
- 与官方 `get_answer` 一致，只解析 `<answer>` 标签；没有标签的输出没有答案，按答错算。
- ReAct 按官方 `ReAct_Model`：相同的说明和 6 个示例、停止字符串、动作解析、20 步上限、强制作答、
  每步从检索到的 1000 个中给 Top-3；单元测试让官方类和我们的类处理相同的模型输出，逐次比对每一次调用。

### 开发

```bash
pytest                  # 单元测试，约 10 秒
ruff format .           # 格式化
ruff check . --fix      # 代码检查
```

提交前把这三条都跑一遍。配置在 `pyproject.toml`。
