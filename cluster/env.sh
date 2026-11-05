# Environment for the csedu cluster: paths, HuggingFace cache and the venv.
# csedu 集群的环境设置：路径、HuggingFace 缓存和 venv。
#
# Usage / 用法（in a terminal or a job script / 在终端或作业脚本里）:
#     source ~/group18-project/cluster/env.sh
#
# Works for every group member: personal folders are derived from $USER.
# 每个组员都能用：个人文件夹根据 $USER（登录名）自动确定。
#
# This file is meant to be sourced, not run, so it must not use `set -e` or `exit`
# (that would close your terminal).
# 这个文件是用 source 读入的，不是直接运行，所以不能用 `set -e` 或 `exit`（否则会关掉你的终端）。

# Course folder on the large, non-backed-up storage.
# 课程文件夹，在空间大、不备份的存储上。
export COURSE=/vol/csedu-nobackup/course/I00041_informationretrieval

# Personal data folder: venv, model cache, index, mapping tables. Home is only 10 GB, so big files go here.
# 个人数据文件夹：venv、模型缓存、索引、映射表。home 只有 10 GB，大文件放这里。
export MYDATA="${MYDATA:-$COURSE/users/$USER}"

# Where HuggingFace stores downloaded models (default would be ~/.cache in home).
# HuggingFace 下载模型的存放位置（默认在 home 的 ~/.cache）。
export HF_HOME="$MYDATA/hf_cache"

# Where pip keeps downloaded packages (default ~/.cache/pip; torch alone is several GB).
# pip 缓存下载安装包的位置（默认 ~/.cache/pip；光 torch 就有好几 GB）。
export PIP_CACHE_DIR="$MYDATA/pip_cache"

# Inputs; each can be overridden by setting it before sourcing this file.
# 输入文件；每个都可以在 source 本文件之前自行设置来覆盖。
#
# Original partial corpus (with title field), provided by the course; read-only, no need to copy.
# 原始格式 partial 语料（含 title 字段），课程提供；只读，不需要复制。
export TRQA_CORPUS="${TRQA_CORPUS:-$COURSE/shared/trqa/data/shared/corpus_wikipedia_en-20251001_rewritten_partial.jsonl}"

# Official Contriever index from the authors' HF dataset repo, downloaded by cluster/setup.sh
# into your own folder (same folder structure as on HF).
# 作者 HF 数据集仓库里的官方 Contriever 索引，由 cluster/setup.sh 下载到你自己的文件夹
# （文件夹结构与 HF 上相同）。
export TRQA_HF_REPO=mahtaa/trqa_experiments
export TRQA_HF_REVISION=a95f566b9247e92317182f761b3b8dc67852cf7b
export TRQA_INDEX_FILE=indices/wiki_partial/contriever_partial/contriever_Flat.index
export TRQA_CONTRIEVER_INDEX="${TRQA_CONTRIEVER_INDEX:-$MYDATA/trqa_experiments/$TRQA_INDEX_FILE}"

# Activate the venv if it exists (cluster/setup.sh creates it).
# 如果 venv 已存在就激活（由 cluster/setup.sh 创建）。
if [ -f "$MYDATA/venv/bin/activate" ]; then
    source "$MYDATA/venv/bin/activate"
fi
