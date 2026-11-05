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

# Personal data folder: venv, model cache, mapping tables. Home is only 10 GB, so big files go here.
# 个人数据文件夹：venv、模型缓存、映射表。home 只有 10 GB，大文件放这里。
export MYDATA="${MYDATA:-$COURSE/users/$USER}"

# Where HuggingFace stores downloaded models (default would be ~/.cache in home).
# HuggingFace 下载模型的存放位置（默认在 home 的 ~/.cache）。
export HF_HOME="$MYDATA/hf_cache"

# Shared inputs; each can be overridden by setting it before sourcing this file.
# 公共输入；每个都可以在 source 本文件之前自行设置来覆盖。
#   TRQA_CORPUS            original partial corpus (with title field), provided by the course
#                          原始格式 partial 语料（含 title 字段），课程提供
#   TRQA_CONTRIEVER_INDEX  official Contriever index (HF mahtaa/trqa_experiments), downloaded by Diego
#                          官方 Contriever 索引（HF mahtaa/trqa_experiments），Diego 已下载
export TRQA_CORPUS="${TRQA_CORPUS:-$COURSE/shared/trqa/data/shared/corpus_wikipedia_en-20251001_rewritten_partial.jsonl}"
export TRQA_CONTRIEVER_INDEX="${TRQA_CONTRIEVER_INDEX:-$COURSE/users/drueda/trqa/index/contriever_Flat.index}"

# Activate the venv if it exists (cluster/setup.sh creates it).
# 如果 venv 已存在就激活（由 cluster/setup.sh 创建）。
if [ -f "$MYDATA/venv/bin/activate" ]; then
    source "$MYDATA/venv/bin/activate"
fi
