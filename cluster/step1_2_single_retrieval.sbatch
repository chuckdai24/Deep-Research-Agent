#!/bin/bash
# Step 1.2 on the csedu cluster: single retrieval + Qwen2.5-7B-Instruct (Table 8 "Single Step Retriever").
# 集群上的 Step 1.2：单次检索 + Qwen2.5-7B-Instruct（Table 8 的 "Single Step Retriever"）。
#
# Needs: Step 1.1 run file (cluster/step1_1_retrieval.sbatch) and the 7B model (cluster/setup.sh).
# 需要：Step 1.1 的 run 文件（cluster/step1_1_retrieval.sbatch）和 7B 模型（cluster/setup.sh）。
#
# Submit from the repo root. Extra arguments are passed on to 05_generate_answers.py, and options before the
# script name override the #SBATCH lines:
# 在仓库根目录提交。脚本名后面的参数会传给 05_generate_answers.py；脚本名前面的选项覆盖 #SBATCH 行：
#     sbatch --time=01:00:00 cluster/step1_2_single_retrieval.sbatch --limit 5   # test run / 先测 5 题
#     sbatch cluster/step1_2_single_retrieval.sbatch                             # all 1258 questions / 全部
#
# Outputs / 输出：
#     results/generation/single_retrieval_qwen7b_fp16.jsonl          per question / 逐题结果
#     results/generation/single_retrieval_qwen7b_fp16_summary.json   EM, SM, s/question / EM、SM、每题耗时
#
# Resources: 7B in fp16 (~15 GB) does not fit on one 11 GB RTX 2080 Ti, so the model is split over 2 GPUs
# (device_map="auto"); csedu-normal allows at most 2 GPUs and 12 h per job.
# 资源：7B 用 fp16 约 15 GB，一块 11 GB 的 RTX 2080 Ti 放不下，所以模型分到 2 块 GPU 上
# （device_map="auto"）；csedu-normal 每个作业最多 2 块 GPU、12 小时。
#SBATCH --partition=csedu
#SBATCH --account=csedui00041
#SBATCH --qos=csedu-normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --job-name=step1_2_single_retrieval
#SBATCH --output=results/%x-%j.out

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

# Models were downloaded by setup.sh; never go online from a GPU node.
# 模型已由 setup.sh 下载；GPU 节点上不联网。
export HF_HUB_OFFLINE=1

echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# --dtype float16: the RTX 2080 Ti has no bfloat16 (the official setting); this is a known deviation.
# --dtype float16：RTX 2080 Ti 不支持官方用的 bfloat16；这是已知的一处不同。
python scripts/05_generate_answers.py \
    --run_file results/runs/contriever_wiki2_test_official.trec \
    --corpus_path "$TRQA_CORPUS" \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dtype float16 \
    --output results/generation/single_retrieval_qwen7b_fp16.jsonl \
    "$@"
