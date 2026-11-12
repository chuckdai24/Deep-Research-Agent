#!/bin/bash
# Step 1.2 on the csedu cluster: ReAct + Qwen2.5-7B-Instruct with Contriever (Table 8 "ReAct").
# 集群上的 Step 1.2：ReAct + Qwen2.5-7B-Instruct，检索器为 Contriever（Table 8 的 "ReAct"）。
#
# Needs: the official index, the corpus and the 7B model (cluster/setup.sh). No run file: ReAct searches
# while it runs, so the index is loaded in this job.
# 需要：官方索引、语料和 7B 模型（cluster/setup.sh）。不需要 run 文件：ReAct 在运行中检索，
# 所以索引在这个作业里加载。
#
# Submit from the repo root. Extra arguments are passed on to 05_generate_answers.py, and options before the
# script name override the #SBATCH lines:
# 在仓库根目录提交。脚本名后面的参数会传给 05_generate_answers.py；脚本名前面的选项覆盖 #SBATCH 行：
#     sbatch --time=01:00:00 cluster/step1_2_react.sbatch --limit 3 --output results/generation/react_test.jsonl
#     sbatch cluster/step1_2_react.sbatch          # all 1258 questions / 全部
#
# Resuming: questions already in the output file are skipped, so if the job stops (time limit, error),
# submit the same command again and it continues.
# 断点续跑：输出文件里已有的题会被跳过，所以作业停止后（超时、报错），再提交一次同样的命令就会接着跑。
#
# Outputs / 输出：
#     results/generation/react_qwen7b_fp16.jsonl          per question / 逐题结果
#     results/generation/react_qwen7b_fp16_summary.json   EM, SM, entity recall, s/question
#     runs/react_qwen7b_fp16.trec                         merged ranking per question (not tracked by git)
#     runs/react_qwen7b_fp16_steps.jsonl                  top 1000 of every Search step (not tracked by git)
#
# Resources: the 7B model is split over 2 GPUs (as in single retrieval); the index (18 GB) is in memory,
# so 30G (csedu-normal allows 31G per job). FAISS searches on the CPU and uses all cores of the job.
# On the first run the corpus line positions are built once (~47 MB, cached in $MYDATA).
# 资源：7B 模型分在 2 块 GPU 上（与单次检索相同）；索引（18 GB）放在内存里，所以要 30G
# （csedu-normal 每个作业最多 31G）。FAISS 在 CPU 上检索，使用作业的全部核。
# 第一次运行时会生成一次语料行位置表（约 47 MB，缓存在 $MYDATA）。
#SBATCH --partition=csedu
#SBATCH --account=csedui00041
#SBATCH --qos=csedu-normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --job-name=step1_2_react
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
    --method react \
    --index_path "$TRQA_CONTRIEVER_INDEX" \
    --corpus_path "$TRQA_CORPUS" \
    --passage_ids_cache "$MYDATA/passage_ids_official.txt" \
    --offsets_cache "$MYDATA/corpus_line_offsets_official.npy" \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dtype float16 \
    --output results/generation/react_qwen7b_fp16.jsonl \
    "$@"

# Peak memory of this job, to check the 30G limit.   本作业的内存峰值，用来核对 30G 上限。
sstat -j "$SLURM_JOB_ID.batch" --format=MaxRSS 2>/dev/null || true
