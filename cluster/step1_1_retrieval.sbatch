#!/bin/bash
# Step 1.1 on the csedu cluster: retrieval with the official Contriever index, then entity-level R@k.
# 集群上的 Step 1.1：用官方 Contriever 索引检索，然后计算实体级 R@k。
#
# Submit from the repo root (run cluster/setup.sh once before):
# 在仓库根目录提交（之前先运行一次 cluster/setup.sh）：
#     sbatch cluster/step1_1_retrieval.sbatch
#     squeue -u $USER                          # job status / 作业状态
#     tail -f results/step1_1_retrieval-*.out  # follow the output / 实时看输出
#
# Outputs / 输出：
#     results/runs/contriever_wiki2_test_official.trec
#     results/eval_official/retrieval_summary.json
#     results/logs/                            # log files of 03 and 04 / 03 和 04 的日志

# Lines starting with #SBATCH are read by SLURM (resources), not by bash.
# Environment variables do not work in them, hence the relative output path (relative to where you run sbatch).
# 以 #SBATCH 开头的行由 SLURM 读取（申请资源），bash 当作注释。
# 这些行里不能用环境变量，所以输出路径写成相对路径（相对于运行 sbatch 的目录）。
#SBATCH --partition=csedu
#SBATCH --account=csedui00041
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --job-name=step1_1_retrieval
#SBATCH --output=results/%x-%j.out

set -eo pipefail

# SLURM starts the job in the folder where sbatch was run; go there explicitly anyway.
# SLURM 会在运行 sbatch 的目录启动作业；这里再明确 cd 一次。
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

# Models were downloaded by setup.sh; never go online from a GPU node.
# 模型已由 setup.sh 下载；GPU 节点上不联网。
export HF_HUB_OFFLINE=1

echo "Node: $(hostname)   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# 03: retrieval. The index needs no building; only the position -> passage id table is created on the
# first run (about 85 MB, cached in $MYDATA), because the official index comes without it.
# 03：检索。索引不用建；第一次运行只会生成"位置 -> 段落 id"映射表（约 85 MB，缓存在 $MYDATA），
# 因为官方索引没有附带这个表。
python scripts/03_retrieve.py \
    --index_path "$TRQA_CONTRIEVER_INDEX" \
    --corpus_path "$TRQA_CORPUS" \
    --passage_ids_cache "$MYDATA/passage_ids_official.txt" \
    --output results/runs/contriever_wiki2_test_official.trec

# 04: entity-level Recall@3/10/100/1000.   04：实体级 Recall@3/10/100/1000。
python scripts/04_evaluate_retrieval.py \
    --run_file results/runs/contriever_wiki2_test_official.trec \
    --output_dir results/eval_official
