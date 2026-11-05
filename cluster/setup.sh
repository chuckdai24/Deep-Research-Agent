#!/bin/bash
# One-time setup on the csedu cluster. Run it on the login node from the repo root:
# csedu 集群的一次性设置。在登录节点、仓库根目录下运行：
#     bash cluster/setup.sh
#
# Safe to run again: steps that are already done are skipped.
# 可以重复运行：已完成的步骤会跳过。
#
# 1. create your data folder and the model cache      创建个人数据文件夹和模型缓存
# 2. create the venv (if missing) and install dra     创建 venv（若不存在）并安装 dra
# 3. download Contriever, so GPU jobs can run offline 下载 Contriever，使 GPU 作业不需要联网
# 4. download the official Contriever index (18 GB)   下载官方 Contriever 索引（18 GB）
#
# Step 4 takes a while. If the connection drops, just run the script again: the download resumes.
# 第 4 步需要一段时间。如果连接断了，重新运行脚本即可，下载会从断点继续。

# Stop at the first error / 出错立即停止
set -eo pipefail

# Repo root = the folder above this script, wherever it was cloned.
# 仓库根目录 = 本脚本的上一级文件夹，不管 clone 到哪里。
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_DIR/cluster/env.sh"

echo "== 1. Data folder: $MYDATA"
mkdir -p "$MYDATA" "$HF_HOME"

echo "== 2. venv: $MYDATA/venv"
if [ ! -f "$MYDATA/venv/bin/activate" ]; then
    python3 -m venv "$MYDATA/venv"
fi
source "$MYDATA/venv/bin/activate"
pip install --upgrade pip
pip install -e "$REPO_DIR[dev]"

echo "== 3. Download Contriever into $HF_HOME"
python -c "
from transformers import AutoModel, AutoTokenizer
m = 'facebook/contriever-msmarco'
AutoTokenizer.from_pretrained(m)
AutoModel.from_pretrained(m)
print('Contriever ready')
"

echo "== 4. Official Contriever index -> $TRQA_CONTRIEVER_INDEX"
if [ ! -f "$TRQA_CONTRIEVER_INDEX" ]; then
    # Only this one file of the authors' HF dataset repo, at a fixed revision (the version of 2026-10-08),
    # so everyone gets exactly the same index. --local-dir keeps the repo's folder structure and does not
    # put a second copy into $HF_HOME.
    # 只下载作者 HF 数据集仓库里的这一个文件，并固定版本（2026-10-08 的版本），保证每个人拿到的索引完全相同。
    # --local-dir 保留仓库里的文件夹结构，不会在 $HF_HOME 里再存一份。
    hf download "$TRQA_HF_REPO" "$TRQA_INDEX_FILE"         --repo-type dataset         --revision "$TRQA_HF_REVISION"         --local-dir "$MYDATA/trqa_experiments"
else
    echo "already downloaded, skipped / 已下载，跳过"
fi

echo "== Done. Check that the inputs exist:"
ls -lh "$TRQA_CORPUS" "$TRQA_CONTRIEVER_INDEX"
