"""
步骤 2：构建 Contriever + FAISS Flat 索引（调用官方 index_builder.py）。

把每个段落编码成 768 维向量并存入 FAISS。这是整个流程最耗时的一步
（591 万段落，索引约 18 GB），只需做一次。
产物：<save_dir>/contriever_Flat.index 与 emb_contriever.memmap

用法：
    python scripts/02_build_index.py
    python scripts/02_build_index.py --corpus_path data/sample_10000.jsonl --save_dir data/my_indices/debug
"""

import argparse
import subprocess
import sys
from pathlib import Path

from trqa import config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus_path", type=Path, default=config.CORPUS_PATH)
    parser.add_argument("--save_dir", type=Path, default=config.INDEX_DIR)
    parser.add_argument("--max_length", type=int, default=config.MAX_LENGTH)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(config.INDEX_BUILDER_SCRIPT),
        "--retrieval_method",
        "contriever",
        "--corpus_path",
        str(args.corpus_path),
        "--save_dir",
        str(args.save_dir),
        "--use_fp16",
        "--max_length",
        str(args.max_length),
        "--batch_size",
        str(args.batch_size),
    ]
    print("运行：", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
