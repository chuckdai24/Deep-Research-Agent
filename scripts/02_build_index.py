"""
Step 2: build the Contriever + FAISS Flat index by calling the official index_builder.py.
步骤 2：调用官方 index_builder.py，构建 Contriever + FAISS Flat 索引。

Every passage is encoded into a 768-dim vector and stored in FAISS. This is the slowest
step of the pipeline (5.91M passages, ~18 GB index) and only needs to be run once.
把每个段落编码成 768 维向量并存入 FAISS。这是整个流程最耗时的一步
（591 万段落，索引约 18 GB），只需做一次。

Output / 产物：<save_dir>/contriever_Flat.index and emb_contriever.memmap

This script does not build the index itself. It only assembles the command line and
runs the official script, so the index is built exactly as in the paper.
本脚本自己不建索引，只是拼好命令行参数、调用官方脚本，保证建索引的方式与论文完全一致。

WARNING: the default --save_dir is the existing index directory and will be overwritten.
注意：默认的 --save_dir 就是现有索引所在目录，运行会覆盖它。

Usage / 用法：
    python scripts/02_build_index.py
    python scripts/02_build_index.py --corpus_path data/sample_10000.jsonl --save_dir data/my_indices/debug
"""

import argparse
import subprocess
import sys
from pathlib import Path

from dra import config
from dra.logging_utils import log_args, setup_logging


def main():
    # Read command-line arguments; anything not given falls back to the defaults in config.py.
    # 读取命令行参数；没有指定的参数使用 config.py 里的默认值。
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--corpus_path",
        type=Path,
        default=config.CORPUS_PATH,
        help="corpus jsonl to encode / 要编码的语料",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=config.INDEX_DIR,
        help="where to write the index / 索引保存目录",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=config.MAX_LENGTH,
        help="max tokens per passage / 每个段落最多取多少 token",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="passages encoded per batch; lower it if GPU memory runs out / 每批编码的段落数，显存不够时调小",
    )
    args = parser.parse_args()
    logger = setup_logging(Path(__file__).stem, config.LOG_DIR)
    log_args(logger, args)

    # Assemble the command as a list: each element is one "word" of the command line.
    # A list (instead of one long string) keeps paths with spaces, such as "Course Material", intact.
    # 把命令拼成列表，每个元素是命令行里的一个"词"。
    # 用列表而不是一整个字符串，路径里的空格（如 "Course Material"）不会被拆开。
    # fmt: off
    cmd = [
        # The Python currently running, i.e. the venv's (not whatever "python" is first on PATH)
        # 当前正在运行的 Python，即 venv 里的（而不是 PATH 上第一个 python）
        sys.executable,
        # Official script: external/total-recall-qa/c2_corpus_creation/index_builder.py
        # 官方脚本
        str(config.INDEX_BUILDER_SCRIPT),
        # From here on: argument name, argument value (names are defined by the official script)
        # 以下是"参数名, 参数值"两两一组（参数名由官方脚本规定）
        "--retrieval_method", "contriever",
        "--corpus_path",      str(args.corpus_path),
        "--save_dir",         str(args.save_dir),
        # A switch without a value: half precision, faster and uses less GPU memory
        # 开关，后面没有值：半精度计算，更快、更省显存
        "--use_fp16",
        "--max_length",       str(args.max_length),
        "--batch_size",       str(args.batch_size),
    ]
    # fmt: on
    # Note: the official script's own output (progress bars) goes to the terminal only, not to our log file.
    # 注意：官方脚本自己的输出（进度条等）只显示在终端，不会进入我们的日志文件。
    logger.info("Running: %s", " ".join(cmd))

    # Start the official script as a child process and wait for it to finish.
    # check=True: if the official script fails, this script fails too instead of exiting silently.
    # 以子进程方式启动官方脚本，并等待它结束。
    # check=True：官方脚本出错时本脚本也报错，而不是悄悄正常结束。
    subprocess.run(cmd, check=True)
    logger.info("Index written to %s", args.save_dir)


if __name__ == "__main__":
    main()
