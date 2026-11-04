"""
Step 1: prepare the corpus.
步骤 1：准备语料。

Download the wiki_partial corpus (5.91M passages) from HuggingFace (mahtaa/trqa) and convert it
into the jsonl format expected by the official index_builder: {"id": ..., "contents": "title\\ncontents"}.
Optionally also write a small sample, to test index building on a small corpus first.
从 HuggingFace（mahtaa/trqa）下载 wiki_partial 语料（591 万段落），转换成官方 index_builder
需要的 jsonl 格式：{"id": ..., "contents": "title\\ncontents"}。
可选再切一个小样本，先在小数据上调通建索引流程。

If the corpus file already exists, the download is skipped (use --overwrite to force it).
语料文件已存在时跳过下载（加 --overwrite 强制重新下载）。

Usage / 用法：
    python scripts/01_prepare_corpus.py
    python scripts/01_prepare_corpus.py --sample_size 10000
"""

import argparse
from pathlib import Path

from dra import config
from dra.data import export_corpus, write_sample
from dra.logging_utils import log_args, setup_logging


def main():
    # Read command-line arguments; anything not given falls back to the defaults.
    # 读取命令行参数；没有指定的参数使用默认值。
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--split",
        default="wiki_partial",
        help="corpus split of mahtaa/trqa to download / 下载 mahtaa/trqa 的哪个语料 split",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.CORPUS_PATH,
        help="where to write the converted corpus / 转换后的语料保存位置",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="if > 0, also write the first N passages as a small sample / 大于 0 时额外输出前 N 条小样本",
    )
    # A switch: present -> True, absent -> False.
    # 开关参数：写上为 True，不写为 False。
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="download again even if the corpus file exists / 语料文件已存在时也重新下载",
    )
    args = parser.parse_args()
    logger = setup_logging(Path(__file__).stem, config.LOG_DIR)
    log_args(logger, args)

    # export_corpus is our own function in src/dra/data.py, so it is imported and called directly
    # (unlike step 2, which runs the official command-line script with subprocess).
    # export_corpus 是我们自己在 src/dra/data.py 里写的函数，所以直接 import 调用
    # （不像步骤 2 那样用 subprocess 运行官方的命令行脚本）。
    if args.output.exists() and not args.overwrite:
        logger.info("Corpus already exists, skipping download (use --overwrite to redo): %s", args.output)
    else:
        n = export_corpus(args.output, split=args.split)
        logger.info("Exported %d passages -> %s", n, args.output)

    # Only runs when --sample_size > 0 is given. The sample goes in the same folder as the corpus,
    # e.g. --sample_size 10000 -> sample_10000.jsonl (with_name keeps the folder, replaces the file name).
    # 只有指定了 --sample_size 且大于 0 才执行。小样本和语料放在同一个文件夹，
    # 例如 --sample_size 10000 -> sample_10000.jsonl（with_name：保留文件夹，只换文件名）。
    if args.sample_size > 0:
        sample_path = args.output.with_name(f"sample_{args.sample_size}.jsonl")
        write_sample(args.output, sample_path, args.sample_size)
        logger.info("Wrote sample -> %s", sample_path)


if __name__ == "__main__":
    main()
