"""
步骤 1：准备语料。

从 HuggingFace 下载 wiki_partial 语料，转换成官方 index_builder 需要的 jsonl 格式；
可选再切一个小样本，用于先在小数据上调通建索引流程。

用法：
    python scripts/01_prepare_corpus.py
    python scripts/01_prepare_corpus.py --sample_size 10000
"""

import argparse
from pathlib import Path

from trqa import config
from trqa.data import export_corpus, write_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="wiki_partial", help="mahtaa/trqa 的语料 split")
    parser.add_argument("--output", type=Path, default=config.CORPUS_PATH)
    parser.add_argument("--sample_size", type=int, default=0, help=">0 时额外输出前 N 条小样本")
    parser.add_argument("--overwrite", action="store_true", help="语料文件已存在时也重新导出")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        print(f"语料已存在，跳过导出：{args.output}（加 --overwrite 重新导出）")
    else:
        n = export_corpus(args.output, split=args.split)
        print(f"已导出 {n} 条段落 -> {args.output}")

    if args.sample_size > 0:
        sample_path = args.output.with_name(f"sample_{args.sample_size}.jsonl")
        write_sample(args.output, sample_path, args.sample_size)
        print(f"已写出小样本 -> {sample_path}")


if __name__ == "__main__":
    main()
