"""data.py 的单元测试：用临时目录里的小文件，不依赖真实语料。"""

import json

from dra.data import (
    load_passage_ids,
    load_passages,
    load_qrels,
    load_queries,
    read_trec_run,
    write_trec_run,
)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


CORPUS = [
    {"id": "10-0001", "contents": "Alpha\nfirst passage"},
    {"id": "10-0002", "contents": "Alpha\nsecond passage"},
    {"id": "20-0001", "contents": "Beta\nthird passage"},
]


def test_load_queries_keeps_order_and_skips_blank_lines(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text('{"id": "q2", "answer": 1}\n\n{"id": "q1", "answer": 2}\n', encoding="utf-8")
    assert [q["id"] for q in load_queries(path)] == ["q2", "q1"]


def test_load_qrels_keeps_only_relevant(tmp_path):
    path = tmp_path / "qrels.txt"
    path.write_text(
        "q1 0 10-0001 1\n"
        "q1 0 20-0001 0\n"  # relevance 0：不算相关
        "q2 0 20-0001 2\n"
        "\n",
        encoding="utf-8",
    )
    assert load_qrels(path) == {"q1": {"10-0001"}, "q2": {"20-0001"}}


def test_trec_run_roundtrip_sorted_by_rank(tmp_path):
    path = tmp_path / "runs" / "run.trec"  # 父目录不存在时应自动创建
    results = {
        "q1": [("10-0001", 0.9), ("20-0001", 0.5)],
        "q2": [("20-0001", 0.8)],
    }
    write_trec_run(path, results)

    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    assert first_line == "q1 Q0 10-0001 1 0.900000 contriever"
    assert read_trec_run(path) == {"q1": ["10-0001", "20-0001"], "q2": ["20-0001"]}


def test_read_trec_run_orders_by_rank_not_file_order(tmp_path):
    path = tmp_path / "run.trec"
    path.write_text("q1 Q0 b 2 0.1 x\nq1 Q0 a 1 0.2 x\n", encoding="utf-8")
    assert read_trec_run(path) == {"q1": ["a", "b"]}


def test_load_passage_ids_follows_corpus_order_and_caches(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    cache = tmp_path / "passage_ids.txt"
    write_jsonl(corpus, CORPUS)

    expected = ["10-0001", "10-0002", "20-0001"]
    assert load_passage_ids(corpus, cache_path=cache) == expected
    assert cache.exists()

    # 第二次应直接读缓存：删掉语料也能得到同样结果
    corpus.unlink()
    assert load_passage_ids(corpus, cache_path=cache) == expected


def test_load_passages_only_wanted_ids(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    write_jsonl(corpus, CORPUS)

    # 返回语料里的整条记录（字典），与官方检索器的格式相同
    passages = load_passages(corpus, wanted_ids=["20-0001", "10-0001"])
    assert passages == {"10-0001": CORPUS[0], "20-0001": CORPUS[2]}

    assert len(load_passages(corpus)) == 3
