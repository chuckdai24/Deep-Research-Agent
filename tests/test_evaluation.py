"""
Unit tests for evaluation.py: entity-level recall and the official answer metric.
evaluation.py 的单元测试：实体级召回与官方答案判分。
"""

import pytest

from dra import config
from dra.evaluation import (
    entity_id,
    entity_recall_at_k,
    evaluate_retrieval,
    load_official_answer_metric,
)


def test_entity_id_splits_on_last_dash():
    assert entity_id("15641-0001") == "15641"
    assert entity_id("a-b-0003") == "a-b"


def test_entity_recall_counts_each_entity_once():
    gold = {"1-0001", "1-0002", "2-0001"}  # two entities: 1 and 2 / 两个实体：1 和 2
    retrieved = ["1-0002", "3-0001", "2-0005"]

    # Entity 1 counts as recalled if either of its two passages is retrieved.
    # 实体 1 的两个段落只要命中一个就算召回。
    assert entity_recall_at_k(retrieved, gold, k=1) == 0.5
    # 2-0005 is not a gold passage, but belongs to gold entity 2, so it also counts.
    # 2-0005 不是金标段落，但属于金标实体 2，同样算召回。
    assert entity_recall_at_k(retrieved, gold, k=3) == 1.0


def test_entity_recall_empty_gold_is_zero():
    assert entity_recall_at_k(["1-0001"], set(), k=10) == 0.0


def test_evaluate_retrieval_averages_and_skips_queries_without_gold():
    run = {
        "q1": ["1-0001", "9-0001"],
        "q2": ["9-0001", "2-0001"],
        "q3": ["1-0001"],  # no qrels, must be skipped / 没有金标，应被跳过
    }
    qrels = {"q1": {"1-0001"}, "q2": {"2-0001", "3-0001"}}

    summary, per_query = evaluate_retrieval(run, qrels, ks=(1, 2))

    assert summary["num_queries"] == 2
    assert summary["num_skipped"] == 1
    assert summary["R@1"] == pytest.approx(100 * (1.0 + 0.0) / 2)
    assert summary["R@2"] == pytest.approx(100 * (1.0 + 0.5) / 2)
    assert [row["query_id"] for row in per_query] == ["q1", "q2"]
    assert per_query[1]["num_gold_entities"] == 2


@pytest.mark.skipif(not config.OFFICIAL_GEN_METRICS.exists(), reason="official repository submodule not initialised")
@pytest.mark.parametrize(
    ("prediction", "exact", "soft"),
    [
        ("1886", True, True),
        # the number is taken after removing units and thousands separators / 去掉单位和千分位逗号后取数字
        ("about 1,886 years", True, True),
        # 1.4% off, within the 5% tolerance / 偏差 1.4%，在 5% 容差内
        ("1912", False, True),
        ("2100", False, False),
        ("", False, False),
        # parse_answer returns None when there is no tag / 没有标签时 parse_answer 返回 None
        (None, False, False),
    ],
)
def test_official_answer_metric(prediction, exact, soft):
    soft_exact_match = load_official_answer_metric(config.OFFICIAL_GEN_METRICS)
    assert soft_exact_match(prediction, 1886, tolerance_pct=5.0) == {"exact_match": exact, "soft_match": soft}
