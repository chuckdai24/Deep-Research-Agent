"""
评估指标。

检索阶段：实体级召回 Entity Recall@k（与 qrels 对比）
生成阶段：答案精确匹配 / 软匹配（与 queries 里的 answer 对比，直接复用官方实现）
"""

import importlib.util

# ---------------------------------------------------------------------------
# 检索评估
# ---------------------------------------------------------------------------


def entity_id(passage_id):
    """
    passage id 形如 "实体id-段落号"（如 "15641-0001"），取出实体部分 "15641"。
    与官方 extract_entity_id 一致：只切最后一个 '-'。
    """
    return passage_id.rsplit("-", 1)[0]


def entity_recall_at_k(retrieved_ids, gold_ids, k):
    """
    Entity Recall@k = Top-k 中覆盖到的金标实体数 / 金标实体总数。

    TRQA 要求"找全所有相关实体"，同一实体的多个段落只算一次：
    只要检索到某实体的任意一个段落，就认为这个实体被召回了。
    """
    gold_entities = {entity_id(pid) for pid in gold_ids}
    if not gold_entities:
        return 0.0
    retrieved_entities = {entity_id(pid) for pid in retrieved_ids[:k]}
    return len(gold_entities & retrieved_entities) / len(gold_entities)


def evaluate_retrieval(run, qrels, ks):
    """
    对整个检索结果计算各 k 下的平均 Entity Recall@k（百分比）。

    run:   {qid: [doc_id, ...]}  检索排名
    qrels: {qid: {doc_id, ...}}  金标相关段落
    没有金标的 query 跳过（无法定义召回率）。

    返回 (summary, per_query)：
        summary   = {"R@3": 0.93, ..., "num_queries": N, "num_skipped": M}
        per_query = [{"query_id": qid, "R@3": ..., ...}, ...]  便于做逐题分析或显著性检验
    """
    per_query = []
    skipped = 0
    for qid, retrieved in run.items():
        gold = qrels.get(qid)
        if not gold:
            skipped += 1
            continue
        row = {"query_id": qid, "num_gold_entities": len({entity_id(p) for p in gold})}
        for k in ks:
            row[f"R@{k}"] = entity_recall_at_k(retrieved, gold, k)
        per_query.append(row)

    summary = {f"R@{k}": 100 * sum(r[f"R@{k}"] for r in per_query) / max(len(per_query), 1) for k in ks}
    summary["num_queries"] = len(per_query)
    summary["num_skipped"] = skipped
    return summary, per_query


# ---------------------------------------------------------------------------
# 生成评估
# ---------------------------------------------------------------------------


def load_official_answer_metric(metrics_path):
    """
    从官方仓库动态加载 soft_exact_match，保证答案判分与论文完全一致。

    官方逻辑：从预测和金标中各抽出第一个数字（去掉单位、千分位逗号），
    四舍五入后比较；可选 tolerance_pct 做容差匹配。
    """
    spec = importlib.util.spec_from_file_location("trqa_official_gen_metrics", metrics_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.soft_exact_match
