"""
Evaluation metrics.
评估指标。

Retrieval:  entity-level recall, Entity Recall@k (compared with the qrels)
Generation: exact match / soft match of the answer (compared with "answer" in the queries;
            the official implementation is reused)
检索阶段：实体级召回 Entity Recall@k（与 qrels 对比）
生成阶段：答案精确匹配 / 软匹配（与 queries 里的 answer 对比，直接复用官方实现）
"""

import importlib.util

# ---------------------------------------------------------------------------
# Retrieval evaluation / 检索评估
# ---------------------------------------------------------------------------


def entity_id(passage_id):
    """
    A passage id has the form "<entity id>-<passage number>", e.g. "15641-0001"; return the entity part "15641".
    Same as the official extract_entity_id: split on the last '-' only, so entity ids containing '-' stay intact.
    段落 id 形如 "实体id-段落号"（如 "15641-0001"），取出实体部分 "15641"。
    与官方 extract_entity_id 一致：只切最后一个 '-'，实体 id 本身含 '-' 时也不会切错。
    """
    return passage_id.rsplit("-", 1)[0]


def entity_recall_at_k(retrieved_ids, gold_ids, k):
    """
    Entity Recall@k = (gold entities covered by the top k) / (number of gold entities).
    Entity Recall@k = 前 k 个结果覆盖到的金标实体数 / 金标实体总数。

    TRQA asks for *all relevant entities*, so several passages of one entity count once:
    an entity counts as recalled if any one of its passages is in the top k.
    TRQA 要求"找全所有相关实体"，同一实体的多个段落只算一次：
    只要前 k 个里有某实体的任意一个段落，就认为这个实体被召回了。

    retrieved_ids: ranked passage ids (list)   检索排名（列表，有顺序）
    gold_ids:      relevant passage ids (set)  相关段落（集合）
    Returns a fraction in 0-1 / 返回 0-1 之间的小数.
    """
    # Passage ids -> entity ids. A set removes duplicates: "15641-0001" and "15641-0003" both become "15641".
    # 段落 id -> 实体 id。用集合去重："15641-0001" 和 "15641-0003" 都变成 "15641"，只算一个。
    gold_entities = {entity_id(pid) for pid in gold_ids}
    if not gold_entities:
        return 0.0
    retrieved_entities = {entity_id(pid) for pid in retrieved_ids[:k]}
    # & = set intersection: entities that are both relevant and retrieved.
    # & 是集合交集：既是相关实体、又被检索到的实体。
    return len(gold_entities & retrieved_entities) / len(gold_entities)


def evaluate_retrieval(run, qrels, ks):
    """
    Mean Entity Recall@k over all questions, for every k.
    对整个检索结果计算各 k 下的平均 Entity Recall@k。

    run:   {qid: [passage_id, ...]}  ranked retrieval results / 检索排名
    qrels: {qid: {passage_id, ...}}  gold relevant passages / 金标相关段落
    Questions without qrels are skipped: with nothing to find, recall is undefined (division by zero).
    没有金标的题跳过：没有需要找的东西，召回率无从定义（分母为 0）。

    Returns (summary, per_query) / 返回 (summary, per_query)：
        summary   = {"R@3": 0.93, ..., "num_queries": N, "num_skipped": M}     mean, in % / 平均值，百分比
        per_query = [{"query_id": qid, "num_gold_entities": n, "R@3": ..., ...}, ...]   0-1, one row per question
                    for error analysis and significance tests / 每题一行，用于错误分析和显著性检验
    """
    per_query = []
    skipped = 0
    # One question at a time: qid = question id, retrieved = its ranked passage ids.
    # 逐题处理：qid 是题目 id，retrieved 是这道题检索到的段落 id 排名。
    for qid, retrieved in run.items():
        gold = qrels.get(qid)  # None if the question has no qrels / 没有金标时为 None
        if not gold:
            skipped += 1
            continue  # go to the next question / 直接进入下一题
        row = {"query_id": qid, "num_gold_entities": len({entity_id(p) for p in gold})}
        for k in ks:
            row[f"R@{k}"] = entity_recall_at_k(retrieved, gold, k)
        per_query.append(row)

    # Mean over questions, times 100 to get %. max(..., 1) avoids dividing by zero if every question was skipped.
    # 对所有题取平均再乘 100 得到百分比。max(..., 1) 避免所有题都被跳过时除以 0。
    summary = {f"R@{k}": 100 * sum(r[f"R@{k}"] for r in per_query) / max(len(per_query), 1) for k in ks}
    summary["num_queries"] = len(per_query)
    summary["num_skipped"] = skipped
    return summary, per_query


# ---------------------------------------------------------------------------
# Generation evaluation / 生成评估
# ---------------------------------------------------------------------------


def load_official_answer_metric(metrics_path):
    """
    Load soft_exact_match from the official repository, so answers are scored exactly as in the paper.
    从官方仓库加载 soft_exact_match，保证答案判分与论文完全一致。

    Official logic: take the first number from the prediction and from the gold answer (units and
    thousands separators removed), round them and compare; tolerance_pct allows a relative error.
    官方逻辑：从预测和金标中各取出第一个数字（去掉单位、千分位逗号），四舍五入后比较；
    tolerance_pct 允许一定比例的误差。

    Why not a normal import: the file (external/total-recall-qa/c5_task_evaluation/metrics/
    generation_eval_metrics.py) is not in an installed package, and adding the official folders to the
    Python path could clash with common module names there (utils, metrics). So this one file is
    loaded directly from its path, without affecting anything else.
    为什么不用普通的 import：这个文件不在已安装的包里；把官方文件夹加进 Python 搜索路径，
    又可能和里面 utils、metrics 这类常见模块名冲突。所以直接按路径只加载这一个文件，不影响其他东西。
    """
    # 1. Register a module: its name (our own choice, must be unique) and the file it comes from.
    #    登记一个模块：名字（自己起，不能和别的模块重名）以及它来自哪个文件。
    spec = importlib.util.spec_from_file_location("trqa_official_gen_metrics", metrics_path)
    # 2. Create an empty module object from it.   按登记信息创建一个空的模块对象。
    module = importlib.util.module_from_spec(spec)
    # 3. Run the file, which fills the module with its functions.   运行这个文件，把其中的函数装进模块。
    spec.loader.exec_module(module)
    # 4. Take out the function we need.           取出需要的函数。
    return module.soft_exact_match
