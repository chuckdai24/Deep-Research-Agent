"""
Unit tests for methods/react.py, mainly: our ReAct and the official ReAct_Model, run side by side.
methods/react.py 的单元测试，主要是：我们的 ReAct 和官方 ReAct_Model 同时运行、逐项对比。

Both get the same scripted LLM outputs (a fake LLM returns them in order) and the same fake retriever.
The test then checks that every prompt sent to the LLM, every stop string, the reasoning path and the
answer are identical. No model, index or GPU is needed.
两者拿到同样的预设 LLM 输出（假 LLM 按顺序返回）和同样的假检索器。然后检查发给 LLM 的每一个 prompt、
每一个停止字符串、推理过程和答案是否完全相同。不需要模型、索引或 GPU。

The official module is imported from the submodule (external/total-recall-qa) without changing it.
Its retriever module needs pyserini/Java, so that one module is replaced by an empty stand-in;
requests/bs4/openai are replaced only if they are not installed (ReAct does not use them).
官方模块直接从子模块导入，不做任何修改。它的检索器模块需要 pyserini/Java，所以只把这一个模块
换成空的替身；requests/bs4/openai 只在没装时才替换（ReAct 用不到它们）。
"""

import ast
import importlib
import importlib.util
import sys
import types
from types import SimpleNamespace

import pytest

from dra import config
from dra.methods.react import ReAct, interleave_rankings, reasoning_path_for_output

OFFICIAL_C5 = config.OFFICIAL_REPO / "c5_task_evaluation"


# ---------------------------------------------------------------------------
# Loading the official code / 加载官方代码
# ---------------------------------------------------------------------------


@pytest.fixture
def official(monkeypatch):
    """
    Import the official retrieval_augmented_models module and return it.
    导入官方 retrieval_augmented_models 模块并返回。
    """
    monkeypatch.syspath_prepend(str(config.OFFICIAL_REPO))
    retrieval_stub = types.ModuleType("c5_task_evaluation.src.retrieval_models_local")
    for name in ["BM25Retriever", "RerankRetriever", "DenseRetriever", "OracleRetriever", "PrecomputedRetriever"]:
        setattr(retrieval_stub, name, object)
    monkeypatch.setitem(sys.modules, "c5_task_evaluation.src.retrieval_models_local", retrieval_stub)
    for name, attr in [("requests", None), ("bs4", "BeautifulSoup"), ("openai", "OpenAI")]:
        if importlib.util.find_spec(name) is None:
            stub = types.ModuleType(name)
            if attr:
                setattr(stub, attr, object)
            monkeypatch.setitem(sys.modules, name, stub)
    module = importlib.import_module("c5_task_evaluation.methods.retrieval_augmented_models")
    # BasicRAG.__init__ would load the LLM and the retriever; skip it, the fakes are attached below.
    # BasicRAG.__init__ 会加载 LLM 和检索器；跳过它，下面再装上假的。
    monkeypatch.setattr(module.BasicRAG, "__init__", lambda self, device, args: setattr(self, "args", args))
    return module


def load_official_function(path, name):
    """
    Take one top-level function out of an official file and return it, without importing the whole file
    (run_evalution.py imports many packages the function does not need).
    从官方文件里取出一个顶层函数并返回，不导入整个文件（run_evalution.py 会导入很多这个函数用不到的包）。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


# ---------------------------------------------------------------------------
# Fakes / 替身
# ---------------------------------------------------------------------------


class CharTokenizer:
    """One token per character, so the official stop criteria can be turned back into strings."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def stops_of(stopping_criteria):
    # Official stop criteria -> the stop strings.   官方的停止条件 -> 停止字符串。
    return ["".join(map(chr, ids)) for ids in stopping_criteria[0].target_ids]


class FakeOfficialGenerator:
    """
    Stands in for the official LLMGenerator_hf_local: returns the scripted outputs in order and records
    (prompt messages, stop strings, max_new_tokens, temperature) of every call.
    代替官方 LLMGenerator_hf_local：按顺序返回预设输出，并记录每次调用的
    (prompt 消息, 停止字符串, max_new_tokens, temperature)。
    """

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []
        self.tokenizer = CharTokenizer()

    def generate(self, messages, stopping_criteria=None, max_new_tokens=1024, do_sample=True, temperature=0.7):
        self.calls.append((messages, stops_of(stopping_criteria), max_new_tokens, temperature))
        return self.outputs.pop(0), None


class FakeLLM:
    """Same as FakeOfficialGenerator, with the interface of dra.llm.LLM."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, messages, max_new_tokens=1024, temperature=0.7, stop_sequences=None):
        self.calls.append((messages, stop_sequences, max_new_tokens, temperature))
        return self.outputs.pop(0)


# A tiny corpus: every query returns these 5 passages, in an order that depends on the query.
# 一个很小的语料：每个查询都返回这 5 个段落，顺序取决于查询。
DOCS = [
    {"id": f"{n}-0001", "title": f"Title {n}", "contents": f"Title {n}\nEntity {n} was born in 18{n}0. It is famous."}
    for n in range(1, 6)
]


def ranking(query):
    shift = len(query) % len(DOCS)
    docs = DOCS[shift:] + DOCS[:shift]
    return docs, [1.0 - 0.1 * i for i in range(len(docs))]


class FakeOfficialRetriever:
    def search(self, query, num=None, return_score=False, qid=None):
        docs, scores = ranking(query)
        return docs[:num], scores[:num]


class FakeStepRetriever:
    def search_for_step(self, query):
        docs, scores = ranking(query)
        return docs[:3], [(d["id"], s) for d, s in zip(docs, scores, strict=True)]


def run_both(official, outputs, max_iter):
    """
    Run the official ReAct_Model and our ReAct on the same question with the same scripted outputs.
    Returns (official LLM calls, our LLM calls, official result, our result).
    用同样的预设输出，让官方 ReAct_Model 和我们的 ReAct 回答同一道题。
    """
    args = SimpleNamespace(
        max_iter=max_iter, generation_method="deep_research", retrieval_eval_ks=[3, 10, 100, 1000], retrieval_topk=3
    )
    off = official.ReAct_Model("cpu", args)
    off.generator = FakeOfficialGenerator(outputs)
    off.retriever = FakeOfficialRetriever()
    ours = ReAct(FakeLLM(outputs), FakeStepRetriever(), max_iter=max_iter)

    off_result = off.inference("How many?", qid="q1")
    our_result = ours.inference("How many?")
    return off.generator.calls, ours.llm.calls, off_result, our_result


def normalise_official_path(path):
    # The official ranking holds full passage records; ours holds passage ids.   官方排序存完整记录，我们存 id。
    out = []
    for step in path:
        step = dict(step)
        if "ranked_list" in step:
            step["ranked_list"] = [(doc["id"], score) for doc, score in step["ranked_list"]]
        out.append(step)
    return out


SCENARIOS = {
    # Search -> Lookup -> Finish, the normal case.   正常情况。
    "search_lookup_finish": (
        [
            " I need to search Entity.\nAction 1: Search[Entity]\nObservation 1:",
            " Look up born.\nAction 2: Lookup[born]\nObservation 2:",
            " Next one.\nAction 3: Lookup[born]\nObservation 3:",
            " So the answer is 1820.\nAction 4: Finish[1820]",
        ],
        20,
    ),
    # Lookup before any Search, "Think", an unknown action type, then Finish.
    # 还没 Search 就 Lookup、"Think"、不认识的动作类型，然后 Finish。
    "lookup_without_page_and_invalid_actions": (
        [
            " Look first.\nAction 1: Lookup[born]\nObservation 1:",
            " Think.\nAction 2: Think[hmm]\nObservation 2:",
            " Odd.\nAction 3: Ponder[x]\nObservation 3:",
            " Done.\nAction 4: Finish[3]",
        ],
        20,
    ),
    # No "Action N:" -> the action is generated separately; a round without "Word[...]" is skipped;
    # max_iter reached without Finish -> forced Finish.
    # 没有 "Action N:" -> 单独生成 action；没有 "单词[...]" 的一轮被跳过；到 max_iter 还没 Finish -> 强制 Finish。
    "regenerate_skip_and_forced_finish": (
        [
            " I should search Beta.",
            " Search[Beta]\n",
            " Hmm.\nAction 2: nothing useful",
            " x\nAction 3: Search[A]\nAction 3: Search[B]",
            " Lookup[born]\n",
            " Again.\nAction 4: Search[Gamma]",
            "42]",
        ],
        4,
    ),
}


@pytest.mark.parametrize("name", SCENARIOS)
def test_react_matches_official(official, name):
    outputs, max_iter = SCENARIOS[name]
    off_calls, our_calls, (off_path, off_pred), (our_path, our_pred) = run_both(official, outputs, max_iter)

    # Every LLM call: same messages (instruction, examples, question, steps), stop strings, lengths, temperature.
    # 每次 LLM 调用：消息（指令、示例、问题、各步）、停止字符串、长度、温度都相同。
    assert len(our_calls) == len(off_calls) == len(outputs)
    for ours, off in zip(our_calls, off_calls, strict=True):
        assert ours == off
    assert our_path == normalise_official_path(off_path)
    assert our_pred == off_pred


def test_react_all_rounds_invalid_raises_like_official(official):
    # No round has a valid action: the official code fails with UnboundLocalError (a NameError); so does ours.
    # 没有任何一轮有有效动作：官方代码报 UnboundLocalError（NameError 的一种）；我们的也一样。
    outputs = [" a\nAction 1: none", " b\nAction 2: none"]
    args = SimpleNamespace(max_iter=2, generation_method="deep_research", retrieval_eval_ks=[1000], retrieval_topk=3)
    off = official.ReAct_Model("cpu", args)
    off.generator = FakeOfficialGenerator(outputs)
    off.retriever = FakeOfficialRetriever()
    with pytest.raises(NameError):
        off.inference("How many?", qid="q1")
    with pytest.raises(NameError):
        ReAct(FakeLLM(outputs), FakeStepRetriever(), max_iter=2).inference("How many?")


def test_first_prompt_starts_with_instruction_and_examples(official):
    # A readable check of what the model sees first (the side-by-side test above checks it exactly).
    # 直观检查模型最先看到的内容（上面的对比测试会逐字检查）。
    llm = FakeLLM([" x\nAction 1: Finish[1]"])
    ReAct(llm, FakeStepRetriever()).inference("How many?")
    messages = llm.calls[0][0]
    assert messages[0] == {"role": "system", "content": ""}
    assert messages[1]["content"].startswith("Solve a question answering task with interleaving Thought")
    assert messages[1]["content"].endswith("Question: How many?\nThought 1:")
    assert messages[1]["content"].count("Question: ") == 7  # 6 examples + ours / 6 个示例 + 我们的问题


# ---------------------------------------------------------------------------
# Output helpers / 输出用的辅助函数
# ---------------------------------------------------------------------------

ID = {"get_passage_id_fn": lambda doc: doc["id"]}


def test_interleave_rankings_matches_official():
    official_fn = load_official_function(OFFICIAL_C5 / "run_evalution.py", "_interleave_rank_lists_with_dedup")
    steps = [
        [("a", 0.9), ("b", 0.8), ("c", 0.7)],
        [("b", 0.95), ("d", 0.5)],
        [("e", 0.4), ("a", 0.3), ("f", 0.2), ("g", 0.1)],
    ]
    official_steps = [[({"id": pid}, s) for pid, s in step] for step in steps]

    ours = interleave_rankings(steps)
    assert ours == official_fn(official_steps, **ID)
    # rank 1 of every step, then rank 2, ...; "b" of step 2 and "a" of step 3 were already taken.
    # 先取每一步的第 1 名，再取第 2 名……；第 2 步的 "b" 和第 3 步的 "a" 之前已经取过。
    assert [pid for pid, *_ in ours] == ["a", "b", "e", "d", "c", "f", "g"]


def test_reasoning_path_for_output_matches_official():
    official_fn = load_official_function(OFFICIAL_C5 / "run_evalution.py", "_reasoning_path_doc_ids_only")
    path = [
        {"think": "t", "action_type": "search", "search_query": "q", "docs": DOCS[:2], "ranked_list": [("x", 1.0)]},
        {"think": "t2", "action_type": "lookup", "entity": "born", "observation": "o"},
        {"think": "t3", "prediction": "42"},
    ]
    assert reasoning_path_for_output(path) == official_fn(path, **ID)
    assert reasoning_path_for_output(path)[0]["doc_ids"] == ["1-0001", "2-0001"]
