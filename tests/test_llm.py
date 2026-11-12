"""
Unit tests for llm.py: dtype names, stop sequences and output parsing (the LLM is not loaded).
llm.py 的单元测试：dtype 名称、停止序列与输出解析（不加载 LLM）。
"""

import pytest
import torch

from dra.llm import StopOnSequence, extract_tag, parse_answer, resolve_dtype


def test_resolve_dtype():
    assert resolve_dtype("float16") is torch.float16
    assert resolve_dtype("bfloat16") is torch.bfloat16


def test_resolve_dtype_unknown_name():
    # A typo must fail loudly instead of silently loading the model in another format.
    # 拼错的名字必须报错，而不是悄悄用别的格式加载模型。
    with pytest.raises(ValueError):
        resolve_dtype("fp16")


def test_extract_tag_takes_first_and_strips():
    text = "<think>x</think><answer> 1886 </answer><answer>2</answer>"
    assert extract_tag(text, "answer") == "1886"
    assert extract_tag(text, "think") == "x"
    assert extract_tag(text, "search") is None


def test_extract_tag_multiline():
    assert extract_tag("<answer>\n42\n</answer>", "answer") == "42"


def test_parse_answer_without_tag():
    raw = "<think>reasoning</think>\n1912"
    assert parse_answer(raw) is None  # official: no <answer>, no answer / 官方设定：没有 <answer> 就没有答案


def test_parse_answer_with_tag():
    assert parse_answer("<answer>1886</answer>\nextra") == "1886"


def test_parse_answer_empty_output():
    assert parse_answer("") is None


class CharTokenizer:
    """
    Fake tokenizer: one token per character (token id = character code).
    假 tokenizer：每个字符一个 token（token id = 字符编码）。
    """

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def ids(text):
    return torch.tensor([CharTokenizer().encode(text)])


def test_stop_on_sequence_stops_only_when_output_ends_with_target():
    stop = StopOnSequence(["Observation 1:", "]"], CharTokenizer())
    assert stop(ids("Action 1: Search[x]\nObservation 1:"), None)
    assert stop(ids("Finish[42]"), None)
    # The target appears, but not at the end: keep going (it is checked after every new token).
    # 目标出现了，但不在末尾：继续生成（每写一个新 token 都会检查一次）。
    assert not stop(ids("Observation 1: more"), None)
    # Shorter than every target / 比所有目标都短
    assert not StopOnSequence(["Observation 1:"], CharTokenizer())(ids("Obs"), None)
