"""
Unit tests for llm.py: dtype names and output parsing (the LLM is not loaded).
llm.py 的单元测试：dtype 名称与输出解析（不加载 LLM）。
"""

import pytest
import torch

from dra.llm import extract_tag, parse_answer, resolve_dtype


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
