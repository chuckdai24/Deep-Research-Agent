"""llm.py 的单元测试：输出解析（不加载 LLM）。"""

from dra.llm import extract_tag, parse_answer


def test_extract_tag_takes_first_and_strips():
    text = "<think>x</think><answer> 1886 </answer><answer>2</answer>"
    assert extract_tag(text, "answer") == "1886"
    assert extract_tag(text, "think") == "x"
    assert extract_tag(text, "search") is None


def test_extract_tag_multiline():
    assert extract_tag("<answer>\n42\n</answer>", "answer") == "42"


def test_parse_answer_without_tag():
    raw = "<think>reasoning</think>\n1912"
    assert parse_answer(raw) is None  # 官方设定：没有 <answer> 就没有答案
    assert parse_answer(raw, fallback=True) == "1912"  # 退而取最后一个非空行


def test_parse_answer_prefers_tag_over_fallback():
    assert parse_answer("<answer>1886</answer>\nextra", fallback=True) == "1886"


def test_parse_answer_empty_output():
    assert parse_answer("", fallback=True) is None
