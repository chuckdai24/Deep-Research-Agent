"""prompts.py 的单元测试：段落拼接格式。"""

from trqa.prompts import format_passages


def test_format_passages_matches_official_format():
    passages = ["Alpha\nfirst body", "Beta\nsecond body"]
    assert format_passages(passages) == "Doc 1 (Title: Alpha) first body\nDoc 2 (Title: Beta) second body\n"
