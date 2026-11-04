"""prompts.py 的单元测试：段落拼接格式与官方 passages2string 完全一致。"""

import importlib.util

import pytest

from dra import config
from dra.prompts import format_passages

OFFICIAL_UTILS = config.OFFICIAL_REPO / "utils" / "general_utils.py"

PASSAGES = [
    {"id": "1-0001", "contents": "Alpha\nfirst body"},
    {"id": "2-0001", "contents": "Beta\nsecond body\nwith a second line"},  # 正文本身有多行
    {"id": "3-0001", "contents": "Only a title"},  # 没有正文
    {"id": "4-0001", "contents": "body text", "title": "Gamma"},  # 有单独的 title 字段（官方另一个分支）
]


def test_format_passages_example():
    passages = [{"id": "1-0001", "contents": "Alpha\nfirst body"}, {"id": "2-0001", "contents": "Beta\nsecond body"}]
    assert format_passages(passages) == "Doc 1 (Title: Alpha) first body\nDoc 2 (Title: Beta) second body\n"


@pytest.mark.skipif(not OFFICIAL_UTILS.exists(), reason="官方仓库子模块未初始化")
def test_format_passages_identical_to_official():
    spec = importlib.util.spec_from_file_location("trqa_official_general_utils", OFFICIAL_UTILS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert format_passages(PASSAGES) == module.passages2string(PASSAGES)
