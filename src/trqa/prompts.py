"""
prompt 模板与检索段落的拼接格式。

模板文字和段落格式都与官方仓库保持一致（c5_task_evaluation/prompts/prompt_templetes.py）。
放在这里集中管理，供各个方法（单次检索、ReAct、我们自己的方法）共用。
"""

from .data import split_title

# 官方单次检索 system prompt
SYSTEM_PROMPT_SINGLE_RETRIEVAL = (
    "You are tasked to answer the given question. To do this, first you MUST conduct "
    "reasoning inside <think> and </think>. Then you MUST provide the answer inside "
    "<answer> and </answer>. The retrieved information is inserted into <information> "
    "</information>. Provide only SHORT form answers, NOT complete sentence, without "
    "any additional text or explanation."
)

USER_PROMPT_TEMPLATE = "<information>{documents}</information>\n\nQuestion: {question}"


def format_passages(passages):
    """
    把段落列表拼成一段文字，格式同官方 passages2string：
        Doc 1 (Title: xxx) 正文
        Doc 2 (Title: yyy) 正文
    passages 中每个元素是 "title\\ncontents" 形式的原文。
    """
    lines = []
    for i, contents in enumerate(passages, start=1):
        title, text = split_title(contents)
        lines.append(f"Doc {i} (Title: {title}) {text}")
    return "\n".join(lines) + "\n"
