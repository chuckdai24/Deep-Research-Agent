"""
单次检索（Single Retrieval）基线：一次检索 + 一次生成。

对应官方 SingleRetrieval 方法：把检索到的 Top-k 段落和问题一起放进 prompt，
LLM 读完后直接作答，中间不再检索。
"""

from ..prompts import SYSTEM_PROMPT_SINGLE_RETRIEVAL, USER_PROMPT_TEMPLATE, format_passages


def build_messages(question, passages):
    """构造 chat 格式的 messages：system 给出作答规则，user 放入检索段落和问题。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_SINGLE_RETRIEVAL},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(documents=format_passages(passages), question=question),
        },
    ]


class SingleRetrieval:
    """用给定的 LLM 读取检索段落并作答。"""

    def __init__(self, llm):
        self.llm = llm

    def answer(self, question, passages, **gen_kwargs):
        """构造 prompt -> 生成 -> 返回模型原始输出（答案由调用方用 parse_answer 解析）。"""
        return self.llm.generate(build_messages(question, passages), **gen_kwargs)
