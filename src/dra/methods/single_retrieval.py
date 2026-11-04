"""
Single-retrieval baseline: retrieve once, generate once.
单次检索（Single Retrieval）基线：检索一次 + 生成一次。

Same as the official SingleRetrieval (c5_task_evaluation/methods/retrieval_augmented_models.py, lines 111-133):
the top-k passages and the question go into one prompt, and the LLM answers directly without searching again.
对应官方 SingleRetrieval（retrieval_augmented_models.py 第 111-133 行）：把检索到的前 k 个段落和问题
放进一个 prompt，LLM 读完后直接作答，中间不再检索。
"""

from ..prompts import SYSTEM_PROMPT_SINGLE_RETRIEVAL, USER_PROMPT_TEMPLATE, format_passages


def build_messages(question, passages):
    """
    Build the chat messages: the system message gives the rules, the user message holds the passages and question.
    构造 chat 格式的 messages：system 给出作答规则，user 放入检索段落和问题。

    passages: passage records {"id", "contents"} (see prompts.format_passages)
    passages：段落记录 {"id", "contents"}（见 prompts.format_passages）
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT_SINGLE_RETRIEVAL},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(documents=format_passages(passages), question=question),
        },
    ]


class SingleRetrieval:
    """
    Answer a question by letting the given LLM read the retrieved passages once.
    用给定的 LLM 读一次检索到的段落并作答。
    """

    def __init__(self, llm):
        self.llm = llm

    def answer(self, question, passages, **gen_kwargs):
        """
        Build the prompt -> generate once -> return the model's raw output.
        The caller extracts the answer with llm.parse_answer.
        构造 prompt -> 生成一次 -> 返回模型原始输出。答案由调用方用 llm.parse_answer 取出。
        """
        return self.llm.generate(build_messages(question, passages), **gen_kwargs)
