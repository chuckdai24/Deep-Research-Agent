"""methods/single_retrieval.py 的单元测试：prompt 构造与调用 LLM 的方式（用假 LLM）。"""

from trqa.methods.single_retrieval import SingleRetrieval, build_messages
from trqa.prompts import SYSTEM_PROMPT_SINGLE_RETRIEVAL


def test_build_messages():
    messages = build_messages("Who?", ["Alpha\nbody"])
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT_SINGLE_RETRIEVAL
    assert messages[1]["content"] == "<information>Doc 1 (Title: Alpha) body\n</information>\n\nQuestion: Who?"


class FakeLLM:
    """记录收到的 messages 和生成参数，返回固定输出。"""

    def generate(self, messages, **gen_kwargs):
        self.messages = messages
        self.gen_kwargs = gen_kwargs
        return "<answer>1886</answer>"


def test_answer_passes_prompt_and_generation_args_to_llm():
    llm = FakeLLM()
    raw = SingleRetrieval(llm).answer("Who?", ["Alpha\nbody"], max_new_tokens=8, temperature=0.0)

    assert raw == "<answer>1886</answer>"
    assert llm.messages == build_messages("Who?", ["Alpha\nbody"])
    assert llm.gen_kwargs == {"max_new_tokens": 8, "temperature": 0.0}
