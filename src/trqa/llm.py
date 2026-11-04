"""
LLM 的加载、生成与输出解析。

与具体方法无关：单次检索、ReAct 等方法都通过这里的 LLM 生成文本，
再用 extract_tag / parse_answer 从输出里取出 <answer>、<search> 等标签内容。
"""

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------


class LLM:
    """基于 HuggingFace 本地模型的 chat LLM。"""

    def __init__(self, model_name, device=None, dtype=torch.bfloat16):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()

    @torch.no_grad()
    def generate(self, messages, max_new_tokens=512, temperature=0.7):
        """
        生成回复，只返回新生成的部分（去掉输入 prompt）。
        temperature 与官方默认一致取 0.7 并采样，因此多次运行结果会有波动。
        """
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# 输出解析
# ---------------------------------------------------------------------------


def extract_tag(text, tag):
    """取出第一个 <tag>...</tag> 中的内容；没有则返回 None。"""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def parse_answer(text, fallback=False):
    """
    从模型输出中解析答案。

    官方做法只认 <answer> 标签。小模型（如 1.5B）经常不按格式输出、直接给出答案，
    fallback=True 时退而取输出的最后一个非空行（会偏离官方设定，报告时需注明）。
    """
    answer = extract_tag(text, "answer")
    if answer is None and fallback:
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        answer = lines[-1] if lines else None
    return answer
