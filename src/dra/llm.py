"""
Loading the LLM, generating text and parsing its output.
LLM 的加载、生成与输出解析。

Independent of any particular method: single retrieval, ReAct, etc. all generate text through LLM here,
then use extract_tag / parse_answer to pull out the contents of tags such as <answer> or <search>.
与具体方法无关：单次检索、ReAct 等方法都通过这里的 LLM 生成文本，
再用 extract_tag / parse_answer 从输出里取出 <answer>、<search> 等标签的内容。
"""

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Model / 模型
# ---------------------------------------------------------------------------


# Names accepted for the number format of the weights (command line -> torch type).
# 权重数值格式可用的名字（命令行参数 -> torch 类型）。
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_dtype(name):
    """
    Turn a dtype name such as "float16" into the torch type; raise ValueError for an unknown name.
    把 "float16" 这样的名字转换成 torch 类型；名字不认识时报 ValueError。
    """
    if name not in DTYPES:
        raise ValueError(f"unknown dtype {name!r}, choose from {sorted(DTYPES)}")
    return DTYPES[name]


class LLM:
    """
    A chat LLM run locally with HuggingFace transformers.
    基于 HuggingFace transformers 在本地运行的 chat LLM。
    """

    def __init__(self, model_name, dtype=torch.bfloat16):
        """
        Load the tokenizer and the model onto the available GPUs. A model too large for GPU memory fails here.
        加载 tokenizer 和模型到可用的 GPU 上。模型太大、显存放不下时会在这里报错。

        dtype: number format of the weights; bfloat16 and float16 both take 2 bytes per parameter
        (7B ≈ 15 GB). The official code uses bfloat16; older GPUs such as the RTX 2080 Ti on csedu do not
        support it, so use float16 there.
        dtype：权重的数值格式；bfloat16 和 float16 每个参数都占 2 字节（7B 约 15 GB）。
        官方用 bfloat16；csedu 上的 RTX 2080 Ti 等旧 GPU 不支持，那里要用 float16。

        device_map="auto" (needs the accelerate package) places the layers over all visible GPUs: with one
        GPU the whole model goes there, with two GPUs (7B on 11 GB cards) the first layers go to GPU 0 and
        the rest to GPU 1. Only one GPU computes at a time, so two GPUs fit a larger model but are not faster.
        Without a GPU the model stays on the CPU.
        device_map="auto"（需要 accelerate 包）把各层分配到所有可见的 GPU 上：只有一块 GPU 时整个模型放在
        那里；两块 GPU 时（11 GB 的卡跑 7B）前面的层放 GPU 0、其余放 GPU 1。同一时刻只有一块卡在计算，
        所以两块卡能放下更大的模型，但不会更快。没有 GPU 时模型留在 CPU 上。
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # .eval(): inference mode (e.g. turns off dropout).   .eval()：推理模式（例如关闭 dropout）。
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map="auto").eval()

    # Decorator: while this function runs, PyTorch does not record the computation for training
    # (no gradients), which saves GPU memory and time. We only use the model, never train it.
    # 装饰器：运行这个函数时 PyTorch 不记录用于训练的计算过程（不算梯度），省显存、更快。
    # 我们只使用模型，不训练模型。
    @torch.no_grad()
    def generate(self, messages, max_new_tokens=1024, temperature=0.7):
        """
        Generate a reply to the chat messages and return only the newly generated text.
        根据对话 messages 生成回复，只返回新生成的文字。

        max_new_tokens: at most this many tokens are written (official default: 1024).
        temperature: 0.7 with sampling, as in the official code, so results vary between runs
                     unless the seed is fixed (dra.utils.set_seed).
        max_new_tokens：最多写这么多个 token（官方默认 1024）。
        temperature：与官方一致取 0.7 并采样，所以不固定随机种子时每次结果会不同（见 dra.utils.set_seed）。
        """
        # 1. Chat messages -> one string in the model's own chat format (for Qwen:
        #    "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n").
        #    add_generation_prompt=True appends the "assistant" header, i.e. "now it's your turn to answer".
        #    对话列表 -> 模型自己约定的对话格式的一长串文字（Qwen 的格式见上）。
        #    add_generation_prompt=True 在末尾加上 assistant 开头，意思是"现在轮到你回答了"。
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 2. Text -> token ids as a PyTorch tensor. Shape: (1, prompt length). The ids go to the device of the
        #    model's first layer (self.model.device), where the computation starts; with device_map the model
        #    may be spread over two GPUs, and accelerate moves the data between them.
        #    文字 -> token id（PyTorch 张量）。形状：(1, prompt 长度)。放到模型第一层所在的设备
        #    （self.model.device），计算从那里开始；用了 device_map 后模型可能分在两块 GPU 上，
        #    数据在两块卡之间的搬运由 accelerate 完成。
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # 3. Generate token by token until the end-of-turn token or max_new_tokens. The slowest step.
        #    do_sample: temperature > 0 -> sample the next token by probability; 0 -> always the most likely one.
        #    temperature is only meaningful when sampling, so pass None otherwise (avoids a warning).
        #    一个 token 一个 token 地往后写，直到写出结束标记或达到 max_new_tokens。最慢的一步。
        #    do_sample：temperature > 0 时按概率抽下一个 token；为 0 时总选概率最高的。
        #    只有采样时 temperature 才有意义，否则传 None（避免警告）。
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
        )

        # 4. outputs[0] is the prompt followed by the new tokens; cut off the prompt and turn the rest back into text.
        #    skip_special_tokens=True removes markers such as <|im_end|>.
        #    outputs[0] 是"输入 prompt + 新写的 token"连在一起；切掉 prompt 部分，把剩下的变回文字。
        #    skip_special_tokens=True 去掉 <|im_end|> 这类特殊标记。
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Output parsing / 输出解析
# ---------------------------------------------------------------------------


def extract_tag(text, tag):
    """
    Return the contents of the first <tag>...</tag> (surrounding whitespace removed), or None if there is none.
    取出第一个 <tag>...</tag> 中的内容（去掉首尾空白）；没有则返回 None。

    Same as the official get_answer / get_think (c5_task_evaluation/methods/retrieval_augmented_models.py,
    lines 79-92), which also take the first match. The official code does not strip whitespace, but the
    official scorer only extracts the number, so this makes no difference to the scores.
    与官方 get_answer / get_think（retrieval_augmented_models.py 第 79-92 行）一致，同样取第一个匹配。
    官方不去首尾空白，但官方判分只取其中的数字，所以对分数没有影响。
    """
    # Regular expression: (.*?) captures as little as possible up to the first closing tag;
    # re.DOTALL lets "." also match newlines, so answers spanning several lines are found too.
    # 正则表达式：(.*?) 尽量少地匹配，到第一个结束标签为止；re.DOTALL 让 "." 也匹配换行，跨多行的内容也能取到。
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def parse_answer(text, fallback=False):
    """
    Extract the answer from the model output.
    从模型输出中取出答案。

    The official code only accepts <answer> tags: no tag means no answer (None).
    Small models (e.g. 1.5B) often answer directly without tags; with fallback=True the last non-empty
    line is used instead. This is NOT the official setting and must be stated when reporting results.
    官方只认 <answer> 标签：没有标签就是没有答案（None）。
    小模型（如 1.5B）经常不写标签、直接给答案；fallback=True 时退而取输出的最后一个非空行。
    这不是官方设定，报告结果时需注明。
    """
    answer = extract_tag(text, "answer")
    if answer is None and fallback:
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        answer = lines[-1] if lines else None
    return answer
