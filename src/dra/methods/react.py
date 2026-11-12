"""
ReAct: the LLM alternates Thought / Action / Observation steps and decides itself when and what to search.
ReAct：LLM 交替写出 Thought / Action / Observation，自己决定什么时候检索、检索什么。

Copied from the official ReAct_Model (external/total-recall-qa/c5_task_evaluation/methods/
retrieval_augmented_models.py, lines 268-623): the same instruction, the same 6 few-shot examples, the same
parsing, loop and stop strings, so the reproduction behaves like the paper's code, including its quirks
(listed below). Only the infrastructure is ours: the LLM (dra.llm.LLM) and the retriever
(dra.retrieval.StepRetriever). tests/test_react.py runs this class and the official one side by side with
the same scripted LLM outputs and checks that every prompt and every result is identical.
照搬官方 ReAct_Model（retrieval_augmented_models.py 第 268-623 行）：同样的指令、同样的 6 个示例、同样的解析方式、
循环和停止字符串，使复现的行为和论文代码一致，连它的特殊之处也一样（见下）。只有基础设施是我们自己的：
LLM（dra.llm.LLM）和检索器（dra.retrieval.StepRetriever）。tests/test_react.py 用同一组预设的 LLM 输出
同时运行本类和官方类，检查每一个 prompt 和每一个结果都完全相同。

One round (at most max_iter = 20) / 每一轮（最多 max_iter = 20 轮）：
    1. the LLM continues the prompt from "Thought N:" and is stopped at "Observation N:"
       LLM 从 "Thought N:" 接着写，写到 "Observation N:" 时被停下
    2. the text is split at "Action N:" into thought and action; if that fails, the LLM is asked again
       for the action alone
       按 "Action N:" 把文字切成 thought 和 action；切不开时，让 LLM 单独再写一次 action
    3. a regular expression finds "Word[...]" in the action:
         Search[x]  -> retrieve x, the top 3 passages become the observation
         Lookup[x]  -> the next sentence containing x in the passages of the last search (no retrieval)
         Finish[x]  -> x is the answer, stop
       用正则表达式在 action 里找 "单词[...]"：
         Search[x]  -> 检索 x，前 3 个段落作为 observation
         Lookup[x]  -> 在上次检索到的段落里找下一句含 x 的句子（不检索）
         Finish[x]  -> x 就是答案，结束
    4. "Thought N / Action N / Observation N" is appended to the prompt, next round
       把 "Thought N / Action N / Observation N" 接到 prompt 后面，进入下一轮
    No Finish after the last round -> "Action N+1: Finish[" is appended and the LLM writes only the answer.
    最后一轮结束还没有 Finish -> 在 prompt 后面加上 "Action N+1: Finish["，让 LLM 只写答案。

Quirks of the official code that are kept on purpose / 有意保留的官方代码特点：
    - A round whose action has no "Word[...]" is skipped: nothing is added to the prompt, but the round counts.
      action 里找不到 "单词[...]" 的一轮被跳过：不往 prompt 里加任何东西，但轮数照样计算。
    - If no round of a question has a valid action, the final check reads an unassigned variable and raises
      an error (UnboundLocalError, a kind of NameError), which stops the whole run, as in the official code.
      The run can then be resubmitted: 05 resumes after the questions already finished.
      如果一道题的所有轮都没有有效的 action，最后的检查会读取一个没有赋值的变量而报错
      （UnboundLocalError，NameError 的一种），整个运行停止，与官方一致。之后可以重新提交：05 会接着已完成的题往下跑。
    - The instruction says Search "returns the first paragraph"; in fact it returns the top 3 passages.
      指令里说 Search "返回第一段"；实际返回的是前 3 个段落。
    - The final forced Finish uses the LLM's default temperature (0.7), not the temperature argument.
      最后强制 Finish 用的是 LLM 的默认温度（0.7），而不是 temperature 参数。
"""

import re

from ..prompts import REACT_INSTRUCTION, format_passages

# The 6 few-shot examples ("webthink_simple6", from the original ReAct HotpotQA notebook),
# copied character for character from the official ReAct_Model.__init__ (lines 277-405).
# 6 个示例（"webthink_simple6"，来自 ReAct 原作者的 HotpotQA notebook），
# 从官方 ReAct_Model.__init__（第 277-405 行）逐字复制。
# Long lines and quote style are kept as in the official file: no reformatting here, and the line-length
# check is switched off for this file in pyproject.toml.
# 长行和引号风格保持官方文件的原样：这里不重新格式化，pyproject.toml 里也对本文件关闭了行长检查。
# fmt: off
REACT_EXAMPLES = {
    "webthink_simple6": [
        {
            "question": 'What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?',
            "steps": [
                {
                    "Thought": 'I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.',
                    "Action": 'Search[Colorado orogeny]',
                    "Observation": 'The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.'
                },
                {
                    "Thought": 'It does not mention the eastern sector. So I need to look up eastern sector.',
                    "Action": 'Lookup[eastern sector]',
                    "Observation": '(Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.',
                },
                {
                    "Thought": 'The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.',
                    "Action": 'Search[High Plains]',
                    "Observation": 'High Plains refers to one of two distinct land regions:',
                },
                {
                    "Thought": 'I need to instead search High Plains (United States).',
                    "Action": 'Search[High Plains (United States)]',
                    "Observation": 'The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).',
                },
                {
                    "Thought": 'High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.',
                    "Action": 'Finish[1,800 to 7,000 ft]',
                }
            ]
        },
        {
            "question": 'Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?',
            "steps": [
                {
                    "Thought": 'The question simplifies to \"The Simpsons\" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.',
                    "Action": 'Search[Milhouse]',
                    "Observation": 'Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.',
                },
                {
                    "Thought": 'The paragraph does not tell who Milhouse is named after, maybe I can look up \"named after\".',
                    "Action": 'Lookup[named after]',
                    "Observation": '(Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous.',
                },
                {
                    "Thought": 'Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.',
                    "Action": 'Finish[Richard Nixon]',
                }
            ]
        },
        {
            "question": 'Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?',
            "steps": [
                {
                    "Thought": 'I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.',
                    "Action": 'Search[Adam Clayton Powell]',
                    "Observation": "Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].",
                },
                {
                    "Thought": 'To find the documentary, I can search Adam Clayton Powell (film).',
                    "Action": 'Search[Adam Clayton Powell (film)]',
                    "Observation": 'Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.\nThe film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.',
                },
                {
                    "Thought": 'Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.',
                    "Action": 'Finish[The Saimaa Gesture]',
                },
            ]

        },
        {
            "question": 'What profession does Nicholas Ray and Elia Kazan have in common?',
            "steps": [
                {
                    "Thought": 'I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.',
                    "Action": 'Search[Nicholas Ray]',
                    "Observation": 'Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 \u2013 June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.',
                },
                {
                    "Thought": 'Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.',
                    "Action": 'Search[Elia Kazan]',
                    "Observation": 'Elia Kazan was an American film and theatre director, producer, screenwriter and actor.',
                },
                {
                    "Thought": 'Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.',
                    "Action": 'Finish[director, screenwriter, actor]',
                },
            ]
        },
        {
            "question": "Which magazine was started first Arthur's Magazine or First for Women?",
            "steps": [
                {
                    "Thought": "I need to search Arthur's Magazine and First for Women, and find which was started first.",
                    "Action": "Search[Arthur's Magazine]",
                    "Observation": "Arthur's Magazine (1844-\u0080\u00931846) was an American literary periodical published in Philadelphia in the 19th century.",
                },
                {
                    "Thought": "Arthur's Magazine was started in 1844. I need to search First for Women next.",
                    "Action": 'Search[First for Women]',
                    "Observation": "First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989.",
                },
                {
                    "Thought": "First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.",
                    "Action": "Finish[Arthur's Magazine]",
                },
            ]
        },
        {
            "question": 'Were Pavel Urysohn and Leonid Levin known for the same type of work?',
            "steps": [
                {
                    "Thought": 'I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.',
                    "Action": 'Search[Pavel Urysohn]',
                    "Observation": 'Pavel Samuilovich Urysohn (February 3, 1898 \u00e2\u0080\u0093 August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.',
                },
                {
                    "Thought": 'Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.',
                    "Action": 'Search[Leonid Levin]',
                    "Observation": 'Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.',
                },
                {
                    "Thought": 'Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.',
                    "Action": 'Finish[yes]',
                },
            ]
        },
    ]
}
# fmt: on


def build_examples_text(examples):
    """
    Turn the examples into prompt text, as the official __init__ does (lines 406-412):
        Question: ...
        Thought 1: ...
        Action 1: ...
        Observation 1: ...
        (empty line after every example)
    把示例拼成 prompt 文字，与官方 __init__（第 406-412 行）相同（格式见上，每个示例后空一行）。
    """
    examples_text = ""
    for example in examples["webthink_simple6"]:
        examples_text += f"Question: {example['question']}\n"
        for step_i, think_step in enumerate(example["steps"]):
            for step_key, step_val in think_step.items():
                examples_text += f"{step_key} {step_i + 1}: {step_val}\n"
        examples_text += "\n"
    return examples_text


class ReAct:
    """
    ReAct deep research agent. / ReAct 深度研究智能体。

    llm:       dra.llm.LLM (or anything with generate(messages, max_new_tokens=..., temperature=...,
               stop_sequences=...) -> text)
    retriever: dra.retrieval.StepRetriever (or anything with search_for_step(query) -> (docs, ranked))
    max_iter:  maximum number of rounds (official default 20)   最多轮数（官方默认 20）
    """

    def __init__(self, llm, retriever, max_iter=20):
        self.llm = llm
        self.retriever = retriever
        self.max_iter = max_iter
        self.instruction = REACT_INSTRUCTION
        self.pattern_action = r"^(\w+)\[(.+?)\]$"
        self.current_step_template = (
            "Thought {iter_num}: {think}\nAction {iter_num}: {action_text}\nObservation {iter_num}: {observation}\n"
        )
        self.examples_text = build_examples_text(REACT_EXAMPLES)
        # Raw LLM outputs, token counts and whether the answer came from the forced Finish, for the current
        # question (ours, for analysis; not in the official code).
        # 当前这道题的 LLM 原始输出、token 数，以及答案是否来自强制 Finish（我们加的，用于分析；官方代码没有）。
        self.llm_outputs = []
        self.usage = None
        self.forced_finish = None

    def _generate(self, messages, stop_sequences, **gen_kwargs):
        """
        Call the LLM and record its output and token counts for this question.
        调用 LLM，并记下这道题的输出和 token 数。
        """
        text = self.llm.generate(messages, stop_sequences=stop_sequences, **gen_kwargs)
        self.llm_outputs.append(text)
        last = getattr(self.llm, "last_usage", None)
        if last:
            self.usage["llm_calls"] += 1
            self.usage["prompt_tokens"] = max(self.usage["prompt_tokens"], last["prompt_tokens"])
            self.usage["output_tokens"] += last["output_tokens"]
        return text

    # ------------------------------------------------------------------
    # Parsing helpers, unchanged from the official code (lines 421-456)
    # 解析用的辅助函数，与官方代码相同（第 421-456 行）
    # ------------------------------------------------------------------

    @staticmethod
    def get_page_obs(page):
        # Not used by the retriever version, kept for completeness. / 检索器版本用不到，为完整保留。
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        sentences = []
        for p in paragraphs:
            sentences += p.split(". ")
        sentences = [s.strip() + "." for s in sentences if s.strip()]
        return " ".join(sentences[:5])

    @staticmethod
    def clean_action(text):
        """
        Cut the text at "Observation N:" (in case the model kept writing).
        在 "Observation N:" 处截断（以防模型继续往下写）。
        """
        pattern = r"Observation \d+:\s*"
        match = re.search(pattern, text)
        return text[: match.start()].strip() if match else text.strip()

    @staticmethod
    def extract_action_text(text):
        """
        First "Word[...]" in the text, e.g. "Search[Leonid Levin]"; None if there is none.
        文字里第一个 "单词[...]"，例如 "Search[Leonid Levin]"；没有就返回 None。
        """
        pattern = r"\w+\[[^\]]*?\]"
        match = re.search(pattern, text)
        return match.group(0) if match else None

    def extract_action_type_entity(self, action_text):
        """
        "Search[Leonid Levin]" -> ("search", "Leonid Levin"); ("", "") if it does not match.
        "Search[Leonid Levin]" -> ("search", "Leonid Levin")；不匹配时返回 ("", "")。
        """
        match = re.match(self.pattern_action, action_text)
        if match:
            action_type = match.group(1).lower()
            action_entity = match.group(2)
        else:
            action_type, action_entity = "", ""
        return action_type, action_entity

    # ------------------------------------------------------------------
    # Actions (official lines 483-541; the Wikipedia web search of the original ReAct is not used)
    # 动作（官方第 483-541 行；原版 ReAct 的维基百科网页搜索不使用）
    # ------------------------------------------------------------------

    def retriever_search(self, search_query):
        """
        Search: retrieve, and keep the text of the passages as the "current page" for Lookup.
        As in the official code, the first line of each "contents" is treated as a title and dropped,
        and the passages are joined without a separator.
        Search：检索，并把段落正文存为 Lookup 用的"当前页面"。与官方一致，每个 "contents" 的第一行
        被当作标题去掉，各段落直接拼接、中间不加分隔。
        """
        search_docs, ranked_list = self.retriever.search_for_step(search_query)
        self.page = ""
        for doc in search_docs:
            content = doc["contents"]
            text = "\n".join(content.split("\n")[1:])
            self.page += text
        self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
        return search_docs, ranked_list

    def construct_lookup_list(self, keyword):
        """
        All sentences of the current page that contain the keyword (case-insensitive).
        当前页面里所有含关键词的句子（不区分大小写）。
        """
        if self.page is None:
            return []
        paragraphs = self.page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        sentences = []
        for p in paragraphs:
            sentences += p.split(". ")
        sentences = [s.strip() + "." for s in sentences if s.strip()]
        return [p for p in sentences if keyword.lower() in p.lower()]

    def get_observation(self, action_text):
        """
        Carry out the action; returns (action_type, action_entity, docs, ranked_list, obs, done).
        执行动作；返回 (动作类型, 动作内容, 段落, 完整排序, observation, 是否结束)。
        """
        done, docs, ranked_list, obs = False, None, None, None

        action_type, action_entity = self.extract_action_type_entity(action_text)
        if action_type == "search":
            docs, ranked_list = self.retriever_search(action_entity)
            # Passages in the same format as the single-retrieval prompt: "Doc 1 (Title: ...) text\n..."
            # 段落格式与单次检索的 prompt 相同："Doc 1 (Title: ...) 正文\n..."
            obs = format_passages(docs)
        elif action_type == "lookup":
            keyword = action_entity
            if self.lookup_keyword != keyword:  # new keyword: start from its first sentence / 新关键词：从第一句开始
                self.lookup_keyword = keyword
                self.lookup_list = self.construct_lookup_list(keyword)
                self.lookup_cnt = 0
            if self.lookup_cnt >= len(self.lookup_list):
                obs = "No more results.\n"
            else:
                obs = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) " + self.lookup_list[self.lookup_cnt]
                self.lookup_cnt += 1
        elif action_type == "finish":
            done = True
        elif action_type == "think":
            obs = "Nice thought."
        else:
            obs = f"Invalid action: {action_text}"

        return action_type, action_entity, docs, ranked_list, obs, done

    # ------------------------------------------------------------------
    # Main loop (official inference, lines 543-623)
    # 主循环（官方 inference，第 543-623 行）
    # ------------------------------------------------------------------

    def inference(self, question, generation_temp=0.7):
        """
        Answer one question. Returns (reasoning_path, pred_answer).
        回答一道题。返回 (推理过程, 答案)。

        reasoning_path: one dict per valid round, as in the official code:
            Search  {"think", "action_type": "search", "search_query", "docs", "ranked_list"}
            Lookup  {"think", "action_type": "lookup", "entity", "observation"}
            Finish  {"think", "prediction"}   (forced Finish: think = "")
        reasoning_path：每个有效轮次一个字典，与官方相同（见上；强制 Finish 时 think 为空字符串）。

        After the call, self.llm_outputs holds every raw LLM output of this question, self.usage the
        number of LLM calls, the longest prompt and the total output tokens, and self.forced_finish
        whether the answer came from the forced Finish.
        调用之后，self.llm_outputs 是这道题每一次 LLM 的原始输出，self.usage 是 LLM 调用次数、
        最长的 prompt 和输出 token 总数，self.forced_finish 表示答案是否来自强制 Finish。
        """
        self.page = None  # passages of the last Search / 上次 Search 得到的段落正文
        self.lookup_keyword = None  # current Lookup keyword / 当前 Lookup 的关键词
        self.lookup_list = None  # sentences containing it / 含该关键词的句子
        self.lookup_cnt = None  # how many of them were shown / 已经给出了几句
        self.llm_outputs = []
        self.usage = {"llm_calls": 0, "prompt_tokens": 0, "output_tokens": 0}
        self.forced_finish = False

        input_prompt = self.instruction + self.examples_text + f"Question: {question}\n"

        reasoning_path = []
        for iter_num in range(1, self.max_iter + 1):
            # 1. Generate "Thought N: ... Action N: ...". The system message is empty, as in the official code
            #    (with no system message at all, Qwen's chat template would insert its default one).
            #    生成 "Thought N: ... Action N: ..."。与官方一致，system 消息为空
            #    （完全不给 system 消息的话，Qwen 的对话模板会自动加上它默认的那一句）。
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": input_prompt + f"Thought {iter_num}:"},
            ]
            thought_action = self._generate(
                messages,
                [
                    f"\n\nObservation {iter_num}:",
                    f"\nObservation {iter_num}:",
                    f"Observation {iter_num}:",
                    f"Observation {iter_num}: ",
                    f"\nObservation {iter_num}: ",
                ],
                temperature=generation_temp,
            )
            # 2. Split into thought and action. If "Action N:" is missing (or appears more than once),
            #    keep the first line as the thought and ask the LLM for the action alone.
            #    切成 thought 和 action。如果没有 "Action N:"（或出现了不止一次），
            #    把第一行当作 thought，再让 LLM 单独写 action。
            try:
                thought, action = thought_action.strip().split(f"Action {iter_num}:")
            except ValueError:
                thought = thought_action.strip().split("\n")[0]
                messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": input_prompt + f"Thought {iter_num}: {thought}\nAction {iter_num}:"},
                ]
                action = self._generate(messages, ["]\n", "]\n\n", " ]\n", " ]\n\n"], temperature=generation_temp)

            # 3. Find "Word[...]" in the action.   在 action 里找 "单词[...]"。
            thought = thought.replace("\n", " ").strip()
            action_text = self.clean_action(action).replace("\n", " ").strip()
            action_text = self.extract_action_text(action_text)

            if action_text:  # a valid action; otherwise this round is skipped / 有效动作；否则这一轮被跳过
                action_type, action_entity, docs, ranked_list, obs, done = self.get_observation(action_text)
                # As in the official code: this replaces the two characters "\" + "n", not real newlines.
                # 与官方一致：这里替换的是 "\" 和 "n" 两个字符，而不是真正的换行。
                obs = obs.replace("\\n", " ").strip() if obs else None

                if action_type == "search":
                    step_entry = {
                        "think": thought,
                        "action_type": action_type,
                        "search_query": action_entity,
                        "docs": docs,
                    }
                    if ranked_list is not None:
                        step_entry["ranked_list"] = ranked_list
                    reasoning_path.append(step_entry)
                elif action_type == "lookup":
                    reasoning_path.append(
                        {"think": thought, "action_type": action_type, "entity": action_entity, "observation": obs}
                    )
                elif action_type == "finish":
                    reasoning_path.append({"think": thought, "prediction": action_entity})
                    pred_answer = action_entity

                if done or (iter_num == self.max_iter):
                    break

                # 4. Append this round to the prompt.   把这一轮接到 prompt 后面。
                input_prompt += self.current_step_template.format(
                    iter_num=iter_num, think=thought, action_text=action_text, observation=obs
                )

        # No Finish: make the LLM write only the answer after "Finish[" (at most 32 tokens, stops at "]").
        # If no round had a valid action, action_type was never assigned and this line raises an error,
        # as in the official code (see the module docstring).
        # 没有 Finish：让 LLM 在 "Finish[" 后面只写答案（最多 32 个 token，写到 "]" 停下）。
        # 如果没有任何一轮有有效动作，action_type 从未赋值，这一行会报错，与官方一致（见文件开头的说明）。
        if action_type != "finish":
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": input_prompt + f"Action {iter_num + 1}: Finish["},
            ]
            output_text = self._generate(messages, ["]", "] ", "]\n", "]\n ", "]\n\n", "]\n\n "], max_new_tokens=32)
            # Drop the last character (the closing "]"), as in the official code.
            # 去掉最后一个字符（即结尾的 "]"），与官方一致。
            pred_answer = output_text[:-1]
            reasoning_path.append({"think": "", "prediction": pred_answer})
            self.forced_finish = True

        return reasoning_path, pred_answer


def reasoning_path_for_output(reasoning_path):
    """
    Copy of reasoning_path for the per-question results file: passage records -> passage ids ("doc_ids"),
    full ranking dropped. Same as the official _reasoning_path_doc_ids_only (run_evalution.py, lines 551-567).
    逐题结果文件用的 reasoning_path 副本：段落记录换成段落 id（"doc_ids"），去掉完整排序。
    与官方 _reasoning_path_doc_ids_only（run_evalution.py 第 551-567 行）相同。
    """
    out = []
    for step in reasoning_path:
        step_copy = dict(step)
        if step_copy.get("docs"):
            step_copy["doc_ids"] = [d["id"] for d in step_copy["docs"]]
            del step_copy["docs"]
        step_copy.pop("ranked_list", None)
        out.append(step_copy)
    return out


def interleave_rankings(step_rankings):
    """
    Merge the rankings of all Search steps of one question into one ranking, as the official
    _interleave_rank_lists_with_dedup (run_evalution.py, lines 570-598): first the rank-1 passage of every
    step, then the rank-2 passage of every step, ...; a passage already taken is skipped.
    把一道题所有 Search 步骤的排序合并成一个排序，与官方 _interleave_rank_lists_with_dedup
    （run_evalution.py 第 570-598 行）相同：先取每一步的第 1 名，再取每一步的第 2 名……；已经取过的段落跳过。

    step_rankings: one [(passage_id, score), ...] per Search step, in step order.
    Returns [(passage_id, score, step_number, rank_in_step), ...] (both numbers start at 1).
    返回 [(段落 id, 分数, 第几步, 在该步的名次), ...]（两个编号都从 1 开始）。
    """
    seen = set()
    out = []
    max_len = max((len(s) for s in step_rankings), default=0)
    for r in range(max_len):
        for step_num, ranking in enumerate(step_rankings, start=1):
            if r >= len(ranking):
                continue
            pid, score = ranking[r]
            if pid in seen:
                continue
            seen.add(pid)
            out.append((pid, score, step_num, r + 1))
    return out
