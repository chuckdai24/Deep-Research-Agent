"""
Prompt templates and the passage format.
prompt 模板与检索段落的拼接格式。

Copied unchanged from the official repository (external/total-recall-qa/), so that the reproduction uses
exactly the prompts of the paper. Shared by all methods (single retrieval, ReAct, our own method).
To try a different prompt in Phase 2, add a new template here instead of editing these ones.
从官方仓库原样照搬，保证复现时用的 prompt 和论文完全一致。供各个方法（单次检索、ReAct、我们自己的方法）共用。
Phase 2 想尝试别的 prompt 时，在这里新增模板，不要改这几个官方版本。
"""

# Official system prompt for single retrieval: the rules given to the model.
# Source: c5_task_evaluation/prompts/prompt_templetes.py, line 44 (identical).
# 官方单次检索的 system prompt，即交给模型的规则。出处：prompt_templetes.py 第 44 行（逐字一致）。
SYSTEM_PROMPT_SINGLE_RETRIEVAL = (
    "You are tasked to answer the given question. To do this, first you MUST conduct "
    "reasoning inside <think> and </think>. Then you MUST provide the answer inside "
    "<answer> and </answer>. The retrieved information is inserted into <information> "
    "</information>. Provide only SHORT form answers, NOT complete sentence, without "
    "any additional text or explanation."
)

# User message: {documents} and {question} are placeholders, filled in by str.format().
# Source: c5_task_evaluation/methods/retrieval_augmented_models.py, line 114 (official name: {user_query}).
# 用户消息模板：{documents}、{question} 是占位符，由 str.format() 填入。
# 出处：retrieval_augmented_models.py 第 114 行（官方的占位符名叫 {user_query}）。
USER_PROMPT_TEMPLATE = "<information>{documents}</information>\n\nQuestion: {question}"


def format_passages(passages):
    """
    Join retrieved passages into one text, one per line:
        Doc 1 (Title: xxx) text
        Doc 2 (Title: yyy) text
    把检索到的段落拼成一段文字，每段一行（格式见上）。

    passages: list of passage records as returned by the retriever / data.load_passages,
    e.g. {"id": "15641-0001", "contents": "Joseph Stalin\\nJoseph Vissarionovich Stalin ..."}.
    passages：检索器 / data.load_passages 返回的段落记录列表（字典）。

    Same logic as the official passages2string (utils/general_utils.py, lines 20-31):
    if a record has a separate "title" field it is used; otherwise (our corpus) the first line of
    "contents" is the title and the remaining lines are the text.
    与官方 passages2string（utils/general_utils.py 第 20-31 行）逻辑相同：
    记录里有单独的 "title" 字段就用它；否则（我们的语料就是这种）"contents" 的第一行是标题，其余是正文。
    """
    format_reference = ""
    for idx, doc_item in enumerate(passages):
        if "title" in doc_item:
            text = doc_item["contents"]
            title = doc_item["title"]
        else:
            content = doc_item["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx + 1} (Title: {title}) {text}\n"
    return format_reference


# Official ReAct instruction, followed in the prompt by the 6 examples (methods/react.py) and the question.
# Source: c5_task_evaluation/prompts/prompt_templetes.py, lines 89-94 (identical, including the final \n).
# 官方 ReAct 指令，prompt 里后面接 6 个示例（methods/react.py）和问题。
# 出处：prompt_templetes.py 第 89-94 行（逐字一致，包括结尾的 \n）。
REACT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps.
Thought can reason about the current situation, and Action can be three types:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.\n"""
