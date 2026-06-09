from collections import OrderedDict
import string


PROMPT_COLLECTIONS = OrderedDict[str, str](
    [
        (
            "default-question-answering",
            (
                "Question: $question\nPlease answer the question based on the following memories:\n$context"
            ), 
        ),

        # See https://arxiv.org/pdf/2601.06966 and https://github.com/AvatarMemory/RealMemBench/blob/67afd0891d603adcc4458ff0449df306ef296b7a/eval/run_generation.py#L31. 
        (
            "realmem-question-answering",
            (
                "You are a personal AI assistant that helps the user with some long-term tasks. "
                "Please respond to the user's latest message based on the reference memories.\n\n"
                "Memories:\n$context\n\n"
                "Query: $question\n\n"
                "Response:"
            ), 
        ), 

        # See https://arxiv.org/abs/2305.12421. 
        (
            "default-exact-match",
            (
                "Here is a question, a list of golden answers, an AI-generated answer. "
                "Can you judge whether the AI-generated answer is correct according to the question and golden answers?"
                "\nQuestion: $question\nGolden Answers: $golden_answers\nAI-generated answer: $prediction"
                "\nSimply answer Yes or No." 
            ),
        ),

        # See https://arxiv.org/abs/2504.03160. 
        (
            "exact-match-zheng-2025", 
            (
                "You will be given a question and its ground truth answer list where each item can be a ground truth answer. "
                "Provided a pred answer, you need to judge if the pred answer correctly answers the question based on the ground truth answer list.\n"
                "You should first give your rationale for the judgement, and then give your judgement result (i.e., Yes or No).\n\n"
                "Here is the criteria for the judgement:\n"
                "1. The pred answer doesn't need to be exactly the same as any of the ground truth answers, but should be semantically same for the question.\n"
                "2. Each item in the ground truth answer list can be viewed as a ground truth answer for the question, " 
                "and the pred_answer should be semantically same to at least one of them.\n\n"
                "question: $question\nground truth answers: $golden_answers\npred_answer: $prediction\n\n"
                "After giving your rationale, you should provide your final answer in the format \\boxed{YOUR_ANSWER}."
            ), 
        ), 

        # See https://arxiv.org/abs/2410.10813 and https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py. 
        (
            "longmemeval-single-session-user",
            (
                "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. "
                "Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, " 
                "you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\n" 
                "Question: $question\n\nCorrect Answer: $golden_answers\n\nModel Response: $prediction\n\n" 
                "Is the model response correct? Answer yes or no only."
            ),
        ), 
        (
            "longmemeval-single-session-assistant", 
            (
                "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. "
                "Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, " 
                "you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\n" 
                "Question: $question\n\nCorrect Answer: $golden_answers\n\nModel Response: $prediction\n\n" 
                "Is the model response correct? Answer yes or no only."
            ), 
        ),
        (
            "longmemeval-multi-session",
            (
                "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. "
                "Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, " 
                "you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\n" 
                "Question: $question\n\nCorrect Answer: $golden_answers\n\nModel Response: $prediction\n\n" 
                "Is the model response correct? Answer yes or no only."
            ),
        ),
        (
            "longmemeval-temporal-reasoning",
            (
                "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. "
                "Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, "
                "you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. " 
                "In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., " 
                "and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\n" 
                "Question: $question\n\nCorrect Answer: $golden_answers\n\nModel Response: $prediction\n\n" 
                "Is the model response correct? Answer yes or no only."
            ),
        ),
        (
            "longmemeval-knowledge-update",
            (
                "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. " 
                "Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct " 
                "as long as the updated answer is the required answer.\n\n" 
                "Question: $question\n\nCorrect Answer: $golden_answers\n\nModel Response: $prediction\n\n" 
                "Is the model response correct? Answer yes or no only."
            ),
        ),
        (
            "longmemeval-single-session-preference",
            (
                "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. " 
                "Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\n" 
                "Question: $question\n\nRubric: $golden_answers\n\nModel Response: $prediction\n\n" 
                "Is the model response correct? Answer yes or no only."
            ),
        ),
        (
            "longmemeval-abstention",
            (
                "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. " 
                "The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\n" 
                "Question: $question\n\nExplanation: $golden_answers\n\nModel Response: $prediction\n\n" 
                "Does the model correctly identify the question as unanswerable? Answer yes or no only."
            ),
        ),

        # See https://arxiv.org/abs/2504.19413. 
        (
            "locomo-judge",
            (
                "Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data: "
                "(1) a question (posed by one user to another user), "
                "(2) a 'gold' (ground truth) answer, "
                "(3) a generated answer "
                "which you will score as CORRECT/WRONG.\n\n"
                "The point of the question is to ask about something one user should know about the other user based on their prior conversations. "
                "The gold answer will usually be a concise and short answer that includes the referenced topic, for example:\n"
                "Question: Do you remember what I got the last time I went to Hawaii?\n"
                "Gold answer: A shell necklace\n"
                "The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.\n\n"
                "For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like 'last Tuesday' or 'next month'), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., 'May 7th' vs '7 May'), consider it CORRECT if it's the same date.\n\n"
                "Now it's time for the real question:\n"
                "Question: $question\n"
                "Gold answer: $golden_answers\n"
                "Generated answer: $prediction\n\n"
                "First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. "
                "Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.\n\n"
                "Just return the label CORRECT or WRONG in a json format with the key as 'label'."
            ),
        ),

        # See https://arxiv.org/pdf/2601.06966. 
        (
            "realmem-lm-score",
            (
                "Your task is to evaluate whether the [candidate answer] correctly uses user-relevant information, "
                "guided by the [reference answer].\n\n"
                "You will be given three pieces of information:\n"
                "1. The user's current query\n"
                "2. A reference answer that represents the ideal response based on relevant user memory\n"
                "3. The candidate answer to be evaluated\n\n"
                "Scoring rules:\n"
                "- Score 1: the candidate answer correctly captures the key user-specific facts, constraints, or preferences "
                "reflected in the reference answer\n"
                "- Score 0: the candidate answer conflicts with, ignores, or fails to incorporate the key user-specific "
                "information present in the reference answer\n\n"
                "Do NOT penalize for tone, style, or minor phrasing differences. "
                "The candidate answer does not need to be identical to the reference answer.\n\n"
                "Question: $question\n"
                "Reference Answer: $golden_answers\n"
                "Candidate Answer: $prediction\n\n"
                "Output your result as JSON with keys 'score' (0 or 1) and 'reason' (one sentence explanation)."
            ), 
        ), 
    ]
)


def register_prompt(
    name: str, 
    template: str, 
    exists_ok: bool = False
) -> None:
    """Register a new prompt template to the global prompt collections.

    Args:
        name (`str`):
            The name of the prompt template.
        template (`str`):
            A string that follows `string.Template` syntax (e.g., uses ``$variable``).
        exists_ok (`bool`, defaults to `False`):
            If it is enabled and the prompt name already exists, it will overwrite the existing prompt.
    """
    if name in PROMPT_COLLECTIONS and not exists_ok:
        raise ValueError(
            f"The prompt name '{name}' already exists. "
            "If you want to overwrite it, set `exists_ok=True`."
        )
    t = string.Template(template)
    if not t.is_valid():
        raise ValueError(
            "The provided template is not a valid template. "
            f"Below is the content of the template:\n{template}"
        )
    PROMPT_COLLECTIONS[name] = template


def get_prompt(name: str) -> string.Template:
    """Get the prompt template by its name.
    
    Args:
        name (`str`): 
            The name of the prompt template.

    Returns:
        `string.Template`: 
            A prompt template.
    """
    prompt = PROMPT_COLLECTIONS.get(name, None)
    if prompt is not None:
        template = string.Template(prompt)
        if not template.is_valid():
            raise ValueError(
                f"The prompt template '{name}' is not valid. "
                f"Below is the content of the prompt template:\n{prompt}"
            )
        return template
    raise ValueError(
        f"The prompt name {name} is not valid. Please choose from the following names: "
        f"{list(PROMPT_COLLECTIONS.keys())}."
    )

