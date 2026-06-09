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

        # Error-Attribution prompt
        ( 
            "evidence-annotation-check",
            (
                "You are an expert evaluator for memory-based question answering systems. "
                "## Your Task\n"
                "Determine whether the source evidence itself contains the key information "
                "needed to answer the question correctly. "
                "Do NOT consider any memory system or retrieval results — only look at the raw source evidence.\n\n"
                "## Question\n$question\n\n"
                "## Golden Answers\n$golden_answers\n\n"
                "## Source Evidence (the original raw message)\n$source_evidence\n\n"
                "## Judgment Criteria\n"
                "- If the source evidence contains (explicitly or implicitly) the information needed to derive "
                "the golden answer, set is_relevant=true.\n"
                "- If the source evidence does NOT contain such information at all, it is likely a dataset "
                "annotation error. Set is_relevant=false.\n"
                "- Consider semantic equivalence and implicit mentions, not just exact text matching.\n\n"
                "Respond in the following JSON format:\n"
                "```json\n"
                '{"explanation": "<your reasoning>", "is_relevant": <true or false>}\n'
                "```"
            ),
        ),
        (
            "memory-construction-error-check",
            (
                "You are an expert evaluator for memory-based question answering systems. "
                "You are performing Step 1 of a 3-step error attribution pipeline:\n\n"
                "  Step 1 (YOUR TASK) - Construction Error Check: Did the memory system fail to preserve "
                "the key information from the source evidence during the memory construction process?\n"
                "  Specifically, your task is to determine whether the essential information from a source evidence "
                "  is present within a set of retrieved memory units.\n\n"
                "  Step 2 - Retrieval Error Check: Was the preserved memory successfully retrieved?\n"
                "  Step 3 - Response Error Check: Was the retrieved memory correctly used to answer?\n\n"
                "## Question\n$question\n\n"
                "## Golden Answers\n$golden_answers\n\n"
                "## Source Evidence (the original raw message, confirmed to contain relevant information)\n"
                "$source_evidence\n\n"
                "## Retrieved Memory Units (what the memory system stored and can retrieve for this evidence)\n"
                "$retrieved_memory_units\n\n"
                "## Your Task\n"
                "Given that the source evidence DOES contain the relevant information, determine whether "
                "that information was correctly preserved during memory construction.\n"
                "Check if the key information from the source evidence is present in the retrieved memory units "
                "(consider semantic equivalence, not just exact text matching).\n"
                "If the information is missing from memory despite being in the source evidence, "
                "that is a construction error.\n\n"
                "Please provide:\n"
                "1. A brief explanation identifying what specific information from the evidence should be "
                "preserved, and whether it appears in the memory units.\n"
                "2. A final judgment: whether the essential information is present (true) or missing (false).\n\n"
                "Respond in the following JSON format:\n"
                "```json\n"
                '{"explanation": "<your reasoning>", "is_present": <true or false>}\n'
                "```"
            ),
        ),
        (
            "root-cause-analysis",
            (
                "You are an expert at diagnosing memory construction errors in AI systems.\n\n"
                "## Original Input\n"
                "**Source Evidence:**\n$evidence\n\n"
                "**Raw Message (containing the evidence):**\n$raw_message\n\n"
                "## Processing Pipeline\n"
                "**Processing Records (how the raw message was transformed into memory):**\n$processing_records\n\n"
                "## The Problem\n"
                "During memory construction, some essential information from the source evidence was lost or incorrectly transformed.\n\n"
                "## Your Task\n"
                "Analyze the processing pipeline and identify the ROOT CAUSE:\n\n"
                "1. **What went wrong?** Which step in the processing pipeline lost the information?\n"
                "2. **Why did it happen?** What aspect of the processing strategy (prompt, model, extraction method) caused this?\n"
                "3. **How to fix it?** What changes to the processing pipeline would prevent this?\n\n"
                "Provide a concise analysis (2-4 sentences) focusing on the PROCESSING PIPELINE ITSELF."
            ),
        ),
        (
            "retrieval-error-check",
            (
                "You are an expert evaluator for memory-based question answering systems. "
                "Your task is to determine whether the retrieval results sufficiently cover the key contents "
                "of the source evidences needed to answer the question.\n\n"
                "## Question\n$question\n\n"
                "## Golden Answers\n$golden_answers\n\n"
                "## Source Evidences\n$source_evidences\n\n"
                "## Retrieval Results\n$retrieval_results\n\n"
                "## Task\n"
                "Analyze whether the retrieval results contain sufficient information from the source evidences "
                "to answer the question correctly. The retrieval results should cover the key contents of all "
                "source evidences. Consider semantic equivalence, not just exact text matching.\n\n"
                "Please provide:\n"
                "1. A brief explanation of your reasoning.\n"
                "2. A final judgment: whether the retrieval sufficiently covers the source evidences (true) or not (false).\n\n"
                "Respond in the following JSON format:\n"
                "```json\n"
                '{"explanation": "<your reasoning>", "is_sufficient": <true or false>}\n'
                "```"
            ),
        ),
        (
            "annotation-error-analysis",
            (
                "You are an expert evaluator for memory-based QA systems.\n"
                "Below are $num_cases sampled cases where QA instances failed due to "
                "annotation errors — the source evidence was deemed irrelevant to the question.\n\n"
                "$cases_text\n\n"
                "## Task\n"
                "Based on the cases above, identify the common patterns and systematic root causes "
                "behind these annotation errors. What types of annotation mistakes recur? "
                "(e.g. adversarial questions with no valid evidence, topically related but "
                "answer-absent evidence, labeling errors, etc.)\n\n"
                "Respond in the following JSON format:\n"
                "```json\n"
                '{"summary": "<overall pattern analysis>", "suggestions": "<actionable suggestions to improve annotation quality>"}\n'
                "```"
            ),
        ),
        (
            "construction-error-analysis",
            (
                "You are an expert evaluator for memory-based QA systems.\n"
                "Below are $num_cases sampled cases where QA instances failed because key "
                "information was not preserved during memory construction.\n\n"
                "$cases_text\n\n"
                "## Task\n"
                "Based on the cases above, identify the common patterns and systematic root causes "
                "behind these construction errors. What types of information are consistently lost? "
                "(e.g. multimodal metadata like image captions ignored, fine-grained entity details "
                "dropped during extraction, aggressive summarization, etc.)\n\n"
                "Respond in the following JSON format:\n"
                "```json\n"
                '{"summary": "<overall pattern analysis>", "suggestions": "<actionable suggestions to improve the memory construction pipeline>"}\n'
                "```"
            ),
        ),
        (
            "retrieval-error-analysis",
            (
                "You are an expert evaluator for memory-based QA systems.\n"
                "Below are $num_cases sampled cases where QA instances failed because the "
                "retrieval step did not surface sufficient information.\n\n"
                "$cases_text\n\n"
                "## Task\n"
                "Based on the cases above, identify the common patterns and systematic root causes "
                "behind these retrieval errors. What types of queries or memory structures "
                "consistently cause retrieval to fail? "
                "(e.g. semantic mismatch, multi-hop questions, ranking issues, vocabulary gaps, etc.)\n\n"
                "Respond in the following JSON format:\n"
                "```json\n"
                '{"summary": "<overall pattern analysis>", "suggestions": "<actionable suggestions to improve the retrieval strategy>"}\n'
                "```"
            ),
        ),
        (
            "response-error-analysis",
            (
                "You are an expert evaluator for memory-based QA systems.\n"
                "Below are $num_cases sampled cases where QA instances failed even though "
                "the retrieval results contained sufficient information.\n\n"
                "$cases_text\n\n"
                "## Task\n"
                "Based on the cases above, identify the common patterns and systematic root causes "
                "behind these response errors. Why does the model fail to produce correct answers "
                "despite having the right information? "
                "(e.g. reasoning failure, hallucination overriding facts, format mismatch, "
                "failure to synthesize across multiple memory units, etc.)\n\n"
                "Respond in the following JSON format:\n"
                "```json\n"
                '{"summary": "<overall pattern analysis>", "suggestions": "<actionable suggestions to improve response generation>"}\n'
                "```"
            ),
        ),
        (
            "trace-annotation-check",
            (
                "You are an expert evaluator for trace-based memory question answering systems. "
                "You are performing the annotation validation stage before construction/retrieval/response diagnosis.\n\n"
                "## Question\n$question\n\n"
                "## Golden Answer\n$golden_answer\n\n"
                "## Source Evidences (from trace metadata)\n$source_evidences\n\n"
                "## Task\n"
                "Determine whether the provided source evidences are actually relevant to answering the question. "
                "If evidences are topically related but do not contain answer-critical information, treat them as annotation mismatch.\n"
                "- If annotation is wrong, set is_annotation_error=true and list bad_source_evidences.\n"
                "- If annotation is acceptable, set is_annotation_error=false and bad_source_evidences=[].\n"
                "- Consider semantic equivalence, temporal wording variants, and implicit mentions.\n"
                "- Note: The source evidence contains the **timestamp** information of the message\n\n"
                "Respond in strict JSON format:\n"
                "```json\n"
                '{"explanation": "<brief reasoning>", "is_annotation_error": <true or false>, "bad_source_evidences": ["<evidence full name>"]}\n'
                "```"
            ),
        ),
        (
            "trace-construction-check",
            (
                "You are an expert evaluator for trace-based memory question answering systems. "
                "You are diagnosing construction-stage failures.\n\n"
                "## Question\n$question\n\n"
                "## Golden Answer\n$golden_answer\n\n"
                "## Source Evidence Full Name\n$source_evidence_full_name\n\n"
                "## Candidate Operation IDs\n$candidate_op_ids\n\n"
                "## Construction Subgraph (trace)\n$construction_subgraph\n\n"
                "## Task\n"
                "Based on this construction subgraph, decide whether memory construction failed to preserve information needed for the golden answer. "
                "If failed, return the most likely op_id from candidate list; otherwise return null.\n"
                "Focus on extraction, transformation, and memory update operations in the trace.\n"
                "Note: When **golden answer** is related to **time**,"
                "the effectiveness of fact extraction will determine whether the time-related information details are properly retained.\n."
                "Respond in strict JSON format:\n"
                "```json\n"
                '{"explanation": "<brief reasoning>", "is_construction_error": <true or false>, "op_id": "<op-id>"}\n'
                "```"
                "If no op_id is appropriate, set \"op_id\": null."
            ),
        ),
        (
            "trace-retrieval-check",
            (
                "You are an expert evaluator for trace-based memory question answering systems. "
                "You are diagnosing retrieval-stage failures.\n\n"
                "## Question\n$question\n\n"
                "## Golden Answer\n$golden_answer\n\n"
                "## Source Evidences\n$source_evidences\n\n"
                "## Candidate Operation IDs\n$candidate_op_ids\n\n"
                "## Retrieval Subgraph (trace)\n$retrieval_subgraph\n\n"
                "## Task\n"
                "Judge whether retrieval failed to surface sufficient information for this question. "
                "If failed, provide the most likely retrieval-related op_id from candidates; otherwise return null.\n"
                "Consider ranking, filtering, query rewriting, and memory-hit quality in trace operations.\n\n"
                "Respond in strict JSON format:\n"
                "```json\n"
                '{"explanation": "<brief reasoning>", "is_retrieval_error": <true or false>, "op_id": "<op-id>"}\n'
                "```"
                "If no op_id is appropriate, set \"op_id\": null."
            ),
        ),
        (
            "trace-response-check",
            (
                "You are an expert evaluator for trace-based memory question answering systems. "
                "Construction and retrieval have already been checked and considered non-blocking. "
                "Now diagnose response-stage failure.\n\n"
                "## Question\n$question\n\n"
                "## Golden Answer\n$golden_answer\n\n"
                "## Prediction\n$prediction\n\n"
                "## Candidate Operation IDs\n$candidate_op_ids\n\n"
                "## Response Subgraph (trace)\n$response_subgraph\n\n"
                "## Task\n"
                "Identify the most likely response-stage faulty operation (or closest causal op) that leads to the wrong final answer.\n"
                "If no clear op can be identified, return null.\n\n"
                "Respond in strict JSON format:\n"
                "```json\n"
                '{"explanation": "<brief reasoning>", "op_id": "<op-id>"}\n'
                "```"
                "If no op_id is appropriate, set \"op_id\": null."
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