import re
from typing import List
from examples.utils.sql_eval import sqlTester
from examples.utils.math_normalizer import normalize_answer
from examples.utils.grader import grade_answer


def create_sql_reward_fn(sqltester: sqlTester):
    """
    Create a SQL reward function with a specific sqlTester instance.
    
    Args:
        sqltester: sqlTester instance for SQL evaluation
    
    Returns:
        Reward function with signature (question, answer1, answer2, standard_answer) -> List[float]
    """
    def sql_reward_fn(question: str, answer1: str, answer2: str, standard_answer: str) -> List[float]:
        """
        Evaluate SQL generation quality by comparing execution results.
        
        Args:
            question: The question containing database ID
            answer1: First SQL answer to compare
            answer2: Second SQL answer to compare
            standard_answer: Standard answer for evaluation
        
        Returns:
            [1.0, 0.0] if answer1 correct and answer2 wrong
            [0.0, 1.0] if answer2 correct and answer1 wrong  
            [0.5, 0.5] if both correct or both wrong
        """
        sql1 = sqltester.extract_sql_from_answer(answer1)
        sql2 = sqltester.extract_sql_from_answer(answer2)
        
        db_id = sqltester.extract_dbid(question)
        result1 = sqltester.evaluate_execution(db_id, sql1, standard_answer)
        result2 = sqltester.evaluate_execution(db_id, sql2, standard_answer)
        
        if result1 and not result2:
            return [1.0, 0.0]
        elif not result1 and result2:
            return [0.0, 1.0]
        else:
            return [0.5, 0.5]
    
    return sql_reward_fn


def _extract_number(text: str, using_boxed: bool = True) -> str:
    """
    Extract the final answer from model response.
    
    Args:
        text: Model response text
        using_boxed: Whether to extract from boxed format
    
    Returns:
        Extracted answer string
    """
    if not using_boxed:
        # Try to extract answer after #### marker (GSM8K format)
        if "####" in text:
            answer_part = text.split('####')[-1].strip()
            # Extract any number or expression
            numbers = re.findall(r'[-+]?\d+', answer_part)
            if numbers:
                return numbers[-1]
        # If no #### marker, try to find the last number in the text
        numbers = re.findall(r'[-+]?\d+', text)
        if numbers:
            return numbers[-1]
        return text.strip()
    
    elif "boxed{" in text:
        # Extract the boxed number/expression
        numbers = re.findall(r'boxed\{(.*?)\}', text)
        if numbers:
            # Extract numeric value from boxed content
            boxed_content = numbers[-1]
            # Try to extract number from boxed content
            numeric_match = re.search(r'[-+]?\d+', boxed_content)
            if numeric_match:
                return numeric_match.group()
            return boxed_content.strip()
    
    # Fallback: try to find the last number in the text
    numbers = re.findall(r'[-+]?\d+', text)
    if numbers:
        return numbers[-1]
    
    return text.strip()


def math_equal_reward_fn(question: str, answer1: str, answer2: str, standard_answer: str) -> List[float]:
    """
    Reward function for AIME problems.
    Extracts answers from model responses and compares with standard answer.
    
    Args:
        question: The question (not used but kept for interface consistency)
        answer1: First answer to compare
        answer2: Second answer to compare
        standard_answer: Standard answer for evaluation
    
    Returns:
        [1, 0] if answer1 is better
        [0, 1] if answer2 is better
        [0.5, 0.5] if both are equal
    """
    is_eval_mode = (answer2 == standard_answer)
    
    # Extract answers
    answer1_extracted = _extract_number(answer1)
    if is_eval_mode:
        answer2_extracted = _extract_number(answer2, using_boxed=False)
    else:
        answer2_extracted = _extract_number(answer2, using_boxed=True)
    standard_answer_extracted = _extract_number(standard_answer, using_boxed=False)
    
    if is_eval_mode:
        # Grade answers using exact match
        answer_1_correct = (answer1_extracted == standard_answer_extracted)
        answer_2_correct = (answer2_extracted == standard_answer_extracted)
        
        if answer_1_correct and not answer_2_correct:
            return [1, 0]
        elif not answer_1_correct and answer_2_correct:
            return [0, 1]
        else:
            return [0.5, 0.5]
    else:
        # Compare numerical distance
        try:
            diff1 = abs(float(answer1_extracted) - float(standard_answer_extracted))
            diff2 = abs(float(answer2_extracted) - float(standard_answer_extracted))
            if diff1 < diff2:
                return [1, 0]
            elif diff1 > diff2:
                return [0, 1]
            else:
                # diff1 == diff2: both answers are equally good (tie)
                return [0.5, 0.5]
        except (ValueError, TypeError):
            return [0, 1]


def _extract_latex_number(text: str, using_boxed: bool = True) -> str:
    """
    Extract number from LaTeX formatted text.
    
    Args:
        text: Model response text with LaTeX formatting
        using_boxed: Whether to extract from boxed format
    
    Returns:
        Extracted answer string
    """
    if not using_boxed:
        # Try to extract answer after #### marker (GSM8K format)
        answer_part = text.split('####')[-1].strip()
        if answer_part:
            return answer_part
    
    if "boxed{" in text:
        # Extract the boxed number in the text
        numbers = re.findall(r'\\boxed\{(.*)\}', text)
        if numbers:
            return numbers[-1]
    
    return text


def math_latex_reward_fn(question: str, answer1: str, answer2: str, standard_answer: str) -> List[float]:
    """
    Reward function for math problems with LaTeX formatting.
    Uses grader to evaluate LaTeX expressions.
    
    Args:
        question: The question (not used but kept for interface consistency)
        answer1: First answer to compare
        answer2: Second answer to compare
        standard_answer: Standard answer for evaluation
    
    Returns:
        [1, 0] if answer1 is correct and answer2 is wrong
        [0, 1] if answer2 is correct and answer1 is wrong
        [0.5, 0.5] if both correct or both wrong
    """
    answer1_extracted = _extract_latex_number(answer1)
    if answer2 == standard_answer:
        answer2_extracted = _extract_latex_number(answer2, using_boxed=False)
    else:
        answer2_extracted = _extract_latex_number(answer2, using_boxed=True)
    standard_answer_extracted = _extract_latex_number(standard_answer, using_boxed=False)

    answer_1_correct = grade_answer(answer1_extracted, standard_answer_extracted)
    answer_2_correct = grade_answer(answer2_extracted, standard_answer_extracted)

    if answer_1_correct and not answer_2_correct:
        return [1, 0]
    elif not answer_1_correct and answer_2_correct:
        return [0, 1]
    else:
        return [0.5, 0.5]


def create_mind2web_reward_fn(tuner_model_batch_fn):
    comparison_prompt_mind2web = """You are an expert in web navigation and user interface interaction.

Given this web navigation task:
{question}

Compare these two proposed actions and determine which one is MORE CORRECT:

Action A: {answer1}
Action B: {answer2}

Evaluation criteria (in order of importance):
1. Task relevance - Does this action directly help achieve the stated goal?
2. UI logic - Is this a logical next step given the current page state?
3. Element availability - Does the target element actually exist on the page?
4. Efficiency - Is this the most direct path to accomplish the task?

Think step by step, then respond with exactly ONE of these options:
- "Action A is more correct" 
- "Action B is more correct"
- "Both are equally correct or equally wrong"

Your response must start with one of these exact phrases."""

    single_judge_prompt_mind2web = """You are an expert in web navigation and user interface interaction evaluation.

Your task is to determine if a candidate answer is correct for a given web navigation task.
You DO NOT have access to a ground truth answer, so you must judge strictly based on the provided web context (HTML), the user's task goal, and the interaction history.

Context and Task:
{question}

Proposed Action to Evaluate:
{candidate_answer}

Evaluation Steps:
1. **Goal Analysis**: What is the user trying to achieve?
2. **State Analysis**: Based on previous actions, where are we in the flow?
3. **Element Verification**: Does the element selected in the proposed action exist in the HTML? Is it the correct element to interact with?
4. **Action Validity**: Is the action (CLICK, TYPE, SELECT) appropriate for this element and goal?

Judgment Criteria:
- **CORRECT**: The action is the logical, necessary, and correct next step to advance the task.
- **INCORRECT**: The action is irrelevant, interacts with the wrong element, uses the wrong action type, or hinders the task.

Respond with exactly ONE of the following lines, followed by your reasoning:
- "Judgment: CORRECT"
- "Judgment: INCORRECT"

"""

    def mind2web_reward_fn(question: str, answer1: str, answer2: str, standard_answer: str) -> List[float]:
        if standard_answer == answer2 == '':
            judge_results = tuner_model_batch_fn(
                prompts=[single_judge_prompt_mind2web.format(question=question, candidate_answer=answer1)],
                system_prompts=None,
                temp=None,
                max_toks=None
            )
            judge_result = judge_results[0]
            if "Judgment: CORRECT" in judge_result:
                return [1, 0]
            else:
                return [0, 1]
        if answer1 and answer2 and not standard_answer:
            judge_results = tuner_model_batch_fn(
                prompts=[comparison_prompt_mind2web.format(question=question, answer1=answer1, answer2=answer2)],
                system_prompts=None,
                temp=None,
                max_toks=None
            )
            judge_result = judge_results[0]
            if "Action A is more correct" in judge_result:
                return [1.0, 0.0]
            elif "Action B is more correct" in judge_result:
                return [0.0, 1.0]
            else:
                return [0.5, 0.5]
        if answer1 and answer2 and standard_answer:
            answer1 = answer1.strip().lower()
            answer2 = answer2.strip().lower()
            ANSWER_RE = re.compile(r"answer:\s*([a-z])\.", re.IGNORECASE)
            ACTION_RE = re.compile(r"action:\s*(click|select|type)", re.IGNORECASE)
            VALUE_RE = re.compile(r"value:\s*([^\n\r]*)", re.IGNORECASE)
            def extract_answer(text: str):
                m_ans = ANSWER_RE.search(text)
                m_act = ACTION_RE.search(text)
                m_val = VALUE_RE.search(text)
                return {
                    "letter": m_ans.group(1) if m_ans else None,
                    "action": (m_act.group(1) if m_act else "").strip(),
                    "value": (m_val.group(1) if m_val else "").strip(),
                }
            answer1_dict = extract_answer(answer1)
            answer2_dict = extract_answer(answer2)
            for key in ["letter", "action", "value"]:
                if answer1_dict[key] != answer2_dict[key]:
                    return [0, 1]
            return [0.5, 0.5]
        else:
            return [0.5, 0.5]
    return mind2web_reward_fn