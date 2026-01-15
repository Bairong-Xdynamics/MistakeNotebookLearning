"""
Evaluator for comparing model responses using user-defined reward functions.

Computes win rates and batch-level metrics for prompt optimization.
"""

from typing import List, Dict, Any, Callable, Tuple
from .llm_client import LLMClient


class Evaluator:
    """
    Evaluates model responses using a user-defined reward function.
    
    The reward function should take (question, answer1, answer2, standard_answer)
    and return:
    - [1, 0] if answer1 is better
    - [0, 1] if answer2 is better
    - [0.5, 0.5] if they are equally good (tie)
    """
    
    def __init__(
        self,
        reward_fn: Callable[[str, str, str, str], List[int]],
        llm_client: LLMClient,
    ):
        """
        Initialize the evaluator.
        
        Args:
            reward_fn: User-defined reward function
                       Signature: (question, answer1, answer2, standard_answer) -> [1,0] or [0,1]
            llm_client: LLM client for generating responses
        """
        self.reward_fn = reward_fn
        self.llm_client = llm_client
    
    def evaluate_single(
        self,
        question: str,
        answer1: str,
        answer2: str,
        standard_answer: str,
    ) -> float:
        """
        Evaluate two answers for a single question.
        
        Args:
            question: The question
            answer1: First answer
            answer2: Second answer
            standard_answer: Ground truth answer
            
        Returns:
            1.0 if answer1 wins, 0.0 if answer2 wins, 0.5 if tie
        """
        result = self.reward_fn(question, answer1, answer2, standard_answer)
        
        if not isinstance(result, list) or len(result) != 2:
            raise ValueError(
                f"Reward function must return [1,0], [0,1], or [0.5,0.5], got {result}"
            )
        
        if result[0] == 1 and result[1] == 0:
            return 1.0  # answer1 wins
        elif result[0] == 0 and result[1] == 1:
            return 0.0  # answer2 wins
        elif result[0] == 0.5 and result[1] == 0.5:
            return 0.5  # tie
        else:
            raise ValueError(
                f"Reward function must return [1,0], [0,1], or [0.5,0.5], got {result}"
            )
    
    def evaluate_batch(
        self,
        questions: List[str],
        answers1: List[str],
        answers2: List[str],
        standard_answers: List[str],
    ) -> Tuple[Dict[str, float], List[float]]:
        """
        Evaluate a batch of question-answer pairs.
        
        Args:
            questions: List of questions
            answers1: List of first answers
            answers2: List of second answers
            standard_answers: List of ground truth answers
            
        Returns:
            Tuple of (metrics, individual_results):
            - metrics: Dictionary with evaluation metrics:
              - win_rate: Proportion of times answers1 wins (0.5 for ties)
              - total_comparisons: Total number of comparisons
              - wins: Number of times answers1 wins
              - losses: Number of times answers2 wins
              - ties: Number of ties
            - individual_results: List of individual scores for each sample
              (1.0 = answers1 wins, 0.0 = answers2 wins, 0.5 = tie)
        """
        if not (len(questions) == len(answers1) == len(answers2) == len(standard_answers)):
            raise ValueError("All input lists must have the same length")
        
        wins = 0
        ties = 0
        total = len(questions)
        individual_results = []
        
        for q, a1, a2, std in zip(questions, answers1, answers2, standard_answers):
            result = self.evaluate_single(q, a1, a2, std)
            individual_results.append(result)
            if result == 1.0:
                wins += 1
            elif result == 0.5:
                ties += 1
        
        losses = (1 - (wins / (total - ties))) if total - ties > 0 else 0.0
        
        metrics = {
            "win_rate": wins / (total - ties) if total - ties > 0 else 0.0,
            "tie_rate": ties / total if total > 0 else 0.0,
            "total_comparisons": total,
            "wins": wins,
            "losses": losses,
            "ties": ties,
        }
        
        return metrics, individual_results
    
    def generate_and_evaluate(
        self,
        questions: List[str],
        standard_answers: List[str],
        system_prompt1: str,
        system_prompt2: str,
        model: str = None,
    ) -> Tuple[Dict[str, float], List[str], List[str]]:
        """
        Generate responses with two different system prompts and evaluate.
        
        This is the core method for comparing prompt performance.
        
        Args:
            questions: List of questions
            standard_answers: List of ground truth answers
            system_prompt1: First system prompt
            system_prompt2: Second system prompt
            model: Model to use (defaults to target_model)
            
        Returns:
            Tuple of (metrics, responses1, responses2)
            - metrics: Evaluation metrics dict
            - responses1: Responses generated with system_prompt1
            - responses2: Responses generated with system_prompt2
        """
        # Generate responses with both prompts
        responses1 = self.llm_client.batch_generate(
            prompts=questions,
            system_prompt=system_prompt1,
            model=model,
        )
        
        responses2 = self.llm_client.batch_generate(
            prompts=questions,
            system_prompt=system_prompt2,
            model=model,
        )
        
        # Evaluate
        metrics, _ = self.evaluate_batch(
            questions=questions,
            answers1=responses1,
            answers2=responses2,
            standard_answers=standard_answers,
        )
        
        return metrics, responses1, responses2
    
    def compute_improvement(
        self,
        baseline_metrics: Dict[str, float],
        updated_metrics: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute improvement metrics between baseline and updated prompts.
        
        Args:
            baseline_metrics: Metrics from baseline prompt
            updated_metrics: Metrics from updated prompt
            
        Returns:
            Dictionary with improvement metrics:
            - win_rate_delta: Change in win rate
            - improvement_percentage: Percentage improvement
        """
        baseline_win_rate = baseline_metrics.get("win_rate", 0.0)
        updated_win_rate = updated_metrics.get("win_rate", 0.0)
        
        delta = updated_win_rate - baseline_win_rate
        
        # Note: In our setup, we compare old prompt (answer1) vs new prompt (answer2)
        # So a positive delta means the NEW prompt (answer2) wins more often
        # We need to invert this for the loss calculation
        
        return {
            "win_rate_delta": delta,
            "improvement_percentage": (delta * 100) if baseline_win_rate > 0 else 0.0,
            "baseline_win_rate": baseline_win_rate,
            "updated_win_rate": updated_win_rate,
        }
    
    def evaluate_prompt_on_dataset(
        self,
        questions: List[str],
        standard_answers: List[str],
        system_prompt: str,
        model: str = None,
    ) -> Tuple[float, List[str]]:
        """
        Evaluate a single prompt on a dataset by comparing with standard answers.
        
        This is used for evaluation sets where we measure absolute performance.
        
        Args:
            questions: List of questions
            standard_answers: List of ground truth answers
            system_prompt: System prompt to evaluate
            model: Model to use
            
        Returns:
            Tuple of (accuracy, responses)
            - accuracy: Proportion of questions where model response wins
            - responses: Generated responses
        """
        # Generate responses
        responses = self.llm_client.batch_generate(
            prompts=questions,
            system_prompt=system_prompt,
            model=model,
        )
        
        # Compare each response with standard answer
        # We treat the generated response as answer1 and standard as answer2
        wins = 0
        for q, r, std in zip(questions, responses, standard_answers):
            # Use the reward function: if generated response wins, count as correct
            result = self.evaluate_single(q, r, std, std)
            wins += result
        
        accuracy = wins / len(questions) if questions else 0.0
        
        return accuracy, responses

