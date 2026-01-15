"""
Main Trainer for SGD-based prompt optimization.

Implements batch processing, subject classification, RAG-based guidance generation,
and checkpoint/rollback mechanisms.
"""

import os
import shutil
import copy
import logging
import random
from typing import List, Dict, Any, Callable, Optional,Union
from tqdm import tqdm
import mlflow

from .llm_client import LLMClient
from .knowledge_base import KnowledgeBase
from .evaluator import Evaluator
from .prompt_builder import PromptBuilder
from .utils import (
    load_jsonl,
    save_jsonl,
    batch_data,
    setup_mlflow,
    log_metrics_to_mlflow
)

# Setup module logger
logger = logging.getLogger(__name__)


class PromptTuner:
    """
    Main trainer class for SGD-based prompt optimization.
    
    Implements:
    - Batch processing with configurable batch size
    - Dynamic subject classification
    - RAG-based guidance retrieval and merging
    - Win rate tracking as loss metric
    - Periodic evaluation with rollback capability
    - MLflow integration for experiment tracking
    """
    
    def __init__(
        self,
        reward_fn: Callable[[str, str, str, str], List[int]],
        tuning_model_fn: Optional[Callable[[str, Optional[str], float, Optional[int]], str]] = None,
        tuner_model_fn: Optional[Callable[[str, Optional[str], float, Optional[int]], str]] = None,
        embedding_model_fn: Optional[Callable[[str], List[float]]] = None,
        tuning_model_batch_fn: Optional[Callable[[List[str], Any, float, Optional[int]], List[str]]] = None,
        tuner_model_batch_fn: Optional[Callable[[List[str], Any, float, Optional[int]], List[str]]] = None,
        embedding_model_batch_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        mlflow_tracking_uri: str = "http://localhost:5000",
        mlflow_experiment_name: str = "prompt_tuning",
        mlflow_project_name: Optional[str] = None,
        batch_size: int = 10,
        eval_steps: int = 5,
        max_guidance_length: int = 500,
        # Question retrieval parameters (for retrieving guidance based on questions)
        question_retrieval_top_k: int = 3,
        question_retrieval_threshold: float = 0.7,
        # Subject retrieval parameters (for merging related subjects)
        subject_retrieval_top_k: int = 5,
        subject_retrieval_threshold: float = 0.9,
        # Evaluation parameters
        eval_retrieval_top_k: int = 3,
        eval_retrieval_threshold: float = 0.7,
        eval_batch_size: int = 10,
        eval_at_n: int = 1,
        knowledge_base_path: Optional[str] = None,
        max_prompt_tokens: Optional[int] = None,
        token_counting_model: str = "gpt-3.5-turbo",
        # Prompt templates for customization
        guidance_extraction_prompt_template: Optional[str] = None,
        guidance_merge_prompt_template: Optional[str] = None,
        shuffle_batches: bool = False,
    ):
        """
        Initialize the prompt tuner.
        
        Args:
            reward_fn: User-defined reward function (question, ans1, ans2, std) -> [1,0], [0,1], or [0.5,0.5]
            tuning_model_fn: Optional single inference function (auto-created from batch if not provided)
            tuner_model_fn: Optional single inference function (auto-created from batch if not provided)
            embedding_model_fn: Optional single embedding function (auto-created from batch if not provided)
            tuning_model_batch_fn: Batch inference function for tuning model (preferred)
            tuner_model_batch_fn: Batch inference function for tuner model (preferred)
            embedding_model_batch_fn: Batch embedding function (preferred)
            mlflow_tracking_uri: MLflow tracking server URI (default: http://localhost:5000)
            mlflow_experiment_name: Name of the MLflow experiment (default: prompt_tuning)
            mlflow_project_name: Optional project/run name
            batch_size: Number of samples per training batch
            eval_steps: Evaluate every N steps
            max_guidance_length: Maximum character length for guidance
            question_retrieval_top_k: Top-k guidance to retrieve when searching by question
            question_retrieval_threshold: Minimum similarity threshold for question-based retrieval
            subject_retrieval_top_k: Top-k entries to retrieve when merging related subjects
            subject_retrieval_threshold: Minimum similarity threshold for subject-based retrieval
            eval_retrieval_top_k: Top-k guidance to retrieve during evaluation
            eval_retrieval_threshold: Minimum similarity threshold for evaluation retrieval
            eval_batch_size: Batch size for evaluation processing
            eval_at_n: Number of candidate answers to generate for @n evaluation (default: 1, i.e., @1)
            knowledge_base_path: Path to knowledge base storage (default: ./knowledge_base.jsonl)
            max_prompt_tokens: Maximum tokens for system prompt
            token_counting_model: Model name for token counting (default: gpt-3.5-turbo)
            guidance_extraction_prompt_template: Custom template for extracting guidance from errors
            guidance_merge_prompt_template: Custom template for merging guidance from different sources
            shuffle_batches: Whether to shuffle training data at the start of each epoch (default: False)
        """
        self.batch_size = batch_size
        self.eval_steps = eval_steps
        self.question_retrieval_top_k = question_retrieval_top_k
        self.question_retrieval_threshold = question_retrieval_threshold
        self.subject_retrieval_top_k = subject_retrieval_top_k
        self.subject_retrieval_threshold = subject_retrieval_threshold
        self.eval_retrieval_top_k = eval_retrieval_top_k
        self.eval_retrieval_threshold = eval_retrieval_threshold
        self.eval_batch_size = eval_batch_size
        self.eval_at_n = eval_at_n
        self.mlflow_project_name = mlflow_project_name
        self.shuffle_batches = shuffle_batches
        
        # Set prompt templates (use defaults if not provided)
        self.guidance_extraction_prompt_template = guidance_extraction_prompt_template or self._get_default_guidance_extraction_prompt()
        self.guidance_merge_prompt_template = guidance_merge_prompt_template or self._get_default_guidance_merge_prompt()
        
        # Initialize LLM client with user-defined functions
        self.llm_client = LLMClient(
            tuning_model_fn=tuning_model_fn,
            tuner_model_fn=tuner_model_fn,
            embedding_model_fn=embedding_model_fn,
            tuning_model_batch_fn=tuning_model_batch_fn,
            tuner_model_batch_fn=tuner_model_batch_fn,
            embedding_model_batch_fn=embedding_model_batch_fn,
        )
        
        # Initialize knowledge base
        kb_path = knowledge_base_path
        self.knowledge_base = KnowledgeBase(
            storage_path=kb_path,
            llm_client=self.llm_client,
            max_guidance_length=max_guidance_length,
            guidance_merge_prompt_template=self.guidance_merge_prompt_template,
        )
        
        # Initialize evaluator
        self.evaluator = Evaluator(
            reward_fn=reward_fn,
            llm_client=self.llm_client,
        )
        
        # Initialize prompt builder
        self.prompt_builder = PromptBuilder(
            max_total_tokens=max_prompt_tokens,
            token_model=token_counting_model,
        )
        # Setup MLflow (only if tracking URI is provided)
        self.use_mlflow = bool(mlflow_tracking_uri and mlflow_tracking_uri.strip())
        if self.use_mlflow:
            setup_mlflow(mlflow_tracking_uri, mlflow_experiment_name, mlflow_project_name)
        
        # Training state
        self.current_prompt = ""
        self.step_count = 0
        self.best_eval_score = -float('inf')
        self.checkpoint_dir = None
        
        # Store cases where updated prompt performs worse than baseline (for debugging)
        self.negative_optimization_cases: List[Dict[str, Any]] = []
        
        # Cumulative statistics for tracking overall win rate trend
        self.cumulative_stats = {
            "total_wins": 0,
            "total_losses": 0,
            "total_ties": 0,
            "total_comparisons": 0
        }
    
    def _get_default_guidance_extraction_prompt(self) -> str:
        """Get the default prompt template for extracting guidance from errors."""
        return (
            "You are analyzing model responses for the subject: {subject}\n\n"
            "Here are some examples where the model may have made mistakes:\n\n"
            "{error_context}\n\n"
            "Based on these examples, provide concise guidance (max 2-3 sentences) that would help "
            "the model perform better on this type of question. Focus on the key principles or "
            "approaches that should be followed.\n\n"
            "Guidance:"
        )
    
    def _get_default_guidance_merge_prompt(self) -> str:
        """Get the default prompt template for merging guidance from different sources."""
        return (
            "You are synthesizing guidance for the subject: {subject}\n\n"
            "Existing guidance from related subjects in the knowledge base:\n"
            "{existing_guidance}\n\n"
            "New guidance to incorporate:\n{new_guidance}\n\n"
            "Please merge these guidance points into a single with same style, coherent guidance text for '{subject}'.\n"
            "Consider insights from related subjects and adapt them to the current context.\n"
            "The output should be concise, clear, and no longer than {max_length} characters.\n"
            "Focus on the most important and actionable advice.\n\n"
            "Merged guidance:"
        )
    
    def _retrieve_guidance_for_batch(
        self,
        subjects: List[str],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[str]:
        """
        Retrieve relevant guidance for each subject in the batch.
        
        Args:
            subjects: List of subjects (required - should be generated from questions using classify_subjects)
            top_k: Override default question_retrieval_top_k
            threshold: Override default question_retrieval_threshold
            
        Returns:
            List of personalized system prompts (one per subject)
        """
        actual_top_k = top_k if top_k is not None else self.question_retrieval_top_k
        actual_threshold = threshold if threshold is not None else self.question_retrieval_threshold
        
        system_prompts = []
        
        for subject in subjects:
            # Retrieve relevant guidance using subject
            retrieved = self.knowledge_base.retrieve(
                query=subject,
                top_k=actual_top_k,
                threshold=actual_threshold,
            )
            
            # Build prompt from retrieved entries
            if retrieved:
                retrieved_entries = [r["entry"] for r in retrieved]
                prompt = self.prompt_builder.build_system_prompt(retrieved_entries)
            else:
                # Fallback to empty prompt if no relevant guidance found
                prompt = ""
            
            system_prompts.append(prompt)
        
        return system_prompts
    
    def _generate_guidance_for_batch(
        self,
        questions: List[str],
        standard_answers: List[str],
        subjects: List[str],
        current_responses: List[str],
        subject_retrieval_top_k: Optional[int] = None,
        subject_retrieval_threshold: Optional[float] = None,
    ) -> Dict[str, str]:
        """
        Generate guidance for a batch based on errors and retrieved knowledge.
        
        Args:
            questions: Batch questions
            standard_answers: Batch standard answers
            subjects: Batch subjects
            current_responses: Current model responses
            subject_retrieval_top_k: Optional override for subject retrieval top_k (default: self.subject_retrieval_top_k)
            subject_retrieval_threshold: Optional override for subject retrieval threshold (default: self.subject_retrieval_threshold)
            
        Returns:
            Dictionary mapping subject to merged guidance
        """
        subject_guidance = {}
        
        # Use provided parameters or fall back to instance variables
        top_k = subject_retrieval_top_k if subject_retrieval_top_k is not None else self.subject_retrieval_top_k
        threshold = subject_retrieval_threshold if subject_retrieval_threshold is not None else self.subject_retrieval_threshold
        
        # Group questions by subject
        from collections import defaultdict
        subject_groups = defaultdict(list)
        for i, subject in enumerate(subjects):
            subject_groups[subject].append({
                "question": questions[i],
                "standard_answer": standard_answers[i],
                "current_response": current_responses[i],
            })
        
        # For each subject, generate guidance
        for subject, items in subject_groups.items():
            # Retrieve relevant guidance from knowledge base
            # Use semantic search to find related subjects, not just exact matches
            retrieved = self.knowledge_base.retrieve(
                query=subject,
                top_k=top_k,
                threshold=threshold,
            )
            retrieved_guidance = [r["entry"]["guidance"] for r in retrieved]
            related_subjects = [r["entry"]["subject"] for r in retrieved]
            # Filter items: only keep cases where standard answer is genuinely better
            filtered_items = self._filter_items_by_reward(items)    
            # Generate new guidance based on filtered error cases
            if filtered_items:
                error_context = self._build_error_context(filtered_items)
                new_guidance = self._generate_new_guidance(subject, error_context)
                # Merge retrieved guidance from related subjects with new guidance
                merged_guidance = self.knowledge_base.merge_guidance(
                    retrieved_guidance=retrieved_guidance,
                    new_guidance=new_guidance,
                    subject=subject,
                    related_subjects=related_subjects,
                )
                merged_subject = self.knowledge_base.merge_subjects(related_subjects) if related_subjects else subject
                subject_guidance[merged_subject] = merged_guidance
        
        return subject_guidance
    
    def _filter_items_by_reward(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter items using reward function to keep only genuine error cases.
        
        Only keeps items where the standard answer is genuinely better than 
        the model's response according to the reward function.
        
        Args:
            items: List of items with question, current_response, and standard_answer
            
        Returns:
            Filtered list of items where standard answer is better
        """
        filtered_items = []
        
        for item in items:
            question = item["question"]
            current_response = item["current_response"]
            standard_answer = item["standard_answer"]
            
            try:
                # Use reward function to compare: current_response vs standard_answer
                # If standard_answer wins (result = [0, 1]), keep this item
                result = self.evaluator.reward_fn(
                    question,
                    current_response,  # answer1: model's response
                    standard_answer,   # answer2: standard answer
                    standard_answer,   # ground truth
                )
                
                # If standard_answer wins ([0, 1]), it's a genuine error case
                if result == [0, 1]:
                    filtered_items.append(item)
                # If current_response wins or ties ([1, 0]), skip this item
                
            except Exception as e:
                # If reward function fails, conservatively keep the item
                logger.warning(
                    f"Reward function failed for item, keeping it: {str(e)}"
                )
                filtered_items.append(item)
        
        return filtered_items
    
    def _build_error_context(self, items: List[Dict[str, Any]]) -> str:
        """Build context string from batch items for guidance generation."""
        context_parts = []
        for i, item in enumerate(items, 1):
            context_parts.append(
                f"Example {i}:\n"
                f"Question: {item['question']}\n"
                f"Current answer: {item['current_response'][-256:]}\n"
                f"Correct answer: {item['standard_answer'][-256:]}\n"
            )
        return "\n".join(context_parts)
    
    def _generate_new_guidance(self, subject: str, error_context: str) -> str:
        """Generate new guidance based on subject and error context."""
        prompt = self.guidance_extraction_prompt_template.format(
            subject=subject,
            error_context=error_context,
        )
        
        guidance = self.llm_client.generate_with_tuner_model(
            prompt=prompt,
            temperature=0.5,
        )
        
        # Handle case where guidance generation failed
        if guidance is None:
            logger.warning(f"Failed to generate guidance for subject: {subject}")
            return ""
        
        return guidance.strip()
    
    def _process_batch(
        self,
        batch: List[Dict[str, Any]],
        update_knowledge_base: bool = True,
        subject_retrieval_top_k: Optional[int] = None,
        subject_retrieval_threshold: Optional[float] = None,
        is_retrieval_subject: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single training batch.
        
        Args:
            batch: List of data items with 'question' and 'standard_answer'
            update_knowledge_base: Whether to update the knowledge base
            subject_retrieval_top_k: Optional override for subject retrieval top_k (default: self.subject_retrieval_top_k)
            subject_retrieval_threshold: Optional override for subject retrieval threshold (default: self.subject_retrieval_threshold)
            is_retrieval_subject: Whether to retrieve guidance based on subjects or questions
        Returns:
            Dictionary with batch metrics, or None if no new guidance was generated
        """
        questions = [item["question"] for item in batch]
        standard_answers = [item["answer"] for item in batch]
        
        # Step 1: Classify subjects for each question
        subjects = self.llm_client.classify_subjects(
            questions=questions
        )
        if subjects is None:
            logger.warning("All cases in batch failed, skipping batch")
            return None
        
        # Step 2: Retrieve relevant guidance for each question (RAG-based)
        # Use subject for retrieval since we already have subjects from classification
        baseline_system_prompts = self._retrieve_guidance_for_batch(
            subjects=subjects if is_retrieval_subject else questions,
        )

        # Step 3: Evaluate with retrieved guidance (baseline)
        baseline_responses = self.llm_client.batch_generate(
            prompts=questions,
            system_prompt=baseline_system_prompts,  # Each question gets its own relevant guidance
        )
        
        # Filter out failed cases (marked as None)
        valid_indices = [i for i, resp in enumerate(baseline_responses) if resp is not None]
        if len(valid_indices) < len(questions):
            failed_count = len(questions) - len(valid_indices)
            logger.warning(f"Filtered out {failed_count} failed case(s) from batch")
        
        # If all cases failed, skip this batch
        if not valid_indices:
            logger.warning("All cases in batch failed, skipping batch")
            return None
        
        # Filter data to only include valid cases
        questions = [questions[i] for i in valid_indices]
        standard_answers = [standard_answers[i] for i in valid_indices]
        subjects = [subjects[i] for i in valid_indices]
        baseline_responses = [baseline_responses[i] for i in valid_indices]
        baseline_system_prompts = [baseline_system_prompts[i] for i in valid_indices]
        
        # Step 4: Generate new guidance for this batch based on errors
        subject_guidance = self._generate_guidance_for_batch(
            questions=questions,
            standard_answers=standard_answers,
            subjects=subjects,
            current_responses=baseline_responses,
            subject_retrieval_top_k=subject_retrieval_top_k,
            subject_retrieval_threshold=subject_retrieval_threshold,
        )
        
        # Early return if no new guidance was generated
        # Without new guidance, there's no point in updating KB or re-evaluating
        if not subject_guidance:
            logger.warning("All cases is already correct, skipping batch")
            return None
        
        # Step 5: Save knowledge base state before update (for potential rollback)
        kb_backup = None
        if update_knowledge_base:
            kb_backup = copy.deepcopy(self.knowledge_base.entries)
            # Update knowledge base with new guidance
            for subject, guidance in subject_guidance.items():
                self.knowledge_base.update_entry(subject, guidance)
        
        # Step 6: Retrieve updated guidance for each question (after knowledge base update)
        # Use subject for retrieval since we already have subjects from classification
        updated_system_prompts = self._retrieve_guidance_for_batch(
            subjects=subjects if is_retrieval_subject else questions,
        )
        if all(prompt.strip() == "" for prompt in updated_system_prompts):
            logger.warning("All updated system prompts are empty, skipping evaluation")
            self.knowledge_base.entries = kb_backup
            return None
        # Record indices where updated_system_prompts are non-empty
        valid_indices = [i for i, prompt in enumerate(updated_system_prompts) if prompt.strip() != ""]
        if len(valid_indices) == 0:
            logger.warning("All updated system prompts are empty, skipping evaluation")
            self.knowledge_base.entries = kb_backup
            return None
        # Filter all lists using the same valid indices
        updated_system_prompts = [updated_system_prompts[i] for i in valid_indices]
        questions = [questions[i] for i in valid_indices]
        baseline_responses = [baseline_responses[i] for i in valid_indices]
        standard_answers = [standard_answers[i] for i in valid_indices]
        if len(questions) == 0:
            logger.warning("All cases in batch failed updated system prompts with non-empty prompts, skipping evaluation")
            self.knowledge_base.entries = kb_backup
            return None
        # Step 7: Evaluate with updated guidance
        updated_responses = self.llm_client.batch_generate(
            prompts=questions,
            system_prompt=updated_system_prompts,  # Each question gets updated relevant guidance
        )
        # Filter out failed cases in updated responses
        valid_indices_updated = [i for i, resp in enumerate(updated_responses) if resp is not None]
        if len(valid_indices_updated) < len(questions):
            failed_count = len(questions) - len(valid_indices_updated)
            logger.warning(f"Filtered out {failed_count} failed case(s) from updated responses")
        
        # If all updated cases failed, skip evaluation and rollback KB update
        if not valid_indices_updated:
            logger.warning("All updated cases failed, skipping evaluation")
            if update_knowledge_base and kb_backup is not None:
                logger.info("Rolling back knowledge base update due to all failures")
                self.knowledge_base.entries = kb_backup
            return None
        
        # Filter to only include cases that succeeded in both baseline and updated
        questions = [questions[i] for i in valid_indices_updated]
        baseline_responses = [baseline_responses[i] for i in valid_indices_updated]
        baseline_system_prompts = [baseline_system_prompts[i] for i in valid_indices_updated]
        updated_responses = [updated_responses[i] for i in valid_indices_updated]
        standard_answers = [standard_answers[i] for i in valid_indices_updated]
        updated_system_prompts = [updated_system_prompts[i] for i in valid_indices_updated]
        # Step 8: Compare performance
        metrics, individual_results = self.evaluator.evaluate_batch(
            questions=questions,
            answers1=updated_responses,
            answers2=baseline_responses,
            standard_answers=standard_answers,
        )
        # Step 8: Record negative optimization cases (where updated prompt performs worse)
        # Use the individual_results from evaluate_batch to avoid re-evaluation
        for result, q, a_updated, a_baseline, std_ans, prompt_updated, prompt_baseline in zip(
            individual_results, questions, updated_responses, baseline_responses,
            standard_answers, updated_system_prompts, baseline_system_prompts
        ):
            # If baseline wins (result == 0.0), or if tie and we want to track it
            # Here we track cases where updated doesn't win (loss or tie)
            if result  == 0.0:  # Updated doesn't win (either loses or ties)
                case_record = {
                    "step": self.step_count + 1,  # Will be incremented after this method returns
                    "question": q,
                    "standard_answer": std_ans,
                    "baseline_prompt": prompt_baseline,
                    "updated_prompt": prompt_updated,
                    "baseline_response": a_baseline,
                    "updated_response": a_updated,
                    "comparison_result": "baseline_wins" if result == 0.0 else "tie",
                    "win_score": result,  # 0.0 = baseline wins, 0.5 = tie, 1.0 = updated wins
                }
                self.negative_optimization_cases.append(case_record)
        
        # Step 9: Check if we should rollback knowledge base update
        # Rollback if: wins < losses OR all ties
        if update_knowledge_base and kb_backup is not None:
            wins = metrics["wins"]
            losses_count = metrics["total_comparisons"] - metrics["wins"] - metrics["ties"]
            tie_rate = metrics.get("tie_rate", 0.0)
            if wins <= losses_count or tie_rate == 1.0:
                logger.info(
                    f"Rolling back knowledge base update: wins={wins}, "
                    f"losses={losses_count}, tie_rate={tie_rate:.2f}"
                )
                self.knowledge_base.entries = kb_backup
            
                return None
        
        # Calculate loss
        # In our comparison: answer1=updated, answer2=baseline
        # win_rate is the rate at which answer1 (updated) wins
        # Higher win_rate means updated prompt is better
        # For loss, we want lower is better, so we invert it
        if metrics['tie_rate'] == 1:
            loss = 0.5
        else:
            loss = metrics["losses"]  # Lower loss means updated prompt is better
        
        # Calculate average prompt length for monitoring
        avg_guidance_length = sum(len(p) for p in updated_system_prompts) / len(updated_system_prompts)
        # Update cumulative statistics
        cumulative_wins = self.cumulative_stats["total_wins"] + metrics["wins"]
        cumulative_losses_count = metrics["total_comparisons"] - metrics["wins"] - metrics["ties"]
        cumulative_losses = self.cumulative_stats["total_losses"] + cumulative_losses_count
        cumulative_ties = self.cumulative_stats["total_ties"] + metrics["ties"]
        cumulative_total = cumulative_wins + cumulative_losses + cumulative_ties
        
        self.cumulative_stats["total_wins"] = cumulative_wins
        self.cumulative_stats["total_losses"] = cumulative_losses
        self.cumulative_stats["total_ties"] = cumulative_ties
        self.cumulative_stats["total_comparisons"] = cumulative_total
        
        # Calculate cumulative win rate (excluding ties)
        if (cumulative_wins + cumulative_losses) > 0:
            cumulative_win_rate = cumulative_wins / (cumulative_wins + cumulative_losses)
        else:
            cumulative_win_rate = 0.5
        
        # Calculate cumulative loss (inverse of win rate for consistency with existing loss metric)
        cumulative_loss = 1.0 - cumulative_win_rate
        if update_knowledge_base:

            self.knowledge_base._save_entries()
        
        return {
            "loss": loss,
            "win_rate_updated": metrics["win_rate"],        # This is updated's win rate
            "win_rate_baseline": metrics.get("loss_rate", 1.0 - metrics["win_rate"]), # This is baseline's win rate
            "tie_rate": metrics.get("tie_rate", 0.0),       # Tie rate
            "subjects_processed": len(subject_guidance),
            "avg_guidance_length": avg_guidance_length,
            "knowledge_base_size": len(self.knowledge_base.entries),
            # Cumulative metrics for overall trend tracking
            "cumulative_win_rate": cumulative_win_rate,
            "cumulative_loss": cumulative_loss,
            "cumulative_wins": cumulative_wins,
            "cumulative_losses": cumulative_losses,
            "cumulative_ties": cumulative_ties,
            "cumulative_total": cumulative_total,
        }
    
    def _evaluate_on_eval_set(
        self,
        eval_data: List[Dict[str, Any]],
        save_wrong_cases_path: Optional[str] = None,
        is_retrieval_subject: bool = True,
        eval_at_n: Optional[int] = None,
    ) -> float:
        """
        Evaluate using RAG-based retrieval on evaluation set with batch processing.
        Supports @n evaluation: generates n candidate answers and checks if standard answer is in top-n.
        
        Args:
            eval_data: Evaluation data
            save_wrong_cases_path: Optional path to save wrong cases for analysis
            is_retrieval_subject: Whether to retrieve guidance based on subjects or questions
            eval_at_n: Override default eval_at_n for this evaluation (default: self.eval_at_n)
            
        Returns:
            Evaluation score (@n accuracy)
        """
        from .utils import batch_data, save_jsonl
        
        # Use provided eval_at_n or default to self.eval_at_n
        actual_eval_at_n = eval_at_n if eval_at_n is not None else self.eval_at_n
        
        # Process evaluation in batches
        eval_batches = batch_data(eval_data, self.eval_batch_size)
        
        total_wins = 0.0
        total_questions = 0
        wrong_cases = []  # Collect wrong cases for analysis
        
        for batch in tqdm(eval_batches, desc=f"Evaluating batches (@{actual_eval_at_n})", unit="batch"):
            questions = [item["question"] for item in batch]
            standard_answers = [item["answer"] for item in batch]
            
            # Step 1: Classify subjects for each question using tuner model (required in eval mode)
            if is_retrieval_subject:
                subjects = self.llm_client.classify_subjects(
                    questions=questions
                )
                if subjects is None:
                    logger.warning("All cases in eval batch failed subject classification, skipping batch")
                    continue
            else:
                subjects = questions
            # Step 2: Retrieve relevant guidance using subject
            system_prompts = self._retrieve_guidance_for_batch(
                subjects=subjects,
                top_k=self.eval_retrieval_top_k,
                threshold=self.eval_retrieval_threshold,
            )
            if len(questions) == 0:
                logger.warning("All cases in eval batch failed system prompts with non-empty prompts, skipping batch")
                continue
            
            # Step 3: Generate n candidate responses for each question (@n evaluation)
            # For @n evaluation, we generate n responses per question
            # Build batch inputs: repeat each question n times for batch generation
            batch_prompts = []
            batch_system_prompts_list = []
            
            for q_idx, question in enumerate(questions):
                # Get system prompt for this question
                sys_prompt = system_prompts[q_idx] if isinstance(system_prompts, list) else system_prompts
                # Repeat question n times for @n evaluation
                batch_prompts.extend([question] * actual_eval_at_n)
                batch_system_prompts_list.extend([sys_prompt] * actual_eval_at_n)
            
            # Batch generate all candidate responses at once
            all_responses = self.llm_client.batch_generate(
                prompts=batch_prompts,
                system_prompt=batch_system_prompts_list,
                use_tuner_model=False,
            )
            
            # Group responses by question: [[candidates for q1], [candidates for q2], ...]
            all_candidate_responses = []
            for q_idx in range(len(questions)):
                start_idx = q_idx * actual_eval_at_n
                end_idx = start_idx + actual_eval_at_n
                candidate_responses = [
                    resp for resp in all_responses[start_idx:end_idx] 
                    if resp is not None
                ]
                if len(candidate_responses) < actual_eval_at_n:
                    logger.warning(f"Filtered out {actual_eval_at_n - len(candidate_responses)} case(s) with all candidates failed from evaluation batch")
                all_candidate_responses.append(candidate_responses)
            
            # Filter out cases where all candidates failed
            valid_indices = [i for i, candidates in enumerate(all_candidate_responses) if len(candidates) > 0]
            if len(valid_indices) < len(questions):
                failed_count = len(questions) - len(valid_indices)
                logger.warning(f"Filtered out {failed_count} case(s) with all candidates failed from evaluation batch")
            
            # Skip this batch if all cases failed
            if not valid_indices:
                logger.warning("All evaluation cases in batch failed")
                continue
            
            # Filter to only valid cases
            batch_questions = [questions[i] for i in valid_indices]
            batch_candidate_responses = [all_candidate_responses[i] for i in valid_indices]
            batch_standard_answers = [standard_answers[i] for i in valid_indices]
            batch_system_prompts = [
                system_prompts[i] if isinstance(system_prompts, list)
                else system_prompts
                for i in valid_indices
            ]
            
            # Calculate @n accuracy: check if standard answer is in top-n candidates
            batch_wins = 0.0
            for q, candidates, std, sys_prompt in zip(
                batch_questions, batch_candidate_responses,
                batch_standard_answers, batch_system_prompts
            ):
                # Check if any candidate matches the standard answer (@n evaluation)
                found_correct = False
                best_candidate = None
                best_result = 0.0
                
                for candidate in candidates:
                    # Use reward function to check if this candidate is correct
                    result = self.evaluator.evaluate_single(q, candidate, std, std)
                    if result != 0.0:  # Candidate wins (is correct)
                        found_correct = True
                        best_candidate = candidate
                        best_result = result
                        break  # Found correct answer in top-n, no need to check others
                    # Track the best candidate even if not correct
                    if result > best_result:
                        best_result = result
                        best_candidate = candidate
                
                # Collect case for analysis (use best candidate if none correct)
                wrong_cases.append({
                    "question": q,
                    "response": best_candidate if best_candidate else (candidates[0] if candidates else None),
                    "candidates": candidates,
                    "standard_answer": std,
                    "system_prompt": sys_prompt,
                    "result": best_result,
                    "found_in_top_n": found_correct,
                })
                
                if found_correct:
                    batch_wins += 1
            
            total_wins += batch_wins
            total_questions += len(batch_questions)
        
        accuracy = total_wins / total_questions if total_questions > 0 else 0.0
        
        # Save wrong cases if path is provided
        if save_wrong_cases_path and wrong_cases:
            save_jsonl(wrong_cases, save_wrong_cases_path)
            logger.info(
                f"Saved {len(wrong_cases)} wrong cases to "
                f"{save_wrong_cases_path}"
            )
        
        return accuracy
    
    def _save_checkpoint(self, checkpoint_dir: str, step: int) -> None:
        """Save checkpoint of current state."""
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save knowledge base (the main state in RAG-based approach)
        # Save directly from memory to ensure consistency
        kb_checkpoint_path = os.path.join(checkpoint_path, "knowledge_base.jsonl")
        save_jsonl(self.knowledge_base.entries, kb_checkpoint_path, append=False)
        
        # Save metadata
        import json
        metadata = {
            "step": step,
            "knowledge_base_size": len(self.knowledge_base.entries),
            "question_retrieval_top_k": self.question_retrieval_top_k,
            "question_retrieval_threshold": self.question_retrieval_threshold,
            "subject_retrieval_top_k": self.subject_retrieval_top_k,
            "subject_retrieval_threshold": self.subject_retrieval_threshold,
            "eval_retrieval_top_k": self.eval_retrieval_top_k,
            "eval_retrieval_threshold": self.eval_retrieval_threshold,
            "eval_batch_size": self.eval_batch_size,
        }
        metadata_path = os.path.join(checkpoint_path, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_checkpoint(self, checkpoint_dir: str, step: int) -> None:
        """Load checkpoint from saved state."""
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}")
        
        # Load knowledge base
        kb_checkpoint_path = os.path.join(checkpoint_path, "knowledge_base.jsonl")
        shutil.copy(kb_checkpoint_path, self.knowledge_base.storage_path)
        
        # Reload knowledge base entries from updated storage file
        self.knowledge_base.entries = self.knowledge_base._load_entries()
    
    def train(
        self,
        train_data_path: Optional[Union[str, List[Dict[str, Any]]]],
        eval_data_path: Optional[str] = None,
        num_epochs: int = 1,
        checkpoint_dir: str = "./checkpoints",
        update_knowledge_base: bool = True,
        subject_retrieval_top_k: Optional[int] = None,
        subject_retrieval_threshold: Optional[float] = None,
        is_retrieval_subject: bool = True,
        save_wrong_cases_path:str = None
    ) -> None:
        """
        Train the prompt optimizer.
        
        Args:
            train_data_path: Path to training data JSONL file
            eval_data_path: Optional path to evaluation data JSONL file
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory for saving checkpoints
            update_knowledge_base: Whether to update the knowledge base during training
            subject_retrieval_top_k: Optional override for subject retrieval top_k (default: self.subject_retrieval_top_k)
            subject_retrieval_threshold: Optional override for subject retrieval threshold (default: self.subject_retrieval_threshold)
        """
        # Load data
        train_data = load_jsonl(train_data_path)
        eval_data = load_jsonl(eval_data_path) if eval_data_path else None
        
        if not train_data:
            raise ValueError(f"No training data found at {train_data_path}")
        
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Start MLflow run (only if MLflow is enabled)
        if self.use_mlflow:
            mlflow_context = mlflow.start_run(run_name=self.mlflow_project_name)
            # Log hyperparameters
            mlflow.log_params({
                "batch_size": self.batch_size,
                "eval_steps": self.eval_steps,
                "question_retrieval_top_k": self.question_retrieval_top_k,
                "question_retrieval_threshold": self.question_retrieval_threshold,
                "subject_retrieval_top_k": self.subject_retrieval_top_k,
                "subject_retrieval_threshold": self.subject_retrieval_threshold,
                "eval_retrieval_top_k": self.eval_retrieval_top_k,
                "eval_retrieval_threshold": self.eval_retrieval_threshold,
                "num_epochs": num_epochs,
                "shuffle_batches": self.shuffle_batches,
            })
        else:
            # Use a dummy context manager when MLflow is disabled
            from contextlib import nullcontext
            mlflow_context = nullcontext()
        
        with mlflow_context:
            for epoch in range(num_epochs):
                logger.info(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
                
                # Shuffle data if enabled
                epoch_data = copy.deepcopy(train_data)
                if self.shuffle_batches:
                    random.shuffle(epoch_data)
                    logger.info("Training data shuffled for this epoch")
                
                # Create batches
                batches = batch_data(epoch_data, self.batch_size)
                
                # Process each batch
                for batch_idx, batch in enumerate(tqdm(batches, desc="Training")):
                    metrics_old = None
                    # Process batch
                    metrics = self._process_batch(
                        batch, 
                        update_knowledge_base=update_knowledge_base,
                        subject_retrieval_top_k=subject_retrieval_top_k,
                        subject_retrieval_threshold=subject_retrieval_threshold,
                        is_retrieval_subject=is_retrieval_subject,
                    )
                    self.step_count += 1
                    if metrics is None:
                        metrics = metrics_old
                    else:
                        metrics_old = metrics
                    # Log Loss information
                    if metrics is not None:
                        logger.info(
                            f"Step {self.step_count} - "
                            f"Batch Loss: {metrics['loss']:.4f}, "
                            f"Batch Win Rate (Updated): {metrics['win_rate_updated']:.4f}, "
                            f"Cumulative Win Rate: {metrics['cumulative_win_rate']:.4f} "
                            f"Subjects Processed: {metrics['subjects_processed']}"
                        )
                        # Log to MLflow (only if enabled)
                        if self.use_mlflow:
                            log_metrics_to_mlflow({
                                "loss_per_batch": metrics["loss"],
                                "subjects_processed": metrics["subjects_processed"],
                                "avg_prompt_length": metrics["avg_guidance_length"], 
                                "knowledge_base_size": metrics["knowledge_base_size"],
                                "cumulative_win_rate": metrics["cumulative_win_rate"],
                                "absolute_wins": metrics["cumulative_wins"] - metrics["cumulative_losses"],
                            }, step=self.step_count)
                    
                    # Periodic evaluation
                    if eval_data and self.step_count % self.eval_steps == 0:
                        eval_score = self._evaluate_on_eval_set(
                            eval_data,
                            is_retrieval_subject=is_retrieval_subject,
                            save_wrong_cases_path=os.path.join(save_wrong_cases_path,f"eval_wrong_cases_{self.step_count}.jsonl") if save_wrong_cases_path else None
                            )
                        logger.info(f"\nStep {self.step_count} - Eval Score: {eval_score:.4f}")
                        if self.use_mlflow:
                            log_metrics_to_mlflow({
                                "eval_accuracy": eval_score,
                            }, step=self.step_count)
                        # Save checkpoint
                        self._save_checkpoint(checkpoint_dir, self.step_count)
                if eval_data:
                    eval_score = self._evaluate_on_eval_set(
                            eval_data,
                            is_retrieval_subject=is_retrieval_subject,
                            save_wrong_cases_path=os.path.join(save_wrong_cases_path,f"eval_wrong_cases_{self.step_count}.jsonl") if save_wrong_cases_path else None
                            )
                    logger.info(f"\nepoch {epoch + 1} - Eval Score: {eval_score:.4f}")
                    if self.use_mlflow:
                        log_metrics_to_mlflow({
                            "eval_accuracy": eval_score,
                        }, step=self.step_count)
                # take -1*epoch for checkpoint step
                self._save_checkpoint(checkpoint_dir, -epoch)

            # Save negative optimization cases if any exist
            if self.negative_optimization_cases:
                self._save_negative_cases(checkpoint_dir)
                logger.info(
                    f"Saved {len(self.negative_optimization_cases)} negative optimization cases "
                    f"to {checkpoint_dir}/negative_optimization_cases.jsonl"
                )
    
    def save_prompt(self, output_path: str) -> None:
        """
        Save the knowledge base summary to a file.
        In RAG mode, prompts are dynamically retrieved per question.
        
        Args:
            output_path: Path to save the knowledge base summary
        """
        summary = self.get_knowledge_base_summary()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Knowledge Base Summary (RAG Mode)\n\n")
            f.write(f"Total entries: {len(self.knowledge_base.entries)}\n")
            f.write(f"Retrieval settings: question_retrieval_top_k={self.question_retrieval_top_k}, question_retrieval_threshold={self.question_retrieval_threshold}, subject_retrieval_top_k={self.subject_retrieval_top_k}, subject_retrieval_threshold={self.subject_retrieval_threshold}, eval_retrieval_top_k={self.eval_retrieval_top_k}, eval_retrieval_threshold={self.eval_retrieval_threshold}, eval_batch_size={self.eval_batch_size}\n\n")
            f.write("=" * 80 + "\n\n")
            f.write(summary)
        logger.info(f"Knowledge base summary saved to {output_path}")
    
    def get_current_prompt(self) -> str:
        """
        Get a summary of the knowledge base.
        In RAG mode, actual prompts are dynamically retrieved per question.
        Use retrieve_prompt_for_question() to get a specific prompt.
        """
        return self.get_knowledge_base_summary()
    
    def retrieve_prompt_for_question(self, question: str, subject: Optional[str] = None) -> str:
        """
        Retrieve a personalized prompt for a specific question.
        
        Args:
            question: The question to get guidance for
            subject: Optional pre-classified subject (for logging/debugging only)
            
        Returns:
            Personalized system prompt with relevant guidance
        """
        # Classify subject if not provided (for debugging/logging purposes)
        if subject is None:
            subjects = self.llm_client.classify_subjects(
                questions=[question]
            )
            if subjects is None or len(subjects) == 0:
                logger.warning("Failed to classify subject for question")
                return ""
            subject = subjects[0]
        
        # Retrieve and build prompt using subject-based retrieval
        system_prompts = self._retrieve_guidance_for_batch(
            subjects=[subject],
        )
        
        return system_prompts[0]
    
    def get_knowledge_base_summary(self) -> str:
        """Get a summary of the current knowledge base."""
        return self.knowledge_base.export_to_text()
    
    def _save_negative_cases(self, output_dir: str) -> None:
        """
        Save negative optimization cases to a JSONL file.
        
        These are cases where the updated prompt performed worse than or equal to the baseline.
        
        Args:
            output_dir: Directory to save the file
        """
        import json
        
        output_path = os.path.join(output_dir, "negative_optimization_cases.jsonl")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for case in self.negative_optimization_cases:
                f.write(json.dumps(case, ensure_ascii=False) + '\n')
        
        # Also create a human-readable text version
        text_output_path = os.path.join(output_dir, "negative_optimization_cases.txt")
        with open(text_output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("NEGATIVE OPTIMIZATION CASES\n")
            f.write("(Cases where updated prompt performed worse than or equal to baseline)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total cases: {len(self.negative_optimization_cases)}\n\n")
            
            # Count by result type
            baseline_wins = sum(
                1 for c in self.negative_optimization_cases
                if c["comparison_result"] == "baseline_wins"
            )
            ties = sum(
                1 for c in self.negative_optimization_cases
                if c["comparison_result"] == "tie"
            )
            
            f.write("Breakdown:\n")
            f.write(f"  - Baseline wins: {baseline_wins}\n")
            f.write(f"  - Ties: {ties}\n\n")
            f.write("=" * 80 + "\n\n")
            
            for idx, case in enumerate(self.negative_optimization_cases, 1):
                f.write(f"Case {idx} (Step {case['step']}, Result: {case['comparison_result']})\n")
                f.write("-" * 80 + "\n\n")
                
                f.write(f"Question:\n{case['question']}\n\n")
                
                f.write(f"Standard Answer:\n{case['standard_answer']}\n\n")
                
                f.write(f"Baseline Prompt (OLD):\n{case['baseline_prompt']}\n\n")
                
                f.write(f"Baseline Response:\n{case['baseline_response']}\n\n")
                
                f.write(f"Updated Prompt (NEW):\n{case['updated_prompt']}\n\n")
                
                f.write(f"Updated Response:\n{case['updated_response']}\n\n")
                
                f.write("=" * 80 + "\n\n")
        
        logger.info(f"Negative cases saved to {output_path} and {text_output_path}")

