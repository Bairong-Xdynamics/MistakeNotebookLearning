"""
LLM Client for user-defined model functions.

Provides unified interface for tuning model, tuner model, and embedding model
using user-provided functions (UDF).
"""

from typing import List, Dict, Any, Optional, Union, Callable
import time
import logging

# Setup module logger
logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified client for interacting with user-defined model functions.
    
    Supports:
    - Tuning model (the model being optimized)
    - Tuner model (for guidance generation and subject classification)
    - Embedding model (for knowledge base retrieval)
    
    All models are provided as user-defined functions for maximum flexibility.
    """
    
    def __init__(
        self,
        tuning_model_fn: Optional[Callable[[str, Optional[str], float, Optional[int]], str]] = None,
        tuner_model_fn: Optional[Callable[[str, Optional[str], float, Optional[int]], str]] = None,
        embedding_model_fn: Optional[Callable[[str], List[float]]] = None,
        tuning_model_batch_fn: Optional[Callable[[List[str], Union[str, List[str]], float, Optional[int]], List[str]]] = None,
        tuner_model_batch_fn: Optional[Callable[[List[str], Union[str, List[str]], float, Optional[int]], List[str]]] = None,
        embedding_model_batch_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the LLM client with user-defined model functions.
        
        If batch functions are provided, single functions are optional and will be
        auto-created from batch functions.
        
        Args:
            tuning_model_fn: Optional function for the model being optimized (single request)
            tuner_model_fn: Optional function for guidance generation (single request)
            embedding_model_fn: Optional function for generating embeddings (single request)
            tuning_model_batch_fn: Batch inference function for tuning model (preferred)
            tuner_model_batch_fn: Batch inference function for tuner model (preferred)
            embedding_model_batch_fn: Batch embedding function (preferred)
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
        """
        # Auto-create single functions from batch functions if needed
        def _create_single_from_batch(batch_fn, name):
            if batch_fn is None:
                return None
            def single_fn(prompt, system_prompt, temp, max_toks):
                result = batch_fn([prompt], system_prompt, temp, max_toks)
                return result[0] if result else None
            return single_fn
        
        # Set batch functions (preferred)
        self.tuning_model_batch_fn = tuning_model_batch_fn
        self.tuner_model_batch_fn = tuner_model_batch_fn
        self.embedding_model_batch_fn = embedding_model_batch_fn
        
        # Set single functions (use provided or auto-create from batch)
        self.tuning_model_fn = tuning_model_fn or _create_single_from_batch(
            tuning_model_batch_fn, "tuning_model"
        )
        self.tuner_model_fn = tuner_model_fn or _create_single_from_batch(
            tuner_model_batch_fn, "tuner_model"
        )
        if embedding_model_fn:
            self.embedding_model_fn = embedding_model_fn
        elif embedding_model_batch_fn:
            def _embed_fn(text):
                result = embedding_model_batch_fn([text])
                return result[0] if result else None
            self.embedding_model_fn = _embed_fn
        else:
            self.embedding_model_fn = None
        
        # Validate that at least batch or single functions are provided
        if not self.tuning_model_fn and not self.tuning_model_batch_fn:
            raise ValueError("Must provide either tuning_model_fn or tuning_model_batch_fn")
        if not self.tuner_model_fn and not self.tuner_model_batch_fn:
            raise ValueError("Must provide either tuner_model_fn or tuner_model_batch_fn")
        if not self.embedding_model_fn and not self.embedding_model_batch_fn:
            raise ValueError("Must provide either embedding_model_fn or embedding_model_batch_fn")
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_tuner_model: bool = False,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """
        Generate a response from the LLM using tuning model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            use_tuner_model: If True, use tuner_model_fn; otherwise use tuning_model_fn
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response, or None if all retries failed
        """
        model_fn = self.tuner_model_fn if use_tuner_model else self.tuning_model_fn
        
        for attempt in range(self.max_retries):
            try:
                response = model_fn(prompt, system_prompt, temperature, max_tokens)
                return response
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    # Return None instead of raising exception to allow graceful degradation
                    logger.warning(f"Failed to generate response after {self.max_retries} attempts: {str(e)}")
                    return None
    
    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[Union[str, List[str]]] = None,
        use_tuner_model: bool = False,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """
        Generate responses for multiple prompts using batch inference if available.
        
        Args:
            prompts: List of user prompts
            system_prompt: Optional system prompt (can be single string for all, or list of strings per prompt)
            use_tuner_model: If True, use tuner_model_fn; otherwise use tuning_model_fn
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            List of generated text responses
        """
        if not prompts:
            return []
        
        # Normalize system_prompts to handle both single string and list
        if isinstance(system_prompt, list):
            if len(system_prompt) != len(prompts):
                raise ValueError(f"Length of system_prompt list ({len(system_prompt)}) must match length of prompts ({len(prompts)})")
            system_prompts = system_prompt
        else:
            # Single system prompt for all
            system_prompts = system_prompt
        
        # Use batch inference function if available
        batch_fn = self.tuner_model_batch_fn if use_tuner_model else self.tuning_model_batch_fn
        
        if batch_fn is not None:
            # Use batch inference for better performance
            for attempt in range(self.max_retries):
                try:
                    responses = batch_fn(prompts, system_prompts, temperature, max_tokens)
                    if len(responses) != len(prompts):
                        raise ValueError(f"Batch function returned {len(responses)} responses for {len(prompts)} prompts")
                    return responses
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))
                    else:
                        raise Exception(f"Batch generation failed after {self.max_retries} attempts: {str(e)}")
        else:
            # Fallback to sequential generation
            responses = []
            # Convert to list if it's a single system_prompt
            if not isinstance(system_prompts, list):
                system_prompts = [system_prompts] * len(prompts)
                
            for idx, (prompt, sys_prompt) in enumerate(zip(prompts, system_prompts)):
                response = self.generate(
                    prompt=prompt,
                    system_prompt=sys_prompt,
                    use_tuner_model=use_tuner_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                # response is None if generation failed, which will be filtered out in trainer
                if response is None:
                    logger.warning(f"Case {idx} failed in sequential generation")
                responses.append(response)
            return responses
    
    def generate_with_tuner_model(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """
        Generate a response using the tuner model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response, or None if all retries failed
        """
        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            use_tuner_model=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for a text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        for attempt in range(self.max_retries):
            try:
                return self.embedding_model_fn(text)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise Exception(f"Failed to get embedding after {self.max_retries} attempts: {str(e)}")
    
    def batch_get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embedding vectors for multiple texts using batch embedding if available.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Use batch embedding function if available
        if self.embedding_model_batch_fn is not None:
            for attempt in range(self.max_retries):
                try:
                    embeddings = self.embedding_model_batch_fn(texts)
                    if len(embeddings) != len(texts):
                        raise ValueError(f"Batch embedding returned {len(embeddings)} embeddings for {len(texts)} texts")
                    return embeddings
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))
                    else:
                        raise Exception(f"Batch embedding failed after {self.max_retries} attempts: {str(e)}")
        else:
            # Fallback to sequential embedding
            embeddings = []
            for text in texts:
                embedding = self.get_embedding(text)
                embeddings.append(embedding)
            return embeddings
    
    def classify_subjects(
        self,
        questions: List[str],
        additional_context: Optional[str] = None,
    ) -> List[str]:
        """
        Classify questions into subjects using the strong model.
        Multiple questions can share the same subject (many-to-one mapping).
        
        Args:
            questions: List of questions
            additional_context: Optional additional context
            
        Returns:
            List of subjects (one per question, multiple questions can have the same subject)
        """
        # Build the classification prompt
        prompt_parts = [
        """You are an expert in categorizing questions into precise, high-relevance subjects for Retrieval-Augmented Generation (RAG).

Your goal is to assign each question a subject label (about 10-50 words) that:
- Maximizes retrieval relevance by precisely describing the problem type and solution method.
- Groups only genuinely similar questions together (same domain AND same approach).
- Avoids over-broad categories that would match unrelated problems.
- Reuses the same subject name for closely related questions.

CRITICAL: The subject must be specific enough to prevent retrieval of irrelevant guidance. Include:
1. Mathematical Domain (e.g., Combinatorics, Complex Analysis, Number Theory)
2. Problem Type (e.g., counting with constraints, roots of unity products, lifting solutions)
3. Solution Method (e.g., stars and bars, complex number identities, Hensel's lemma)

Examples of GOOD subjects (specific and discriminative):
✓ "Combinatorics: Counting arrangements in grids with row and column sum constraints using stars and bars"
✓ "Complex Analysis: Evaluating products over roots of unity using polynomial evaluation and complex identities"
✓ "Number Theory: Finding quartic congruence solutions modulo prime squares using Hensel's lemma lifting"

Examples of BAD subjects (too broad, would match unrelated problems):
✗ "modulo arithmetic" (too broad - could match any modulo problem)
✗ "number theory" (too broad - could match any number theory problem)

Key principle: If a problem involves modulo but is primarily about something else (e.g., combinatorics, complex numbers), emphasize the primary domain, not the modulo aspect.

Output only the finalized subject names."""
        ]

        if additional_context:
            prompt_parts.append(f"Additional context: {additional_context}")
            prompt_parts.append("")
        
        prompt_parts.append("Questions:")
        for i, q in enumerate(questions):
            prompt_parts.append(f"{i+1}. {q}")
        
        prompt_parts.append(f"\nProvide your response as a JSON list of subjects, one for each question (reuse subject names for similar questions, you must proivde {len(questions)} subjects).")
        prompt_parts.append('Format: ["subject1", "subject2", "subject1", ...]  # Note: subject1 appears twice for questions 1 and 3')

        prompt = "\n".join(prompt_parts)
        
        response = self.generate_with_tuner_model(
            prompt=prompt,
            temperature=0.1,  # Lower temperature for more consistent classification
        )
        
        # Handle case where classification failed
        if response is None:
            logger.warning("Failed to classify subjects, using generic fallback")
            return None
        
        # Parse the response to extract subjects
        import json
        try:
            # Try to find JSON in the response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                subjects = json.loads(json_str)
                if len(subjects) == len(questions):
                    return subjects
        except Exception:
            pass
        
        # Fallback: assign a generic subject
        return None

