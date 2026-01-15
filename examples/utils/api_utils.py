"""
Unified API utilities for model inference.

This module provides standardized functions for creating model inference wrappers
that can be used with the PromptTuner framework.
"""

from typing import Optional, List, Union, Dict, Any, Callable
from openai import OpenAI
import logging
import time
from .async_batch_client import batch_request_with_messages_sync

logger = logging.getLogger(__name__)


def _sequential_request_with_messages(
    client: OpenAI,
    messages_list: List[List[Dict[str, str]]],
    model: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    sleep_time: float = 0.0,
    max_retries: int = 3,
    **kwargs
) -> List[str]:
    """
    Make sequential (non-parallel) API requests one by one with optional sleep between requests.
    
    This function processes requests sequentially to avoid concurrency issues with commercial APIs
    that don't support parallel requests.
    
    Args:
        client: OpenAI client instance
        messages_list: List of message lists, one for each request
        model: Model name/path
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        extra_body: Additional parameters for API call
        sleep_time: Sleep time in seconds between requests (default: 0.0)
        max_retries: Maximum number of retry attempts (default: 3)
        **kwargs: Additional parameters for API call
    
    Returns:
        List of generated responses in the same order as input messages_list
    """
    if not messages_list:
        return []
    
    responses = []
    api_kwargs = {"temperature": temperature, "max_tokens": max_tokens}
    if extra_body:
        api_kwargs["extra_body"] = extra_body
    api_kwargs.update(kwargs)
    
    for idx, messages in enumerate(messages_list):
        # Sleep before request (except for the first one)
        if idx > 0 and sleep_time > 0:
            time.sleep(sleep_time)
        
        # Retry logic
        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **api_kwargs
                )
                content = response.choices[0].message.content
                responses.append(content.strip() if content else "")
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        f"Request {idx} failed on attempt {attempt + 1}/{max_retries + 1}: "
                        f"{type(e).__name__}: {str(e)}. Retrying..."
                    )
                    # Sleep before retry
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                else:
                    logger.error(
                        f"Request {idx} failed on all {max_retries + 1} attempts. "
                        f"Last error: {type(e).__name__}: {str(e)}. Returning empty string."
                    )
                    responses.append("")  # Return empty string on failure
    
    return responses


def create_model_batch_fn(
    clients: List[OpenAI],
    model: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    prompt_postprocessor: Optional[Callable[[str], str]] = None,
    system_prompt_processor: Optional[Callable[[str, str], str]] = None,
    sequential: bool = False,
    sleep_time: float = 0.0,
    **kwargs
) -> Callable[[List[str], Union[str, List[str]], float, Optional[int]], List[str]]:
    """
    Create a batch model function using async batch client or sequential requests.
    
    Args:
        clients: List of OpenAI client instances for parallel processing (or single client for sequential)
        model: Model name/path
        temperature: Default temperature
        max_tokens: Default max tokens
        extra_body: Additional parameters for API call
        prompt_postprocessor: Optional function to modify prompts before sending
        system_prompt_processor: Optional function to modify system prompts
        sequential: If True, process requests sequentially (one by one) instead of in parallel.
                    This is useful for commercial APIs that don't support concurrency.
                    Default: False (use parallel async processing)
        sleep_time: Sleep time in seconds between requests when sequential=True (default: 0.0)
        **kwargs: Additional parameters for API call (e.g., top_p, presence_penalty, frequency_penalty)
    
    Returns:
        Function with signature (prompts, system_prompts, temperature, max_tokens) -> List[str]
    """
    def tuning_model_batch_fn(
        prompts: List[str],
        system_prompts: Union[str, List[str]],
        temp: float,
        max_toks: Optional[int]
    ) -> List[str]:
        # Normalize system_prompts to list
        if isinstance(system_prompts, str):
            system_prompts = [system_prompts] * len(prompts)
        
        # Apply postprocessing
        processed_prompts = prompts
        if prompt_postprocessor:
            processed_prompts = [prompt_postprocessor(p) for p in prompts]
        
        # Build messages list
        messages_list = []
        system_prompts = ['' for _ in range(len(prompts))] if system_prompts is None else system_prompts
        for prompt, sys_prompt in zip(processed_prompts, system_prompts):
            messages = []
            if sys_prompt or system_prompt_processor:
                if system_prompt_processor:
                    sys_prompt = system_prompt_processor(sys_prompt, prompt)
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": prompt})
            messages_list.append(messages)
        
        # Use provided temperature/max_tokens or defaults
        final_temp = temp if temp is not None else temperature
        final_max_tokens = max_toks if max_toks is not None else max_tokens
        
        # Choose between sequential or parallel processing
        if sequential:
            # Use sequential processing (one by one) for APIs that don't support concurrency
            # Use the first client for sequential requests
            if not clients:
                raise ValueError("At least one client must be provided for sequential processing")
            client = clients[0]
            responses = _sequential_request_with_messages(
                client=client,
                messages_list=messages_list,
                model=model,
                temperature=final_temp,
                max_tokens=final_max_tokens,
                extra_body=extra_body,
                sleep_time=sleep_time,
                **kwargs
            )
        else:
            # Use parallel async processing (default behavior)
            responses = batch_request_with_messages_sync(
                clients=clients,
                messages_list=messages_list,
                model=model,
                temperature=final_temp,
                max_tokens=final_max_tokens,
                extra_body=extra_body,
                **kwargs
            )
        
        return [r.strip() for r in responses]
    
    return tuning_model_batch_fn

def create_embedding_model_fn(
    api_base: str = "http://127.0.0.1:10013/v1",
    api_key: str = "EMPTY",
    model: str = "bge-m3",
) -> Callable[[str], List[float]]:
    """
    Create an embedding model function.
    
    Args:
        api_base: API base URL for embedding service
        api_key: API key
        model: Embedding model name/path
    
    Returns:
        Function with signature (text: str) -> List[float]
    """
    client = OpenAI(api_key=api_key, base_url=api_base)
    
    def embedding_model_fn(text: str) -> List[float]:
        response = client.embeddings.create(
            model=model,
            input=[text]
        )
        return response.data[0].embedding
    
    return embedding_model_fn