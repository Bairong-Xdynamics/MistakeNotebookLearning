"""
Async batch client utility for parallel API requests with multiple clients.

This module provides functions to perform asynchronous batch API requests
using multiple OpenAI clients in parallel for improved performance.
"""

import asyncio
from typing import List, Optional, Union, Dict, Any
from openai import OpenAI, AsyncOpenAI
import logging

logger = logging.getLogger(__name__)


async def _async_request_single(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    timeout: float = 360.0,
    **kwargs
) -> str:
    """
    Make a single async API request with timeout.
    
    Args:
        client: AsyncOpenAI client instance
        model: Model name/path
        messages: List of message dictionaries
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds (default: 120.0 = 2 minutes)
        **kwargs: Additional parameters for API call
    
    Returns:
        Generated text response
    
    Raises:
        asyncio.TimeoutError: If request times out
        Exception: Other exceptions from API call
    """
    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        ),
        timeout=timeout
    )
    return response.choices[0].message.content.strip()


async def _async_request(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    timeout: float = 360.0,
    max_retries: int = 3,
    **kwargs
) -> str:
    """
    Make a single async API request with timeout and retry logic.
    
    This function will retry up to max_retries times if the request fails or times out.
    Each retry will wait for the full timeout period before timing out again.
    
    Args:
        client: AsyncOpenAI client instance
        model: Model name/path
        messages: List of message dictionaries
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds (default: 120.0 = 2 minutes)
        max_retries: Maximum number of retry attempts (default: 3)
        **kwargs: Additional parameters for API call
    
    Returns:
        Generated text response, or empty string if all retries failed
    
    Note:
        If all retries fail, the function returns an empty string instead of raising
        an exception to allow batch processing to continue with other requests.
    """
    for attempt in range(max_retries + 1):  # +1 because first attempt is not a retry
        try:
            response = await _async_request_single(
                client=client,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs
            )
            # Success - return immediately
            if attempt > 0:
                logger.info(f"Request succeeded on attempt {attempt + 1}/{max_retries + 1}")
            return response
            
        except asyncio.TimeoutError:
            if attempt < max_retries:
                logger.warning(
                    f"Request timed out after {timeout}s on attempt {attempt + 1}/{max_retries + 1}. "
                    f"Retrying..."
                )
            else:
                logger.error(
                    f"Request timed out after {timeout}s on all {max_retries + 1} attempts. "
                    f"Giving up."
                )
                
        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    f"Request failed on attempt {attempt + 1}/{max_retries + 1}: {type(e).__name__}: {str(e)}. "
                    f"Retrying..."
                )
            else:
                logger.error(
                    f"Request failed on all {max_retries + 1} attempts. "
                    f"Last error: {type(e).__name__}: {str(e)}. Giving up."
                )
    
    # All retries failed - return empty string to allow batch processing to continue
    logger.error(f"All {max_retries + 1} attempts failed. Returning empty string.")
    return ""


def _sync_to_async_client(sync_client: OpenAI) -> AsyncOpenAI:
    """
    Convert a synchronous OpenAI client to an async client.
    
    Args:
        sync_client: Synchronous OpenAI client
    
    Returns:
        AsyncOpenAI client with same configuration
    """
    # Extract configuration from sync client
    api_key = sync_client.api_key
    base_url = sync_client.base_url
    
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url
    )


async def async_batch_request_with_messages(
    clients: List[OpenAI],
    messages_list: List[List[Dict[str, str]]],
    model: str = "/opt/nlp/MODELS/Qwen/Qwen3-8B",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs
) -> List[str]:
    """
    Perform asynchronous batch requests using multiple clients with custom messages.
    
    This function allows you to provide custom message lists for each request,
    which is useful when you need to add context or modify messages per prompt.
    
    Args:
        clients: List of OpenAI client instances (can be sync or async)
        messages_list: List of message lists, one for each request
        model: Model name/path to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        extra_body: Additional parameters for API call
        **kwargs: Additional parameters passed to API call
    
    Returns:
        List of generated responses in the same order as input messages_list
    
    Example:
        >>> clients = [OpenAI(api_key="key1", base_url="url1")]
        >>> messages_list = [
        ...     [{"role": "user", "content": "What is 2+2?"}],
        ...     [{"role": "user", "content": "What is 3+3?"}]
        ... ]
        >>> responses = await async_batch_request_with_messages(clients, messages_list)
    """
    if not clients:
        raise ValueError("At least one client must be provided")
    
    if not messages_list:
        return []
    
    # Convert sync clients to async clients
    # Track which clients we created (need to close them)
    async_clients = []
    created_clients = []  # Track clients we created from sync clients
    for client in clients:
        if isinstance(client, AsyncOpenAI):
            async_clients.append(client)
        elif isinstance(client, OpenAI):
            async_client = _sync_to_async_client(client)
            async_clients.append(async_client)
            created_clients.append(async_client)
        else:
            raise TypeError(f"Unsupported client type: {type(client)}")
    
    try:
        # Prepare tasks
        tasks = []
        for messages in messages_list:
            # Round-robin distribution of tasks across clients
            client_idx = len(tasks) % len(async_clients)
            client = async_clients[client_idx]
            
            # Prepare API call parameters
            api_kwargs = {"temperature": temperature, "max_tokens": max_tokens}
            if extra_body:
                api_kwargs["extra_body"] = extra_body
            api_kwargs.update(kwargs)
            
            # Create async task
            task = _async_request(
                client=client,
                model=model,
                messages=messages,
                **api_kwargs
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        results = []
        for idx, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Request {idx} failed: {response}")
                results.append("")  # Return empty string on error
            else:
                results.append(response)
        
        return results
    finally:
        # Close all clients we created before event loop closes
        # This prevents "Event loop is closed" errors
        if created_clients:
            try:
                # Close all clients concurrently with shield to protect from cancellation
                # AsyncOpenAI uses close() method (not aclose())
                close_tasks = [asyncio.shield(client.close()) for client in created_clients]
                # Wait for close operations with timeout
                await asyncio.wait_for(
                    asyncio.gather(*close_tasks, return_exceptions=True),
                    timeout=2.0
                )
                # Give httpx internal cleanup tasks time to complete
                # httpx may create background tasks that need to run after close()
                await asyncio.sleep(0.05)
            except asyncio.TimeoutError:
                logger.warning("Timeout while closing async clients, but continuing...")
            except Exception as e:
                logger.warning(f"Error closing async clients: {e}")


async def async_batch_request(
    clients: List[OpenAI],
    prompts: List[str],
    system_prompts: Union[str, List[str], None] = None,
    model: str = "/opt/nlp/MODELS/Qwen/Qwen3-8B",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs
) -> List[str]:
    """
    Perform asynchronous batch requests using multiple clients in parallel.
    
    This function distributes prompts across multiple clients and processes
    them concurrently for improved performance.
    
    Args:
        clients: List of OpenAI client instances (can be sync or async)
        prompts: List of prompt strings to process
        system_prompts: System prompt(s) - can be a single string or list
        model: Model name/path to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        extra_body: Additional parameters for API call
        **kwargs: Additional parameters passed to API call
    
    Returns:
        List of generated responses in the same order as input prompts
    
    Example:
        >>> clients = [OpenAI(api_key="key1", base_url="url1"),
        ...            OpenAI(api_key="key2", base_url="url2")]
        >>> prompts = ["What is 2+2?", "What is 3+3?"]
        >>> responses = await async_batch_request(clients, prompts)
    """
    if not clients:
        raise ValueError("At least one client must be provided")
    
    if not prompts:
        return []
    
    # Convert system_prompts to list format
    if system_prompts is None:
        system_prompts = [None] * len(prompts)
    elif isinstance(system_prompts, str):
        system_prompts = [system_prompts] * len(prompts)
    elif len(system_prompts) != len(prompts):
        raise ValueError(f"system_prompts length ({len(system_prompts)}) must match prompts length ({len(prompts)})")
    
    # Convert sync clients to async clients
    # Track which clients we created (need to close them)
    async_clients = []
    created_clients = []  # Track clients we created from sync clients
    for client in clients:
        if isinstance(client, AsyncOpenAI):
            async_clients.append(client)
        elif isinstance(client, OpenAI):
            async_client = _sync_to_async_client(client)
            async_clients.append(async_client)
            created_clients.append(async_client)
        else:
            raise TypeError(f"Unsupported client type: {type(client)}")
    
    try:
        # Prepare messages for each prompt
        tasks = []
        for prompt, sys_prompt in zip(prompts, system_prompts):
            messages = []
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Round-robin distribution of tasks across clients
            client_idx = len(tasks) % len(async_clients)
            client = async_clients[client_idx]
            
            # Prepare API call parameters
            api_kwargs = {"temperature": temperature, "max_tokens": max_tokens}
            if extra_body:
                api_kwargs["extra_body"] = extra_body
            api_kwargs.update(kwargs)
            
            # Create async task
            task = _async_request(
                client=client,
                model=model,
                messages=messages,
                **api_kwargs
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        results = []
        for idx, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Request {idx} failed: {response}")
                results.append("")  # Return empty string on error
            else:
                results.append(response)
        
        return results
    finally:
        # Close all clients we created before event loop closes
        # This prevents "Event loop is closed" errors
        if created_clients:
            try:
                # Close all clients concurrently with shield to protect from cancellation
                # AsyncOpenAI uses close() method (not aclose())
                close_tasks = [asyncio.shield(client.close()) for client in created_clients]
                # Wait for close operations with timeout
                await asyncio.wait_for(
                    asyncio.gather(*close_tasks, return_exceptions=True),
                    timeout=2.0
                )
                # Give httpx internal cleanup tasks time to complete
                # httpx may create background tasks that need to run after close()
                await asyncio.sleep(0.05)
            except asyncio.TimeoutError:
                logger.warning("Timeout while closing async clients, but continuing...")
            except Exception as e:
                logger.warning(f"Error closing async clients: {e}")


def batch_request_with_messages_sync(
    clients: List[OpenAI],
    messages_list: List[List[Dict[str, str]]],
    model: str = "/opt/nlp/MODELS/Qwen/Qwen3-8B",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs
) -> List[str]:
    """
    Synchronous wrapper for async_batch_request_with_messages.
    
    This function ensures all cleanup tasks complete before the event loop closes,
    preventing "Event loop is closed" errors from httpx cleanup tasks.
    
    Args:
        clients: List of OpenAI client instances
        messages_list: List of message lists, one for each request
        model: Model name/path to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        extra_body: Additional parameters for API call
        **kwargs: Additional parameters passed to API call
    
    Returns:
        List of generated responses in the same order as input messages_list
    """
    async def _run_with_cleanup():
        try:
            return await async_batch_request_with_messages(
                clients=clients,
                messages_list=messages_list,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
                **kwargs
            )
        finally:
            # Ensure all pending tasks complete before event loop closes
            # This is critical to prevent "Event loop is closed" errors
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Get all pending tasks (excluding the current one)
                current_task = asyncio.current_task(loop)
                pending = [t for t in asyncio.all_tasks(loop) 
                          if not t.done() and t is not current_task]
                if pending:
                    try:
                        # Wait for pending tasks with timeout
                        await asyncio.wait_for(
                            asyncio.gather(*pending, return_exceptions=True),
                            timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        # If timeout, cancel remaining tasks
                        for task in pending:
                            if not task.done():
                                task.cancel()
                        # Wait a bit more for cancellation to complete
                        await asyncio.sleep(0.1)
    
    return asyncio.run(_run_with_cleanup())


def batch_request_sync(
    clients: List[OpenAI],
    prompts: List[str],
    system_prompts: Union[str, List[str], None] = None,
    model: str = "/opt/nlp/MODELS/Qwen/Qwen3-8B",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs
) -> List[str]:
    """
    Synchronous wrapper for async_batch_request.
    
    This function ensures all cleanup tasks complete before the event loop closes,
    preventing "Event loop is closed" errors from httpx cleanup tasks.
    
    Args:
        clients: List of OpenAI client instances
        prompts: List of prompt strings to process
        system_prompts: System prompt(s) - can be a single string or list
        model: Model name/path to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        extra_body: Additional parameters for API call
        **kwargs: Additional parameters passed to API call
    
    Returns:
        List of generated responses in the same order as input prompts
    
    Example:
        >>> clients = [OpenAI(api_key="key1", base_url="url1"),
        ...            OpenAI(api_key="key2", base_url="url2")]
        >>> prompts = ["What is 2+2?", "What is 3+3?"]
        >>> responses = batch_request_sync(clients, prompts)
    """
    async def _run_with_cleanup():
        try:
            return await async_batch_request(
                clients=clients,
                prompts=prompts,
                system_prompts=system_prompts,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
                **kwargs
            )
        finally:
            # Ensure all pending tasks complete before event loop closes
            # This is critical to prevent "Event loop is closed" errors
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Get all pending tasks (excluding the current one)
                current_task = asyncio.current_task(loop)
                pending = [t for t in asyncio.all_tasks(loop) 
                          if not t.done() and t is not current_task]
                if pending:
                    try:
                        # Wait for pending tasks with timeout
                        await asyncio.wait_for(
                            asyncio.gather(*pending, return_exceptions=True),
                            timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        # If timeout, cancel remaining tasks
                        for task in pending:
                            if not task.done():
                                task.cancel()
                        # Wait a bit more for cancellation to complete
                        await asyncio.sleep(0.1)
    
    return asyncio.run(_run_with_cleanup())


if __name__ == "__main__":
    API_BASE_URL_1 = os.getenv("API_BASE_URL_1", "http://localhost:8000/v1")
    API_BASE_URL_2 = os.getenv("API_BASE_URL_2", "http://localhost:8001/v1")
    API_KEY = os.getenv("API_KEY", "EMPTY")
    MODEL_PATH = os.getenv("MODEL_PATH", "your-model-path-here")
    
    clients = [
        OpenAI(api_key=API_KEY, base_url=API_BASE_URL_1),
        OpenAI(api_key=API_KEY, base_url=API_BASE_URL_2),
    ]

    messages_list = [
        [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "Solve: 15 * 23 = ?"}
        ],
        [
            {"role": "system", "content": "You are a geography expert."},
            {"role": "user", "content": "Name three major rivers in China."}
        ],
    ]
    try:
        responses = batch_request_with_messages_sync(
            clients=clients,
            messages_list=messages_list,
            model=MODEL_PATH,
            temperature=0.0,
        )
        for i, (messages, response) in enumerate(zip(messages_list, responses), 1):
            print(f"\nRequest {i}:")
            print(f"  Messages: {messages}")
            print(f"  Response: {response[:100]}..." if len(response) > 100 else f"  Response: {response}")
    except Exception as e:
        print(f"Test 3 failed: {e}")