"""
Utility functions for the prompt tuning framework.

Includes: cosine similarity, JSONL I/O, MLflow helpers, and token counting.
"""

import json
import os
from typing import List, Dict, Any, Optional
import numpy as np
import tiktoken
import mlflow


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score between -1 and 1
    """
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    
    # Handle zero vectors
    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(vec1_np, vec2_np) / (norm1 * norm2))


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries, one per line
    """
    data = []
    if not os.path.exists(file_path):
        return data
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str, append: bool = False) -> None:
    """
    Save data to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path to the JSONL file
        append: If True, append to existing file; otherwise overwrite
    """
    mode = 'a' if append else 'w'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    
    with open(file_path, mode, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: Text to count tokens for
        model: Model name to use for encoding (default: gpt-3.5-turbo)
        
    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Default to cl100k_base encoding if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))


def truncate_text(text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> str:
    """
    Truncate text to a maximum number of tokens.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        model: Model name to use for encoding
        
    Returns:
        Truncated text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


def setup_mlflow(
    tracking_uri: str,
    experiment_name: str,
    project_name: Optional[str] = None
) -> Optional[str]:
    """
    Setup MLflow experiment tracking.
    
    Args:
        tracking_uri: MLflow tracking server URI (if empty, MLflow is disabled)
        experiment_name: Name of the experiment
        project_name: Optional project/run name
        
    Returns:
        Experiment ID if MLflow is enabled, None otherwise
    """
    # Skip MLflow setup if tracking URI is empty or None
    if not tracking_uri or not tracking_uri.strip():
        return None
        
    mlflow.set_tracking_uri(tracking_uri)
    
    # Get or create experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    return experiment_id


def log_metrics_to_mlflow(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Log metrics to MLflow.
    
    Args:
        metrics: Dictionary of metric names and values
        step: Optional step number
    """
    try:
        # Check if MLflow is active (has an active run)
        if mlflow.active_run() is not None:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
    except Exception:
        # Silently ignore MLflow errors when it's not properly configured
        pass


def log_artifact_to_mlflow(local_path: str, artifact_path: Optional[str] = None) -> None:
    """
    Log an artifact (file) to MLflow.
    
    Args:
        local_path: Local path to the artifact
        artifact_path: Optional path within the artifact store
    """
    try:
        # Check if MLflow is active (has an active run)
        if mlflow.active_run() is not None:
            mlflow.log_artifact(local_path, artifact_path)
    except Exception:
        # Silently ignore MLflow errors when it's not properly configured
        pass


def log_text_to_mlflow(text: str, filename: str) -> None:
    """
    Log text content as an artifact to MLflow.
    
    Args:
        text: Text content to log
        filename: Name of the file to create
    """
    try:
        # Check if MLflow is active (has an active run)
        if mlflow.active_run() is not None:
            # Create temporary file
            temp_path = f"/tmp/{filename}"
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            mlflow.log_artifact(temp_path)
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    except Exception:
        # Silently ignore MLflow errors when it's not properly configured
        pass


def batch_data(data: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split data into batches.
    
    Args:
        data: List of data items
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(data), batch_size):
        batches.append(data[i:i + batch_size])
    return batches

