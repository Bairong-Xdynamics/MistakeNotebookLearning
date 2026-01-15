"""
SGD-Based Prompt Auto-Tuning Framework

A framework for automatically optimizing LLM prompts using SGD-inspired batch processing,
dynamic subject classification, and RAG-based knowledge management.
"""

__version__ = "0.1.0"

from .trainer import PromptTuner
from .knowledge_base import KnowledgeBase
from .evaluator import Evaluator
from .prompt_builder import PromptBuilder
from .llm_client import LLMClient

__all__ = [
    "PromptTuner",
    "KnowledgeBase",
    "Evaluator",
    "PromptBuilder",
    "LLMClient",
]

