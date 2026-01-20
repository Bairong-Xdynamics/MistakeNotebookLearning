"""
Example: AppWorld Task Solving with Prompt Tuning using Remote API

This example demonstrates how to use the PromptTuning framework with
remote API inference to optimize prompts for AppWorld task solving.

Features:
- API-based inference for model calls
- AppWorld task execution and evaluation
- Prompt optimization through reinforcement learning
- Knowledge base for error analysis and guidance
"""
import os
import logging
import random
random.seed(42)

from dotenv import load_dotenv
from mnl import PromptTuner
from mnl.utils import load_jsonl
from openai import OpenAI
from examples.utils.api_utils import (
    create_model_batch_fn,
    create_embedding_model_fn
)
from examples.utils.appworld_utils import create_appworld_batch_fn
from examples.utils.rewards import create_appworld_reward_fn

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Suppress httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# Load and validate environment variables
load_dotenv()

def get_env_or_raise(key):
    value = os.getenv(key)
    if not value:
        logging.warning(f"Environment variable {key} is missing in .env")
        return ""
    return value

# Tuning model configuration
TUNING_BASE_URLS = get_env_or_raise("TUNING_BASE_URLS").split(",")
TUNING_MODEL_NAME = get_env_or_raise("TUNING_MODEL_NAME")
TUNING_API_KEYS = get_env_or_raise("TUNING_MODEL_API_KEYS").split(",")

# Tuner model configuration
TUNER_BASE_URLS = get_env_or_raise("TUNER_BASE_URLS").split(",")
TUNER_MODEL_NAME = get_env_or_raise("TUNER_MODEL_NAME")
TUNER_API_KEYS = get_env_or_raise("TUNER_MODEL_API_KEYS").split(",")

# Knowledge base
KNOWLEDGE_BASE_PATH = get_env_or_raise("KNOWLEDGE_BASE_PATH")
# EMBEDDING MODEL CONFIGURATION
EMBEDDING_BASE_URL = get_env_or_raise("EMBEDDING_BASE_URL")
EMBEDDING_MODEL_NAME = get_env_or_raise("EMBEDDING_MODEL_NAME")
EMBEDDING_API_KEY = get_env_or_raise("EMBEDDING_API_KEY")

logger.info(f"Tuning model: {TUNING_MODEL_NAME} with {len(TUNING_BASE_URLS)} endpoints")
logger.info(f"Tuner model: {TUNER_MODEL_NAME} with {len(TUNER_BASE_URLS)} endpoints")

# Initialize separate clients for tuning and tuner models
tuning_clients = [
    OpenAI(api_key=TUNING_API_KEYS[i % len(TUNING_API_KEYS)], base_url=TUNING_BASE_URLS[i % len(TUNING_BASE_URLS)]) 
    for i in range(len(TUNING_BASE_URLS))
]

tuner_clients = [
    OpenAI(api_key=TUNER_API_KEYS[i % len(TUNER_API_KEYS)], base_url=TUNER_BASE_URLS[i % len(TUNER_BASE_URLS)]) 
    for i in range(len(TUNER_BASE_URLS))
]

# Config
# Data paths
train_path = "./resources/agents/appworld/appworld_test_normal_56.jsonl"
eval_path = "./resources/agents/appworld/eval_test_normal_56.jsonl"
# Knowledge base path
knowledge_base_path = KNOWLEDGE_BASE_PATH

# Training/Evaluation mode
is_train = True
is_eval = True
is_thinking = False
checkpoint_dir = "./examples/results/appworld"

# Base evaluation score - will be computed if None
base_eval_score = 14
seed = 42

# Training parameters
batch_sizes = [16]
epoch = 1
question_retrieval_top_k = 1
question_retrieval_threshold = 0.5
subject_retrieval_top_k = 3
subject_retrieval_threshold = 0.9
shuffle_batches = True
max_guidance_length = 2048
eval_steps = 100

# Model names from environment
tuning_model_name = TUNING_MODEL_NAME
tuner_model_name = TUNER_MODEL_NAME

# Token configuration
tuning_max_tokens = 4096

# Evaluation parameters
eval_retrieval_top_k = 1
eval_retrieval_threshold = 0.5
eval_batch_size = 10
eval_at_n = 1

# MLflow configuration
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", None)
mlflow_project_name = f"appworld_tuner_{tuner_model_name}_tuning_{tuning_model_name}"
mlflow_experiment_name = "AppWorld_Task_Solving"

# Guidance extraction prompt template for AppWorld
guidance_extraction_prompt_template = """
You are an expert in analyzing AppWorld task execution errors and maintaining a "mistake notebook" to improve future performance.

Subject: {subject}

Error Examples:
{error_context}

Task:
Extract insights from the AppWorld task execution mistakes and rewrite them as a structured mistake note.

Your response must include (Total output ≤ 250 words):
1. Corrected Examples with mistake trajectories (1–3 examples, each ≤ 50 words)
   - For each, include:
     • The original task instruction and mistake trajectory
     • Correct execution approach and reasoning process  

2. Correct Approach (≤ 60 words)
   - Provide the correct reasoning method or step-by-step approach that should be applied in AppWorld environment.

3. Mistake Summary (≤ 60 words)
   - Identify the root cause behind the errors (reasoning flaw, misunderstanding of app functions, missing steps, incorrect code generation, etc.).

4. Generalizable Strategy (2–4 sentences, ≤ 80 words total)
   - Summarize reusable problem-solving patterns for AppWorld tasks and how to avoid future mistakes.

5. ANTI-PATTERNS (REQUIRED - ≤ 40 words)
    List specific things to AVOID in AppWorld tasks:
    - Common ways this guidance gets misapplied
    - Situations where following this guidance would be WRONG
    - Red flags that indicate the guidance doesn't fit

Output format should resemble a mistake notebook entry: concise, structured, knowledge-focused, and reusable for similar future AppWorld tasks.
"""


guidance_merge_prompt_template = """
Synthesize guidance for AppWorld subject: {subject}

Existing guidance from related subjects:
{existing_guidance}

New guidance:
{new_guidance}

Merge into a single coherent guidance (max {max_length} chars) that:
- Combines insights from related subjects with new guidance
- Eliminates redundancy while preserving key information and examples of the mistakes
- **Preserves and emphasizes applicability conditions** - clearly state when each method applies to AppWorld tasks
- Focuses on actionable advice for code generation and environment interaction
- Maintains consistent style
- **Includes warnings about when NOT to apply the guidance** to avoid misapplication in AppWorld

Merged guidance:
"""


logger.info(
    f"Knowledge base path: {knowledge_base_path}\n"
    f"Tuning model name: {tuning_model_name}\n"
    f"Tuner model name: {tuner_model_name}\n"
    f"Batch sizes: {batch_sizes}\n"
    f"Epoch: {epoch}\n"
    f"Question retrieval top k: {question_retrieval_top_k}\n"
    f"Question retrieval threshold: {question_retrieval_threshold}\n"
    f"Subject retrieval top k: {subject_retrieval_top_k}\n"
    f"Subject retrieval threshold: {subject_retrieval_threshold}\n"
    f"Eval retrieval top k: {eval_retrieval_top_k}\n"
    f"Eval retrieval threshold: {eval_retrieval_threshold}\n"
    f"Eval batch size: {eval_batch_size}\n"
    f"Eval at n: {eval_at_n}\n"
    f"MLflow tracking uri: {mlflow_tracking_uri}\n"
    f"MLflow project name: {mlflow_project_name}\n"
    f"MLflow experiment name: {mlflow_experiment_name}\n"
    f"Max tokens: {tuning_max_tokens}\n "
    f"Seed: {seed}\n"
    f"Shuffle batches: {shuffle_batches}\n"
    f"Max guidance length: {max_guidance_length}\n"
    f"Eval steps: {eval_steps}\n"
    f"Training mode: {is_train}\n"
    f"Evaluation mode: {is_eval}\n"
    f"Thinking mode: {is_thinking}\n"
)

# Create batch model functions
extra_body = {
    "chat_template_kwargs": {"enable_thinking": is_thinking},
    "top_k": 20
}

# Create AppWorld batch function
num_workers = len(tuning_clients)*8
tuning_model_batch_fn = create_appworld_batch_fn(
    clients=tuning_clients,
    model_name=tuning_model_name,
    temperature=0.0,
    max_tokens=tuning_max_tokens,
    presence_penalty=1.5,
    top_p=0.9,
    extra_body=extra_body,
    num_workers=num_workers,
    seed=seed
)

# Create tuner model batch function
tuner_model_batch_fn = create_model_batch_fn(
    clients=tuner_clients,
    model=tuner_model_name,
    temperature=0.6,
    max_tokens=2048,
    top_p=0.9,
    presence_penalty=1.0,
    extra_body=extra_body,
    seed=seed
)

# Create embedding function
embedding_model_fn = create_embedding_model_fn(
    api_base=EMBEDDING_BASE_URL,
    api_key=EMBEDDING_API_KEY,
    model=EMBEDDING_MODEL_NAME
)

user_defined_reward_fn = create_appworld_reward_fn(tuner_model_batch_fn)

test_data = load_jsonl(eval_path)
batch_size_results: dict = {}

# Initialize PromptTuner
tuner = PromptTuner(
    reward_fn=user_defined_reward_fn,
    tuning_model_batch_fn=tuning_model_batch_fn,
    tuner_model_batch_fn=tuner_model_batch_fn,
    embedding_model_fn=embedding_model_fn,
    mlflow_tracking_uri=mlflow_tracking_uri,
    mlflow_project_name=mlflow_project_name,
    mlflow_experiment_name=mlflow_experiment_name,
    eval_steps=eval_steps,
    knowledge_base_path=knowledge_base_path,
    question_retrieval_top_k=question_retrieval_top_k,
    question_retrieval_threshold=question_retrieval_threshold,
    subject_retrieval_top_k=subject_retrieval_top_k,
    subject_retrieval_threshold=subject_retrieval_threshold,
    eval_retrieval_top_k=eval_retrieval_top_k,
    eval_retrieval_threshold=eval_retrieval_threshold,
    eval_batch_size=eval_batch_size,
    guidance_extraction_prompt_template=guidance_extraction_prompt_template,
    guidance_merge_prompt_template=guidance_merge_prompt_template,
    shuffle_batches=shuffle_batches,
    max_guidance_length=max_guidance_length,
    eval_at_n=eval_at_n
)

# Initialize a temporary tuner for baseline evaluation if needed
if base_eval_score is None:
    tuner.knowledge_base.entries = []
    wrong_cases_path = os.path.join(
        checkpoint_dir, f'wrong_cases_raw_tuning_api_{tuning_model_name.split("/")[-1]}_{tuner_model_name.split("/")[-1]}.jsonl'
    ) if checkpoint_dir else None
    base_eval_score = tuner._evaluate_on_eval_set(
        test_data,
        save_wrong_cases_path=wrong_cases_path,
        is_retrieval_subject=False,
        eval_at_n=eval_at_n
    )
    logger.info(f"base eval score: {base_eval_score}")

if is_train:
    for batch_size in batch_sizes:
        logger.info(f"Starting training with batch size: {batch_size}")
        checkpoint_dir_current = os.path.join(
            checkpoint_dir, 
            f"tuning_{tuning_model_name.split('/')[-1]}_tuner_{tuner_model_name.split('/')[-1]}_appworld_batch{batch_size}_epoch{epoch}_is_thinking_{is_thinking}"
        )
        # Reset knowledge base and step count for each batch size
        tuner.knowledge_base.entries = []
        tuner.step_count = 0
        tuner.batch_size = batch_size
        tuner.mlflow_project_name = mlflow_project_name
        tuner.train(
            train_data_path=train_path,
            eval_data_path=eval_path,
            num_epochs=epoch,
            checkpoint_dir=checkpoint_dir_current,
            is_retrieval_subject=False,
            save_wrong_cases_path=checkpoint_dir_current
        )

# Final evaluation
if is_eval:
    wrong_cases_path = os.path.join(checkpoint_dir, 'wrong_cases_eval.jsonl')
    eval_score = tuner._evaluate_on_eval_set(
        test_data,
        save_wrong_cases_path=wrong_cases_path,
        is_retrieval_subject=False,
        eval_at_n=eval_at_n
    )
    logger.info(f"Final eval score: {eval_score}")