"""
Example: Mind2Web Web Navigation with Online Learning using Remote API

This example demonstrates how to use the PromptTuning framework with
remote API inference to optimize prompts for Mind2Web web navigation tasks.

Benefits of using Remote API:
- No need to load models locally
- Can leverage remote GPU resources
- Easy to scale with multiple API endpoints
- No local GPU memory requirements

Requirements:
- pip install openai
- Access to remote API endpoint
"""
import random
random.seed(42)

import os
import logging
from dotenv import load_dotenv
from mnl import PromptTuner
from mnl.utils import load_jsonl
from openai import OpenAI
from examples.utils.api_utils import (
    create_model_batch_fn,
    create_embedding_model_fn
)
from examples.utils.rewards import create_mind2web_reward_fn



fewshot_prompt = """You are a helpful assistant that is great at website design, navigation, and executing tasks for the user.

Here are some examples and you must follow the Output Formate:

Example 1:
Input:
'''
<html> <div> <div> <a tock home page /> <button id=0 book a reservation. toggle open> <span> Book a reservation </span> </button> <button book a reservation. toggle open> </button> </div> <div> <select id=1 type> <option reservations true> Dine in </option> <option pickup> Pickup </option> <option delivery> Delivery </option> <option events> Events </option> <option wineries> Wineries </option> <option all> Everything </option> </select> <div id=2> <p> Celebrating and supporting leading women shaking up the industry. </p> <span> Explore now </span> </div> </div> </div> </html>
'''

Based on the HTML webpage above, try to complete the following task:
Task: Check for pickup restaurant available in Boston, NY on March 18, 5pm with just one guest
Previous actions:
None
What should be the next action? Please select from the following choices (If the correct action is not in the page above, please select A. 'None of the above'):

A. None of the above
B. <button id=0 book a reservation. toggle open> <span> Book a
C. <select id=1 type> <option reservations true> Dine in </option> <option
D. <div id=2> <p> Celebrating and supporting leading women shaking up

Output Formate:
Answer: C.
Action: SELECT
Value: Pickup

Example 2:
Input:
'''
<html> <div> <main main> <section tabpanel> <div> <ul tablist> <li tab heading level 3 search and> </li> <li id=0 tab heading level 3 search and> <span> Hotel </span> </li> <li tab heading level 3 search and> </li> <li tab heading level 3 search and> </li> </ul> <div tabpanel> <div id=1> <div> <span> Dates* </span> <button button clear dates /> </div> <div> <label> Travelers </label> <div> <p> 1 Adult </p> <button button> 1 Adult </button> <div dialog> <button button travel with a pet. this> <span> Travel with a pet </span> </button> <div> <button button clear all fields> Clear all </button> <button button> </button> </div> </div> </div> </div> </div> </div> </div> </section> </main> <footer contentinfo> <div> <h3> Stay Connected </h3> <ul id=2> <a mobile tools> </a> <a open united's tiktok feed in> </a> <a open united's facebook page in> </a> <a open united's twitter feed in> </a> <a open united's youtube page in> </a> <a open united's instagram feed in> </a> <a open united's linkedin profile in> </a> </ul> </div> </footer> </div> </html>
'''

Based on the HTML webpage above, try to complete the following task:
Task: Compare the fare types to book a 1-adult ticket from Springfiels, IL to Austin, TX for April 29th 2023
Previous actions:
[combobox]  Enter your departing city, airport name, or airpor... -> TYPE: SPRINGFIELD
[button]  Springfield, IL, US (SPI) -> CLICK
[combobox]  Enter your destination city, airport name, or airp... -> TYPE: AUSTIN
[button]  Austin, TX, US (AUS) -> CLICK
What should be the next action? Please select from the following choices (If the correct action is not in the page above, please select A. 'None of the above'):

A. None of the above
B. <li id=0 tab heading level 3 search and> <span> Hotel
C. <div id=1> <div> <span> Dates* </span> <button button clear dates
D. <ul id=2> <a mobile tools> </a> <a open united's tiktok

Output Formate:
Answer: A.

Example 3:
Input:
'''
<html> <div> <nav main menu> <ul> <li> <div button> Car Sales </div> <div id=0> <div> <div> <div> Buy A Car </div> <div> Plan Your Purchase </div> </div> <div> <h4> Its Tax Refund Time. Treat Yourself to an Upgrade. </h4> <p> With a variety of options, invest your refund in what you really want - a quality, used vehicle from Enterprise. </p> <a> View Inventory </a> </div> </div> </div> </li> <div id=1> Enterprise Fleet Management </div> </ul> </nav> <div region> <button id=2 selected pick-up date 03/19/2023> <span> <span> 19 </span> <div> <span> Mar </span> <span> 2023 </span> </div> </span> </button> </div> </div> </html>
'''

Based on the HTML webpage above, try to complete the following task:
Task: Find a mini van at Brooklyn City from April 5th to April 8th for a 22 year old renter.
Previous actions:
[searchbox]  Pick-up & Return Location (ZIP, City or Airport) (... -> TYPE: Brooklyn
[option]  Brooklyn, NY, US Select -> CLICK
What should be the next action? Please select from the following choices (If the correct action is not in the page above, please select A. 'None of the above'):

A. None of the above
B. <div id=0> <div> <div> <div> Buy A Car </div> <div>
C. <div id=1> Enterprise Fleet Management </div>
D. <button id=2 selected pick-up date 03/19/2023> <span> <span> 19 </span>

Output Formate:
Answer: D.
Action: CLICK
"""


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
        raise ValueError(f"Environment variable {key} is missing in .env")
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
train_path = "./resources/agents/mind2web_topk20_100_train.jsonl"
eval_path = "./resources/agents/mind2web_topk20_100_eval.jsonl"

# Knowledge base path
knowledge_base_path = KNOWLEDGE_BASE_PATH

# Training/Evaluation mode
is_train = False
is_thinking = False
checkpoint_dir = "./examples/results/mind2web"

# Base evaluation score - will be computed if None
base_eval_score = None
seed = 42

# Training parameters
batch_sizes = [16]
epoch = 1
question_retrieval_top_k = 1
question_retrieval_threshold = 0.6
subject_retrieval_top_k = 3
subject_retrieval_threshold = 0.9
shuffle_batches = True
max_guidance_length = 2048
eval_steps = 200

# Model names from environment
tuning_model_name = TUNING_MODEL_NAME
tuner_model_name = TUNER_MODEL_NAME

# Token configuration
tuning_max_tokens = 8192

# Evaluation parameters
eval_retrieval_top_k = 3
eval_retrieval_threshold = 0.6
eval_batch_size = 16
eval_at_n = 1

# MLflow configuration
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", None)
mlflow_project_name = f"tuner_{tuner_model_name}_tuning_{tuning_model_name}_mind2web"
mlflow_experiment_name = "Mind2Web_Online_Learning"

# Guidance extraction prompt template for web navigation
guidance_extraction_prompt_template = """
You are an expert in web navigation and user interface interaction.

Subject: {subject}

Interaction Examples (may include mistakes or correct actions):
{error_context}

Task:
Analyze the interactions (both correct and incorrect) and create structured guidance for web automation.

Your response must include (Total output ≤ 250 words):
1. Case Analysis (1–3 examples, each ≤ 50 words)
   - Label: Correct or Incorrect
   - The original question and mistake answer 
   - Why it is correct/incorrect
   - If incorrect, provide the correct action and reasoning

2. Web UI Best Practices (≤ 60 words)
   - Guidelines for identifying interactive elements
   - How to interpret page structure and context

3. Task-Specific Strategies (≤ 60 words)
   - Patterns in successful navigation for this task type
   - Key elements to prioritize

4. Generalizable Patterns (2–4 sentences, ≤ 80 words total)
   - Reusable navigation strategies
   - How to handle common UI patterns

5. ANTI-PATTERNS (REQUIRED - ≤ 40 words)
   - Specific behaviors to AVOID
   - Elements that should not be interacted with
   - When to stop and reassess the strategy

Focus on practical, actionable advice applicable to similar web navigation tasks.
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
    f"Thinking mode: {is_thinking}\n"
)

# Create batch model functions
extra_body = {
    "chat_template_kwargs": {"enable_thinking": True} if is_thinking else {"enable_thinking": False},
    "top_k": 20
}
test_data = load_jsonl(eval_path)
# Create tuner model batch function
tuner_model_batch_fn = create_model_batch_fn(
    clients=tuner_clients,
    model=tuner_model_name,
    temperature=0.6,
    max_tokens=2048,
    top_p=0.9,
    presence_penalty=1.5,
    extra_body=extra_body
)

# def prompt_postprocessor(system_prompt:str, prompt: str) -> str:
#     """
#     Post-process the prompt to ensure it's in the correct format for the model.
#     """
#     return system_prompt + "\n\n" + fewshot_prompt

def prompt_postprocessor(prompt: str) -> str:
    """
    Post-process the prompt to ensure it's in the correct format for the model.
    """
    return fewshot_prompt + "\n\n" + prompt

# Create tuning model batch function
tuning_model_batch_fn = create_model_batch_fn(
    clients=tuning_clients,
    model=tuning_model_name,
    temperature=0.0,  # For reproducibility
    presence_penalty=1.5,
    top_p=0.9,
    max_tokens=tuning_max_tokens,
    extra_body=extra_body,
    seed=seed,
    # system_prompt_processor=prompt_postprocessor,
    prompt_postprocessor=prompt_postprocessor,
)
embedding_model_fn = create_embedding_model_fn(
    api_base=EMBEDDING_BASE_URL,
    api_key=EMBEDDING_API_KEY,
    model=EMBEDDING_MODEL_NAME
)
mind2web_reward_fn = create_mind2web_reward_fn(tuner_model_batch_fn)

# Initialize PromptTuner
tuner = PromptTuner(
    reward_fn=mind2web_reward_fn,
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
    shuffle_batches=shuffle_batches,
    max_guidance_length=max_guidance_length,
    eval_at_n=eval_at_n,
)

# Baseline evaluation
if base_eval_score is None:
    logger.info("Computing base evaluation score...")
    tuner.knowledge_base.entries = []
    wrong_cases_path = os.path.join(checkpoint_dir, f"wrong_cases_base_{tuning_model_name.split('/')[-1]}_is_thinking_{is_thinking}.jsonl")
    base_eval_score= tuner._evaluate_on_eval_set(
    test_data,
    save_wrong_cases_path=wrong_cases_path,
    is_retrieval_subject=False,
    eval_at_n=eval_at_n
    )
    logger.info(f"base eval score: {base_eval_score}")
# Training phase
if is_train:
    for batch_size in batch_sizes:
        logger.info(f"Starting training with batch size: {batch_size}")
        checkpoint_dir_current = os.path.join(
            checkpoint_dir, 
            f"tuning_{tuning_model_name.split('/')[-1]}_tuner_{tuner_model_name.split('/')[-1]}_mind2web_batch{batch_size}_epoch{epoch}_is_thinking_{is_thinking}"
        )
        # Reset knowledge base and step count for each batch size
        tuner.knowledge_base.entries = []
        tuner.step_count = 0
        tuner.batch_size = batch_size
        tuner.mlflow_project_name = mlflow_project_name.format(batch_size, epoch)
        tuner.train(
            train_data_path=train_path,
            eval_data_path=eval_path,
            num_epochs=epoch,
            checkpoint_dir=checkpoint_dir_current,
            is_retrieval_subject=False,
            save_wrong_cases_path=checkpoint_dir_current
        )