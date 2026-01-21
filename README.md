# Mistake Notebook Learning (MNL): Batch-Clustered Failures for Training-Free Agent Adaptation

Mistake Notebook Learning (MNL) is a novel, training-free memory framework that enables Large Language Model (LLM) agents to systematically learn from their mistakes. Instead of updating model weights, MNL distills shared error patterns from batch-clustered failures into structured "mistake notes." These notes are stored in an external memory and retrieved at test time to steer agents away from known pitfalls, enabling continuous improvement with minimal computational overhead.

---

## 🚀 Key Features
- **Training-Free Adaptation**: No gradient updates or parameter tuning required.
- **Batch-Clustered Abstraction**: Groups errors by subject to distill generalized, stable guidance.
- **Conservative Evolution**: Uses an "accept-if-improves" rule to ensure memory updates only enhance performance.
- **Cross-Domain Versatility**: Validated on Mathematical Reasoning (AIME, GSM8K), Text-to-SQL (KaggleDBQA, Spider), and Interactive Agents (Mind2Web, AppWorld).
- **Efficiency**: Compact memory structure and short inference-time prompts compared to retrieval-heavy baselines.

---

## 🧠 Methodology

MNL operates through a closed-loop iterative process involving two roles: the **Tuning Model** (the agent being improved) and the **Tuner Model** (the supervisor analyzing errors).

### 1. The MNL Evolution Protocol
1.  **Baseline Generation**: The Tuning Model generates initial responses for a batch of queries, augmented by existing memory context.
2.  **Memory Update**:
    - **Failure Identification**: Failed trajectories are identified using ground truth (Supervised) or an LLM judge (Self-Evolution).
    - **Subject Clustering**: Failures are grouped into semantic subjects (e.g., "SQL: Join conditions on null values").
    - **Guidance Distillation**: The Tuner Model extracts structured insights (Correct Approach, Mistake Summary, Generalizable Strategy, and Anti-Patterns) from each cluster.
    - **Memory Fusion**: New insights are merged with existing entries or appended as new nodes.
3.  **Post-Update Evaluation**: The batch is re-evaluated with the updated memory. The update is **accepted** only if net batch performance improves.

### 2. Learning Regimes
- **Supervised Evolution**: Uses ground-truth labels for feedback (e.g., Math, SQL).
- **Self-Evolution**: Uses a proxy verifier (LLM Judge) for feedback (e.g., web navigation, API interaction).

---

## 📅 Datasets

The datasets used in our experiments are available for download on Hugging Face:

🔗 **[MultiSense/MNL_PlayData](https://huggingface.co/datasets/MultiSense/MNL_Mind2Web_APPWorld)**

---

## 📊 Performance Results

MNL achieves competitive results across multiple benchmarks while maintaining a significantly smaller memory footprint than alternative methods:

| Task | Benchmark | Metric | Vanilla | MNL | Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Math** | AIME 2024 | Pass@32 | 30.0% | **33.0%** | +3.0% |
| **Text-to-SQL** | KaggleDBQA | EA (%) | 19.0% | **28.0%** | +9.0% |
| **Web Agent** | Mind2Web | Step Acc (%)| 11.5% | **15.6%** | +4.1% |
| **Tool Agent** | AppWorld | Task Success | 12.5% | **14.3%** | +1.8% |

*Results based on Qwen3-8B. MNL also demonstrates strong compatibility with test-time scaling (TTS) and think-enabled models. you can find more details in paper*
 **[Paper](https://arxiv.org/abs/2512.11485v3)**

---


## 🛠 Usage Examples

MNL provides a `PromptTuner` class to manage the evolution process. Below are examples of how to initialize and run tuning for different tasks.
### 0. Environment Setup
Before running the examples, create a `.env` file in the root directory and configure your API keys and endpoints:

```bash
# API Configuration
# Support multiple endpoints separated by comma for parallel processing

# Tuning Model API Configuration
TUNING_BASE_URLS="http://localhost:8814/v1,http://localhost:8815/v1"
TUNING_MODEL_NAME="Qwen3-8b"
TUNING_MODEL_API_KEYS="EMPTY"

# Tuner Model API Configuration  
TUNER_BASE_URLS="http://localhost:8814/v1,http://localhost:8815/v1"
TUNER_MODEL_NAME="Qwen3-8b"
TUNER_MODEL_API_KEYS="EMPTY"

# Knowledge Base Path
KNOWLEDGE_BASE_PATH="./knowledge_base.jsonl"

# Embedding Model API Configuration
EMBEDDING_BASE_URL = "http://127.0.0.1:10013/v1/"
EMBEDDING_MODEL_NAME = "bge-m3"
EMBEDDING_API_KEY = "EMPTY"


```

### 1. Text-to-SQL Optimization (`examples/example_dbqa.py`)
```python
from mnl import PromptTuner
from examples.utils.rewards import create_sql_reward_fn
from examples.utils.api_utils import create_model_batch_fn

# 1. Define Reward Function
reward_fn = create_sql_reward_fn(sqltester)

# 2. Define Batch Inference Functions
tuning_model_fn = create_model_batch_fn(model="qwen3-8b", ...)
tuner_model_fn = create_model_batch_fn(model="deepseek-v3", ...)

# 3. Initialize Tuner
tuner = PromptTuner(
    reward_fn=reward_fn,
    tuning_model_batch_fn=tuning_model_fn,
    tuner_model_batch_fn=tuner_model_fn,
    knowledge_base_path="knowledge_base.jsonl"
)

# 4. Start Training
tuner.train(train_data_path="train.jsonl", num_epochs=1)
```

### 2. Web Navigation Agent (`examples/example_mind2web.py`)
```python
from mnl import PromptTuner
from examples.utils.rewards import create_mind2web_reward_fn

# Uses an LLM Judge as a reward function for self-evolution
reward_fn = create_mind2web_reward_fn(tuner_model_batch_fn)

tuner = PromptTuner(
    reward_fn=reward_fn,
    tuning_model_batch_fn=tuning_model_batch_fn,
    tuner_model_batch_fn=tuner_model_batch_fn,
    knowledge_base_path="web_agent_kb.jsonl"
)

tuner.train(train_data_path="mind2web_train.jsonl", num_epochs=1)
```

---

## 🔧 Custom Reward Function (`reward_fn`)

The evolution process of MNL relies on a `reward_fn` to quantify model performance. This function acts as the objective function for prompt optimization, determining whether a distilled "mistake note" actually improves the agent's behavior. During each iteration, the `PromptTuner` compares responses from the updated memory against the baseline; an update is accepted only if the `reward_fn` indicates a net performance gain.

### Interface Specification
The reward function should follow this signature:

```python
def reward_fn(question: str, answer1: str, answer2: str, standard_answer: str) -> List[float]:
    """
    Compare the quality of two answers.
    
    Args:
        question: The input query or task description.
        answer1: The model's response under the updated prompt (or candidate 1).
        answer2: The model's response under the baseline prompt (or candidate 2).
        standard_answer: The ground-truth answer or a reference.
        
    Returns:
        [1.0, 0.0]: If answer1 is better than answer2.
        [0.0, 1.0]: If answer2 is better than answer1.
        [0.5, 0.5]: If both answers are of equal quality (tie).
    """
```

### Task Examples

#### 1. Mathematical Reasoning (Exact Match)
Suitable for tasks with clear-cut answers like GSM8K or AIME.

```python
def math_reward_fn(question, answer1, answer2, standard_answer):
    # Extract final numerical values from model responses
    a1 = extract_number(answer1)
    a2 = extract_number(answer2)
    std = extract_number(standard_answer)
    
    correct1 = (a1 == std)
    correct2 = (a2 == std)
    
    if correct1 and not correct2: return [1.0, 0.0]
    if not correct1 and correct2: return [0.0, 1.0]
    return [0.5, 0.5]
```

#### 2. Code/SQL (Execution-based)
Suitable for Text-to-SQL or programming tasks, where correctness is judged by comparing execution results.

```python
def sql_reward_fn(question, answer1, answer2, standard_answer):
    # Execute SQL and compare result sets
    res1 = db_engine.execute(extract_sql(answer1))
    res2 = db_engine.execute(extract_sql(answer2))
    res_std = db_engine.execute(standard_answer)
    
    correct1 = (res1 == res_std)
    correct2 = (res2 == res_std)
    
    if correct1 and not correct2: return [1.0, 0.0]
    if not correct1 and correct2: return [0.0, 1.0]
    return [0.5, 0.5]
```

#### 3. Open-ended Tasks (LLM-as-a-Judge)
When ground truth is unavailable, a stronger model (e.g., DeepSeek-V3 or GPT-4o) can act as a judge for self-evolution.

```python
def judge_reward_fn(question, answer1, answer2, standard_answer):
    judge_prompt = f"Question: {question}\nAnswer A: {answer1}\nAnswer B: {answer2}\nWhich one is better?"
    response = tuner_model.generate(judge_prompt)
    
    if "Answer A is better" in response:
        return [1.0, 0.0]
    elif "Answer B is better" in response:
        return [0.0, 1.0]
    return [0.5, 0.5]
```

---

## 📂 Project Structure
- `mnl/`: Core framework implementation (Tuner, Memory, Knowledge Base).
- `examples/`: Task-specific scripts for SQL, Web Agent, etc.
- `resources/`: Dataset files and database schemas.

---

## 📜 Citation
If you find this work useful, please cite our paper:
```bibtex
@misc{su2026mistakenotebooklearningbatchclustered,
      title={Mistake Notebook Learning: Batch-Clustered Failures for Training-Free Agent Adaptation}, 
      author={Xuanbo Su and Yingfang Zhang and Hao Luo and Xiaoteng Liu and Leo Huang},
      year={2026},
      eprint={2512.11485},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.11485}, 
}
```

