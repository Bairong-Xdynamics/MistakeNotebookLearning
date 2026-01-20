#!/usr/bin/env python3
import os
import sys
import json
import logging
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Union, Callable
from jinja2 import Template
import openai

# ================= 1. Path & Environment =================

def _setup_env(libs_path: str, appworld_root: str, api_key: str = "EMPTY", model_server_url: str = None):
    if libs_path not in sys.path:
        sys.path.insert(0, libs_path)
    os.environ["APPWORLD_ROOT"] = appworld_root
    os.environ["OPENAI_API_KEY"] = api_key
    if model_server_url:
        os.environ["MODEL_SERVER_URL"] = model_server_url

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
MY_RESOURCE_ROOT = os.path.join(project_root, "resources", "agents/appworld")
LIBS_PATH = os.path.join(MY_RESOURCE_ROOT, "libs")
_setup_env(LIBS_PATH, MY_RESOURCE_ROOT)

DEFAULT_PROMPT_PATH = os.path.join(MY_RESOURCE_ROOT, "instructions.txt")

from appworld_agents.code.simplified.react_code_agent import SimplifiedReActCodeAgent
from appworld_agents.code.simplified.agent import ExecutionIO

# ================= 2. Tuning Agent =================

class TuningReActAgent(SimplifiedReActCodeAgent):
    """ReAct agent with prompt tuning for AppWorld"""
    def __init__(self, prompt_file_path: str, model_config: Dict[str, Any], **kwargs):
        super().__init__(prompt_file_path=prompt_file_path, model_config=model_config, **kwargs)

    def _render_prompt(self, world: Any, guidance: str) -> str:
        template = Template(self.prompt_template)
        app_descriptions = json.dumps(
            [{"name": k, "description": v} for k, v in world.task.app_descriptions.items()],
            indent=2, ensure_ascii=False
        )
        params = {
            "instruction": world.task.instruction,
            "main_user": world.task.supervisor,
            "app_descriptions": app_descriptions,
            "guidance": guidance
        }
        return self.truncate_input(template.render(params)) + "\n\n"

    def initialize(self, world: Any, guidance: str = "") -> None:
        super().initialize(world)
        self.messages = self.text_to_messages(self._render_prompt(world, guidance))
        self.num_instruction_messages = len(self.messages)

# ================= 3. Worker Process =================

def _worker_process_task(
    task_id: str,
    instruction: str,
    guidance: str,
    merged_config: Dict[str, Any],
    prompt_template_path: str,
    max_steps: int,
    experiment_name: str,
    paths_config: Dict[str, str],
    model_server_url: str,
) -> str:
    _setup_env(paths_config["libs_path"], paths_config["appworld_root"],
               api_key=merged_config.get("api_key", "EMPTY"),
               model_server_url=model_server_url)

    from appworld import AppWorld
    from appworld.evaluator import evaluate_task

    AppWorld.initializer(update_defaults=True,
                          config={"model_server": {"enabled": False}, "experiment_name": experiment_name})

    agent = TuningReActAgent(
        prompt_file_path=prompt_template_path,
        model_config=merged_config,
        max_steps=max_steps,
        max_prompt_length=None,
        max_output_length=None,
        logger_config={"verbose": False},
        log_lm_calls=True,
        skip_if_finished=False,
        ignore_multiple_calls=True,
        appworld_config={"random_seed": 42, "raise_on_extra_parameters": True},
        usage_tracker_config={"max_cost_overall": 1000, "max_cost_per_task": 10, "max_output_tokens_per_task": 100000},
    )

    trajectory, is_success = [], False


    with AppWorld(task_id=task_id, experiment_name=experiment_name) as world:
        if instruction:
            world.task.instruction = instruction
        agent.initialize(world, guidance=guidance)
        execution_outputs = []

        for _ in range(agent.max_steps):
            agent.step_number += 1
            execution_inputs, usage, status = agent.next_execution_inputs_usage_and_status(execution_outputs)
            if status.failed or not execution_inputs:
                break
            raw_outputs = world.batch_execute([inp.content for inp in execution_inputs])
            execution_outputs = [
                ExecutionIO(content=out, metadata=inp.metadata)
                for inp, out in zip(execution_inputs, raw_outputs, strict=True)
            ]
            agent.usage_tracker.add(task_id, usage)
            if world.task_completed() or agent.usage_tracker.exceeded(task_id):
                break
        trajectory = agent.messages


    test_tracker = evaluate_task(task_id=task_id, experiment_name=experiment_name, suppress_errors=True)
    is_success = test_tracker.success



    tag = "<<<EVAL_RESULT: SUCCESS>>>" if is_success else "<<<EVAL_RESULT: FAILURE>>>"
    trajectory.append({"role": "system", "content": f"[System]: {tag}"})
    return json.dumps(trajectory, ensure_ascii=False)

# ================= 4. Batch Function Factory =================

def create_appworld_batch_fn(
    clients: List[openai.OpenAI],
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    presence_penalty: float = 1.5,
    top_p: float = 0.9,
    extra_body: Dict[str, Any] = None,
    prompt_template_path: str = None,
    max_steps: int = 40,
    experiment_name: str = "appworld_prompt_tuning_run",
    num_workers: int = 4,
    seed: int = 42,
) -> Callable:

    if prompt_template_path is None:
        prompt_template_path = DEFAULT_PROMPT_PATH

    base_llm_config = {
        "client_name": "openai",
        "api_type": "chat_completions",
        "api_key_env_name": "OPENAI_API_KEY",
        "name": model_name,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "extra_body": extra_body or {},
        "retry_after_n_seconds": 15,
        "max_retries": 5,
        "presence_penalty": presence_penalty,
        "top_p": top_p,
        "seed": seed,
        "use_cache": False,
        "drop_reasoning_content": False,
        "cost_per_token": {"input_cache_hit": 0.0, "input_cache_miss": 0.0,
                           "input_cache_write": 0.0, "output": 0.0},
    }

    client_configs = [{"base_url": str(c.base_url), "api_key": c.api_key} for c in clients]

    def appworld_batch_fn(prompts: List[str], system_prompts: Union[str, List[str]], temperature: float, max_tokens: int):
        paths_config = {"libs_path": LIBS_PATH, "appworld_root": MY_RESOURCE_ROOT}
        task_args_list = []

        for i, prompt_item in enumerate(prompts):
            task_id, instruction = (prompt_item.split("|", 1) if "|" in prompt_item else (prompt_item, ""))
            task_id = task_id.strip()
            guidance = system_prompts[i] if isinstance(system_prompts, list) and i < len(system_prompts) else (system_prompts or "")
            assigned_client = client_configs[i % len(client_configs)]
            task_config = base_llm_config.copy()
            task_config.update({"base_url": assigned_client["base_url"], "api_key": assigned_client["api_key"]})
            task_args_list.append((
                task_id, instruction, guidance, task_config, prompt_template_path,
                max_steps, experiment_name, paths_config, assigned_client["base_url"]
            ))

        logging.info(f"🚀 Starting AppWorld batch: {len(task_args_list)} tasks on {len(client_configs)} endpoints")
        results = [None] * len(task_args_list)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_index = {executor.submit(_worker_process_task, *args): i for i, args in enumerate(task_args_list)}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    results[index] = json.dumps([{"role": "system", "content": f"Worker Exception: {e}"}])
        return results

    return appworld_batch_fn
