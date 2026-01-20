local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
local experiment_configs_path = std.extVar("APPWORLD_EXPERIMENT_CONFIGS_PATH");
local experiment_code_path = std.extVar("APPWORLD_EXPERIMENT_CODE_PATH");
local model_config = {
    "client_name": "litellm",
    "api_type": "chat_completions",
    "name": "anthropic/claude-3-7-sonnet-20250219",
    "reasoning_effort": "high",
    "temperature": 1.0,
    "drop_reasoning_content": false,
    "cost_per_token": {"input_cache_hit": 3e-07, "input_cache_miss": 3e-06, "input_cache_write": 3.75e-06, "output": 1.5e-05},
    "retry_after_n_seconds": 15,
    "use_cache": false,
    "max_retries": 100,
};
local demo_task_ids = ["82e2fac_1", "29caf6f_1", "d0b1f43_1"];
{
    "type": "simplified",
    "config": {
        "agent": {
            "type": "simplified_full_code_agent",
            "model_config": model_config,
            "api_predictor_config": {
                "mode": "predicted",
                "model_config": model_config,
                "prompt_file_path": experiment_prompts_path + "/api_predictor.txt",
                "demo_task_ids": demo_task_ids,
                "max_predicted_apis": 20,
            },
            "appworld_config": {
                "random_seed": 100,
                "raise_on_extra_parameters": true,
            },
            "logger_config": {
                "color": true,
                "verbose": true,
            },
            "usage_tracker_config": {
                "max_cost_overall": 1000,
                "max_cost_per_task": 10,
                "max_output_tokens_per_task": 100000,
            },
            "compress_api_docs": true,
            "demo_task_ids": demo_task_ids,
            "max_num_retrials": 5,
            "remove_code_demo_comments": true,
            "code_prompt_file_path": experiment_prompts_path + "/full_code_agent/full_code_instructions.txt",
            "retrial_prompt_file_path": experiment_prompts_path + "/full_code_agent/reflexion_instructions.txt",
            "max_steps": 20,
            "log_lm_calls": true,
            "skip_if_finished": true,
        },
        "dataset": "dev",
    },
    "metadata": {
        "model": {
            "file_name": "claude-3-7-sonnet-20250219-high-reasoning",
            "humanized_name": "Claude 3 7 Sonnet 20250219 High Reasoning",
            "precise_name": "anthropic/claude-3-7-sonnet-20250219",
            "creator": "anthropic",
            "provider": "anthropic",
        },
        "agent": {
            "file_name": "simplified_full_code_agent",
            "humanized_name": "Full Code Agent",
        },
    },
}