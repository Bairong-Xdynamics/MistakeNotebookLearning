local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
local experiment_configs_path = std.extVar("APPWORLD_EXPERIMENT_CONFIGS_PATH");
local experiment_code_path = std.extVar("APPWORLD_EXPERIMENT_CODE_PATH");
{
    "type": "smolagents",
    "config": {
        "model": {
            "type": "openai",
            "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "api_key_env_name": "DASHSCOPE_API_KEY",
            "model_id": "qwen3-max-2025-09-23",
            "temperature": 0.0,
            "parallel_tool_calls": true,
            "seed": 100,
            "cost_per_token": {"input_cache_hit": 2.4e-07, "input_cache_miss": 1.2e-06, "input_cache_write": 0.0, "output": 6e-06},  # NOTE: Not used, need to figure out how to use it.
            "use_cache": false,
        },
        "api_predictor": {
            "mode": "predicted",
            "prompt_file_path": experiment_prompts_path + "/api_predictor.txt",
            "demo_task_ids": [
                "82e2fac_1",
                "29caf6f_1",
                "d0b1f43_1"
            ],
            "max_predicted_apis": 16,
        },
        "agent": {
            "type": "tool_calling",
            "prompt_templates": experiment_prompts_path + "/smolagents/tool_calling_instructions.yaml",
            "max_steps": 50,
            "max_seconds": 500,
            "max_cost_overall": 1000,
        },
        "task_completer": {
            "prompt_file_path": experiment_prompts_path + "/smolagents/task_completer_instructions.txt",
        },
        "appworld": {
            "random_seed": 100,
            "max_interactions": 1000,
            "max_api_calls_per_interaction": 5000,
            "raise_on_extra_parameters": true,
            "include_direct_functions": true,
            "direct_function_separator": "__",
        },
        "dataset": "dev",
    },
    "metadata": {
        "model": {
            "file_name": "qwen3-max-2025-09-23",
            "humanized_name": "Qwen3 Max 2025 09 23",
            "precise_name": "qwen3-max-2025-09-23",
            "creator": "alibaba",
            "provider": "alibaba",
        },
        "agent": {
            "file_name": "smolagents_tool_calling_agent",
            "humanized_name": "Smolagents ToolCallingAgent",
        },
    },
}