local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
local experiment_configs_path = std.extVar("APPWORLD_EXPERIMENT_CONFIGS_PATH");
local experiment_code_path = std.extVar("APPWORLD_EXPERIMENT_CODE_PATH");
{
    "type": "smolagents",
    "config": {
        "model": {
            "type": "litellm",
            "model_id": "openrouter/openai/gpt-oss-120b",
            "reasoning": {"effort": "low"},
            "temperature": 0.0,
            "parallel_tool_calls": true,
            "cost_per_token": {"input_cache_hit": 5e-08, "input_cache_miss": 5e-08, "input_cache_write": 0.0, "output": 2.5e-07},  # NOTE: Not used, need to figure out how to use it.
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
            "file_name": "gpt-oss-120b-low-reasoning_openrouter",
            "humanized_name": "Gpt Oss 120B Low Reasoning Openrouter",
            "precise_name": "openrouter/openai/gpt-oss-120b",
            "creator": "openai",
            "provider": "openrouter",
        },
        "agent": {
            "file_name": "smolagents_tool_calling_agent",
            "humanized_name": "Smolagents ToolCallingAgent",
        },
    },
}