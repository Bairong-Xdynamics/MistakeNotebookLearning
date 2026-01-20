local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
local experiment_configs_path = std.extVar("APPWORLD_EXPERIMENT_CONFIGS_PATH");
local experiment_code_path = std.extVar("APPWORLD_EXPERIMENT_CODE_PATH");
{
    "type": "smolagents",
    "config": {
        "model": {
            "type": "litellm",
            "model_id": "anthropic/claude-opus-4-1-20250805",
            "temperature": 0.0,
            "parallel_tool_calls": true,
            "cost_per_token": {"input_cache_hit": 1.5e-06, "input_cache_miss": 1.5e-05, "input_cache_write": 1.875e-05, "output": 7.5e-05},  # NOTE: Not used, need to figure out how to use it.
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
            "file_name": "claude-opus-4-1-20250805-no-reasoning",
            "humanized_name": "Claude Opus 4 1 20250805 No Reasoning",
            "precise_name": "anthropic/claude-opus-4-1-20250805",
            "creator": "anthropic",
            "provider": "anthropic",
        },
        "agent": {
            "file_name": "smolagents_tool_calling_agent",
            "humanized_name": "Smolagents ToolCallingAgent",
        },
    },
}