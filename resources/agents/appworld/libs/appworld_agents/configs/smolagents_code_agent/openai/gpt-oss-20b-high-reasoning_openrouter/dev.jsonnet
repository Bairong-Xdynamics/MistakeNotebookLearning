local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
local experiment_configs_path = std.extVar("APPWORLD_EXPERIMENT_CONFIGS_PATH");
local experiment_code_path = std.extVar("APPWORLD_EXPERIMENT_CODE_PATH");
{
    "type": "smolagents",
    "config": {
        "model": {
            "type": "litellm",
            "model_id": "openrouter/openai/gpt-oss-20b",
            "reasoning": {"effort": "high"},
            "temperature": 0.0,
            "cost_per_token": {"input_cache_hit": 3e-08, "input_cache_miss": 3e-08, "input_cache_write": 0.0, "output": 1.5e-07},  # NOTE: Not used, need to figure out how to use it.
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
            "max_predicted_apis": 20,
        },
        "agent": {
            "type": "code",
            "prompt_templates": experiment_prompts_path + "/smolagents/code_instructions.yaml",
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
        },
        "dataset": "dev",
    },
    "metadata": {
        "model": {
            "file_name": "gpt-oss-20b-high-reasoning_openrouter",
            "humanized_name": "Gpt Oss 20B High Reasoning Openrouter",
            "precise_name": "openrouter/openai/gpt-oss-20b",
            "creator": "openai",
            "provider": "openrouter",
        },
        "agent": {
            "file_name": "smolagents_code_agent",
            "humanized_name": "Smolagents CodeAgent",
        },
    },
}