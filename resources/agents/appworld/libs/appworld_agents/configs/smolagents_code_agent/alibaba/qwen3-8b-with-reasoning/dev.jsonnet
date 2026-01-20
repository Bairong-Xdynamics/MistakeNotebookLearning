local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
local experiment_configs_path = std.extVar("APPWORLD_EXPERIMENT_CONFIGS_PATH");
local experiment_code_path = std.extVar("APPWORLD_EXPERIMENT_CODE_PATH");
{
    "type": "smolagents",
    "config": {
        "model_server": {
            "command": "vllm serve Qwen/Qwen3-8B --reasoning-parser deepseek_r1 --max-num-seqs 5 --max-model-len 32000 --enable-auto-tool-choice --tool-call-parser hermes --port {port}",
            "enabled": true,
            "show_logs": false,
            "timeout": 600
        },
        "model": {
            "type": "openai",
            "base_url": "{MODEL_SERVER_URL}/v1",
            "api_key_env_name": "NO_API_KEY",
            "model_id": "Qwen/Qwen3-8B",
            "temperature": 0.0,
            "seed": 100,
            "max_completion_tokens": 3000,
            "cost_per_token": {"input_cache_hit": 0.0, "input_cache_miss": 0.0, "input_cache_write": 0.0, "output": 0.0},  # NOTE: Not used, need to figure out how to use it.
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
            "file_name": "qwen3-8b-with-reasoning",
            "humanized_name": "Qwen3 8B With Reasoning",
            "precise_name": "Qwen/Qwen3-8B",
            "creator": "alibaba",
            "provider": "vllm",
        },
        "agent": {
            "file_name": "smolagents_code_agent",
            "humanized_name": "Smolagents CodeAgent",
        },
    },
}