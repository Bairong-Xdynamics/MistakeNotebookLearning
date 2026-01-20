local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
local experiment_configs_path = std.extVar("APPWORLD_EXPERIMENT_CONFIGS_PATH");
local experiment_code_path = std.extVar("APPWORLD_EXPERIMENT_CODE_PATH");
{
    "type": "simplified",
    "config": {
        "model_server": {
            "command": "vllm serve baidu/ERNIE-4.5-21B-A3B-Thinking --reasoning-parser **TODO** --max-num-seqs 5 --max-model-len **TODO** --enable-auto-tool-choice --tool-call-parser **TODO** --port {port}",
            "enabled": true,
            "show_logs": false,
            "timeout": 600
        },
        "agent": {
            "type": "simplified_react_code_agent",
            "model_config": {
                "client_name": "openai",
                "api_type": "chat_completions",
                "base_url": "{MODEL_SERVER_URL}/v1",
                "api_key_env_name": "NO_API_KEY",
                "name": "baidu/ERNIE-4.5-21B-A3B-Thinking",
                "temperature": 0.0,
                "seed": 100,
                "max_completion_tokens": 3000,
                "drop_reasoning_content": false,
                "cost_per_token": {"input_cache_hit": 0.0, "input_cache_miss": 0.0, "input_cache_write": 0.0, "output": 0.0},
                "retry_after_n_seconds": 15,
                "use_cache": false,
                "max_retries": 100,
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
            "prompt_file_path": experiment_prompts_path + "/react_code_agent/instructions.txt",
            "ignore_multiple_calls": true,
            "max_prompt_length": null,
            "max_output_length": null,
            "max_steps": 50,
            "log_lm_calls": true,
            "skip_if_finished": true,
        },
        "dataset": "dev",
    },
    "metadata": {
        "model": {
            "file_name": "ernie-4.5-21b-a3b-thinking",
            "humanized_name": "Ernie 4.5 21B A3B Thinking",
            "precise_name": "baidu/ERNIE-4.5-21B-A3B-Thinking",
            "creator": "baidu",
            "provider": "vllm",
        },
        "agent": {
            "file_name": "simplified_react_code_agent",
            "humanized_name": "ReAct Code Agent",
        },
    },
}