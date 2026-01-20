local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
local experiment_configs_path = std.extVar("APPWORLD_EXPERIMENT_CONFIGS_PATH");
local experiment_code_path = std.extVar("APPWORLD_EXPERIMENT_CODE_PATH");
{
    "type": "simplified",
    "config": {
        "model_server": {
            "command": "vllm serve nvidia/NVIDIA-Nemotron-Nano-12B-v2 --max-num-seqs 10 --port {port} --trust-remote-code --mamba_ssm_cache_dtype float32 --enable-auto-tool-choice --tool-call-parser nemotron_json "  + "--tool-parser-plugin " + experiment_code_path + "/common/vllm_plugins/nemotron_toolcall_parser.py ",
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
                "name": "nvidia/NVIDIA-Nemotron-Nano-12B-v2",
                "temperature": 0.0,
                "seed": 100,
                "max_completion_tokens": 3000,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": false}},
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
            "file_name": "nvidia-nemotron-nano-12b-v2-without-reasoning",
            "humanized_name": "Nvidia Nemotron Nano 12B V2 Without Reasoning",
            "precise_name": "nvidia/NVIDIA-Nemotron-Nano-12B-v2",
            "creator": "nvidia",
            "provider": "vllm",
        },
        "agent": {
            "file_name": "simplified_react_code_agent",
            "humanized_name": "ReAct Code Agent",
        },
    },
}