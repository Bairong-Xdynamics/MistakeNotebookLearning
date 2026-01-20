local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
local experiment_configs_path = std.extVar("APPWORLD_EXPERIMENT_CONFIGS_PATH");
local experiment_code_path = std.extVar("APPWORLD_EXPERIMENT_CODE_PATH");
{
    "type": "smolagents",
    "config": {
        "model_server": {
            "command": "vllm serve nvidia/NVIDIA-Nemotron-Nano-12B-v2 --max-num-seqs 10 --port {port} --trust-remote-code --mamba_ssm_cache_dtype float32 --enable-auto-tool-choice --tool-call-parser nemotron_json "  + "--tool-parser-plugin " + experiment_code_path + "/common/vllm_plugins/nemotron_toolcall_parser.py ",
            "enabled": true,
            "show_logs": false,
            "timeout": 600
        },
        "model": {
            "type": "openai",
            "base_url": "{MODEL_SERVER_URL}/v1",
            "api_key_env_name": "NO_API_KEY",
            "model_id": "nvidia/NVIDIA-Nemotron-Nano-12B-v2",
            "temperature": 0.0,
            "parallel_tool_calls": true,
            "seed": 100,
            "max_completion_tokens": 3000,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": true}},
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
            "file_name": "nvidia-nemotron-nano-12b-v2-with-reasoning",
            "humanized_name": "Nvidia Nemotron Nano 12B V2 With Reasoning",
            "precise_name": "nvidia/NVIDIA-Nemotron-Nano-12B-v2",
            "creator": "nvidia",
            "provider": "vllm",
        },
        "agent": {
            "file_name": "smolagents_tool_calling_agent",
            "humanized_name": "Smolagents ToolCallingAgent",
        },
    },
}