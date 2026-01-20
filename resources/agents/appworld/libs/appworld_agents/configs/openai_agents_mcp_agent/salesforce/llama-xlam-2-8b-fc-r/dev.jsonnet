local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
local experiment_configs_path = std.extVar("APPWORLD_EXPERIMENT_CONFIGS_PATH");
local experiment_code_path = std.extVar("APPWORLD_EXPERIMENT_CODE_PATH");
local model = {
    "type": "openai",
    "name": "Salesforce/Llama-xLAM-2-8b-fc-r",
    "settings": {
        "type": "openai",
        "api_type": "chat_completions",
        "base_url": "{MODEL_SERVER_URL}/v1",
        "api_key_env_name": "NO_API_KEY",
        "temperature": 0.0,
        "seed": 100,
        "max_completion_tokens": 3000,
        "cost_per_token": {"input_cache_hit": 0.0, "input_cache_miss": 0.0, "input_cache_write": 0.0, "output": 0.0},  # NOTE: Not used, need to figure out how to use it.
        "store": false,
    },
    "extras": {},
};
{
    "type": "openai_agents",
    "config": {
        "model_server": {
            "command": "vllm serve Salesforce/Llama-xLAM-2-8b-fc-r --max-num-seqs 5 --max-model-len 16000 --served-model-name llama-xlam-2-8b-fc-r --port {port} --enable-auto-tool-choice --tool-call-parser xlam "  + "--tool-parser-plugin " + experiment_code_path + "/common/vllm_plugins/xlam_tool_call_parser.py ",
            "enabled": true,
            "show_logs": false,
            "timeout": 600
        },
        "agent": {
            "model": model + {
                "settings": model.settings + {
                    "tool_choice": "auto",
                    "parallel_tool_calls": true,
                }
            },
            "max_steps": 50,
            "retrieve_apis": true,
            "prompt_file_path": experiment_prompts_path + "/function_calling_v2_zero_shot.txt",
            "demo_messages_file_path": null,
        },
        "api_predictor": {
            "mode": "predicted",
            "model_config": model,
            "prompt_file_path": experiment_prompts_path + "/api_predictor.txt",
            "demo_task_ids": [
                "82e2fac_1",
                "29caf6f_1",
                "d0b1f43_1"
            ],
            "max_predicted_apis": 20,
        },
        "appworld": {
            "start_servers": true,
            "show_server_logs": false,
            "remote_apis_url": "http://localhost:{port}",
            "remote_mcp_url": "http://localhost:{port}",
            "mcp_server_kwargs": {
                "output_type": "both_but_empty_text",
            },
            "random_seed": 100,
            "include_direct_functions": true,
            "direct_function_separator": "__",
        },
        "logger": {
            "color": true,
            "verbose": true,
        },
        "dataset": "dev",
    },
    "metadata": {
        "model": {
            "file_name": "llama-xlam-2-8b-fc-r",
            "humanized_name": "Llama Xlam 2 8B Fc R",
            "precise_name": "Salesforce/Llama-xLAM-2-8b-fc-r",
            "creator": "salesforce",
            "provider": "vllm",
        },
        "agent": {
            "file_name": "openai_agents_mcp_agent",
            "humanized_name": "OpenAI Agents MCP",
        },
    },
}