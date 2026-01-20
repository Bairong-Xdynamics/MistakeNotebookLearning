local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
local experiment_configs_path = std.extVar("APPWORLD_EXPERIMENT_CONFIGS_PATH");
local experiment_code_path = std.extVar("APPWORLD_EXPERIMENT_CODE_PATH");
local model = {
    "type": "litellm",
    "name": "anthropic/claude-opus-4-20250514",
    "settings": {
        "type": "litellm",
        "api_type": "chat_completions",
        "reasoning_effort": "medium",
        "temperature": 1.0,
        "cost_per_token": {"input_cache_hit": 1.5e-06, "input_cache_miss": 1.5e-05, "input_cache_write": 1.875e-05, "output": 7.5e-05},  # NOTE: Not used, need to figure out how to use it.
        "store": false,
    },
    "extras": {},
};
{
    "type": "openai_agents",
    "config": {
        "agent": {
            "model": model + {
                "settings": model.settings + {
                    "tool_choice": "auto",
                    "parallel_tool_calls": true,
                }
            },
            "max_steps": 50,
            "retrieve_apis": true,
            "prompt_file_path": experiment_prompts_path + "/function_calling_agent/instructions.txt",
            "demo_messages_file_path": experiment_prompts_path + "/function_calling_agent/demos.json",
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
        "dataset": "test_normal",
    },
    "metadata": {
        "model": {
            "file_name": "claude-opus-4-20250514-medium-reasoning",
            "humanized_name": "Claude Opus 4 20250514 Medium Reasoning",
            "precise_name": "anthropic/claude-opus-4-20250514",
            "creator": "anthropic",
            "provider": "anthropic",
        },
        "agent": {
            "file_name": "openai_agents_mcp_agent",
            "humanized_name": "OpenAI Agents MCP",
        },
    },
}