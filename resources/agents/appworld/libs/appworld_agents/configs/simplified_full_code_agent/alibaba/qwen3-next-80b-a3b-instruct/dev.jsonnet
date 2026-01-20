local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
local experiment_configs_path = std.extVar("APPWORLD_EXPERIMENT_CONFIGS_PATH");
local experiment_code_path = std.extVar("APPWORLD_EXPERIMENT_CODE_PATH");
local model_config = {
    "client_name": "openai",
    "api_type": "chat_completions",
    "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "api_key_env_name": "DASHSCOPE_API_KEY",
    "name": "qwen3-next-80b-a3b-instruct",
    "temperature": 0.0,
    "seed": 100,
    "drop_reasoning_content": false,
    "cost_per_token": {"input_cache_hit": 5e-07, "input_cache_miss": 5e-07, "input_cache_write": 0.0, "output": 2e-06},
    "retry_after_n_seconds": 15,
    "use_cache": false,
    "max_retries": 100,
};
local demo_task_ids = ["82e2fac_1", "29caf6f_1", "d0b1f43_1"];
{
    "type": "simplified",
    "config": {
        "agent": {
            "type": "simplified_full_code_agent",
            "model_config": model_config,
            "api_predictor_config": {
                "mode": "predicted",
                "model_config": model_config,
                "prompt_file_path": experiment_prompts_path + "/api_predictor.txt",
                "demo_task_ids": demo_task_ids,
                "max_predicted_apis": 20,
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
            "compress_api_docs": true,
            "demo_task_ids": demo_task_ids,
            "max_num_retrials": 5,
            "remove_code_demo_comments": true,
            "code_prompt_file_path": experiment_prompts_path + "/full_code_agent/full_code_instructions.txt",
            "retrial_prompt_file_path": experiment_prompts_path + "/full_code_agent/reflexion_instructions.txt",
            "max_steps": 20,
            "log_lm_calls": true,
            "skip_if_finished": true,
        },
        "dataset": "dev",
    },
    "metadata": {
        "model": {
            "file_name": "qwen3-next-80b-a3b-instruct",
            "humanized_name": "Qwen3 Next 80B A3B Instruct",
            "precise_name": "qwen3-next-80b-a3b-instruct",
            "creator": "alibaba",
            "provider": "alibaba",
        },
        "agent": {
            "file_name": "simplified_full_code_agent",
            "humanized_name": "Full Code Agent",
        },
    },
}