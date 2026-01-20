# 文件路径: experiments/configs/_generator/models/local.py

MODEL_INFOS = [
    {
        "model_name": "vllm-qwen3-8b",
        "client_name": "openai",
        "model_id": "/data2/models/Qwen3-8B",  # 对应你本地 vLLM 服务的 model name
        "model_kwargs": {
            "api_type": "chat_completions",
            "temperature": 0,
            "seed": 100,
            
            # [修改] 使用标准 Key 环境变量
            "api_key_env_name": "OPENAI_API_KEY", 
            
            # [修改] 直接指向你手动启动的 4390 端口
            "base_url": "http://localhost:4390/v1", 
            
            "max_completion_tokens": 3000,
            
            # [关键] 禁用 Thinking 模式
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            
            "tool_parser_name": None,
            "parallel_tool_calls": True,
            "cost_per_token": {
                "input_cache_miss": 0.0,
                "input_cache_hit": 0.0,
                "input_cache_write": 0.0,
                "output": 0.0,
            },
        },
        "function_calling": True,
        "tool_choice": "auto",
        "function_calling_demos": False, # 这个配置里 Demo 设为了 False，保持原样
        
        # vLLM 特有的属性修复
        "remove_function_property_keys": [
            "exclusiveMinimum",
            "exclusiveMaximum",
            "minimum",
            "maximum",
        ],
        
        # [修改] 禁用 AppWorld 自动启动 Server 的功能，因为你已经手动跑了
        "model_server_config": {
            "enabled": False, 
        },
        
        "part_of": ["vn", "vllm"],
        "provider": "vllm",
    }
]