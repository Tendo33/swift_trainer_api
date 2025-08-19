class TrainingHandler:
    """训练任务处理器，负责处理不同类型的训练任务参数"""
    
    # 默认参数配置
    DEFAULT_PARAMS = {
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-4,
        "gradient_accumulation_steps": 4,
        "eval_steps": 100,
        "save_steps": 100,
        "save_total_limit": 2,
        "logging_steps": 5,
        "warmup_ratio": 0.05,
        "dataloader_num_workers": 4,
        "dataset_num_proc": 4,
        "save_only_model": True,
        "torch_dtype": "bfloat16",
    }
    
    # 多模态训练特定参数
    MULTIMODAL_PARAMS = {
        "vit_lr": 1e-5,
        "aligner_lr": 1e-5,
        "lora_rank": 16,
        "lora_alpha": 32,
        "max_length": 8192,
        "train_type": "lora",
    }
    
    # 语言模型训练特定参数
    LANGUAGE_MODEL_PARAMS = {
        "max_length": 2048,
        "train_type": "standard",
    }
    
    def _extract_params(self, request) -> dict:
        """提取请求参数"""
        params = request.train_params or {}
        if hasattr(params, "dict"):
            params = params.model_dump()
        return params
    
    def _build_base_kwargs(self, request, gpu_id_list, params: dict) -> dict:
        """构建基础任务参数"""
        return {
            "gpu_id": ",".join(gpu_id_list),
            "data_path": request.data_path,
            "model_path": request.model_path,
            "output_dir": request.output_dir,
            "task_type": request.task_type,
        }
    
    def _apply_params_with_defaults(self, base_kwargs: dict, params: dict, 
                                   task_specific_defaults: dict) -> dict:
        """应用参数配置和默认值"""
        # 合并默认参数
        all_defaults = {**self.DEFAULT_PARAMS, **task_specific_defaults}
        
        # 应用参数，使用默认值作为后备
        for key, default_value in all_defaults.items():
            base_kwargs[key] = params.get(key, default_value)
        
        return base_kwargs

    def handle_multimodal(self, request, gpu_id_list):
        """处理多模态训练任务"""
        params = self._extract_params(request)
        base_kwargs = self._build_base_kwargs(request, gpu_id_list, params)
        return self._apply_params_with_defaults(
            base_kwargs, params, self.MULTIMODAL_PARAMS
        )

    def handle_language_model(self, request, gpu_id_list):
        """处理语言模型训练任务"""
        params = self._extract_params(request)
        base_kwargs = self._build_base_kwargs(request, gpu_id_list, params)
        return self._apply_params_with_defaults(
            base_kwargs, params, self.LANGUAGE_MODEL_PARAMS
        )
