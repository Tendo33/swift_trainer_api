class TrainingHandler:
    def handle_multimodal(self, request, gpu_id_list):
        params = request.train_params or {}
        if hasattr(params, "dict"):
            params = params.model_dump()
        job_kwargs = dict(
            gpu_id=",".join(gpu_id_list),
            data_path=request.data_path,
            model_path=request.model_path,
            output_dir=request.output_dir,
            task_type=request.task_type,
            num_epochs=params.get("num_epochs", 1),
            batch_size=params.get("batch_size", 1),
            learning_rate=params.get("learning_rate", 1e-4),
            vit_lr=params.get("vit_lr", 1e-5),
            aligner_lr=params.get("aligner_lr", 1e-5),
            lora_rank=params.get("lora_rank", 16),
            lora_alpha=params.get("lora_alpha", 32),
            gradient_accumulation_steps=params.get("gradient_accumulation_steps", 4),
            eval_steps=params.get("eval_steps", 100),
            save_steps=params.get("save_steps", 100),
            save_total_limit=params.get("save_total_limit", 2),
            logging_steps=params.get("logging_steps", 5),
            max_length=params.get("max_length", 8192),
            warmup_ratio=params.get("warmup_ratio", 0.05),
            dataloader_num_workers=params.get("dataloader_num_workers", 4),
            dataset_num_proc=params.get("dataset_num_proc", 4),
            save_only_model=params.get("save_only_model", True),
            train_type=params.get("train_type", "lora"),
            torch_dtype=params.get("torch_dtype", "bfloat16"),
        )
        return job_kwargs

    def handle_language_model(self, request, gpu_id_list):
        params = request.train_params or {}
        if hasattr(params, "dict"):
            params = params.model_dump()
        job_kwargs = dict(
            gpu_id=",".join(gpu_id_list),
            data_path=request.data_path,
            model_path=request.model_path,
            output_dir=request.output_dir,
            task_type=request.task_type,
            num_epochs=params.get("num_epochs", 1),
            batch_size=params.get("batch_size", 1),
            learning_rate=params.get("learning_rate", 1e-4),
            gradient_accumulation_steps=params.get("gradient_accumulation_steps", 4),
            eval_steps=params.get("eval_steps", 100),
            save_steps=params.get("save_steps", 100),
            save_total_limit=params.get("save_total_limit", 2),
            logging_steps=params.get("logging_steps", 5),
            max_length=params.get("max_length", 2048),
            warmup_ratio=params.get("warmup_ratio", 0.05),
            dataloader_num_workers=params.get("dataloader_num_workers", 4),
            dataset_num_proc=params.get("dataset_num_proc", 4),
            save_only_model=params.get("save_only_model", True),
            train_type=params.get("train_type", "standard"),
            torch_dtype=params.get("torch_dtype", "bfloat16"),
        )
        return job_kwargs
