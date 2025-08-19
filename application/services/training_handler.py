from typing import Any, Dict

from application.models.training_model import (
    TrainingConfigManager,
    TrainingConfiguration,
    TrainingTaskType,
)
from application.setting import settings


class TrainingHandler:
    """训练任务处理器，负责处理不同类型的训练任务参数"""
    
    def __init__(self):
        self._config_cache: Dict[str, TrainingConfiguration] = {}
    
    def _get_config(self) -> TrainingConfiguration:
        """获取配置，支持缓存和环境感知"""
        if "default" not in self._config_cache:
            self._config_cache["default"] = TrainingConfigManager.get_profile_config_with_env(settings.ENVIRONMENT)
        return self._config_cache["default"]
    
    def _extract_params(self, request) -> dict:
        """提取请求参数"""
        params = request.train_params or {}
        if hasattr(params, "dict"):
            params = params.model_dump()
        elif hasattr(params, "model_dump"):
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
    
    def _get_task_type_enum(self, task_type_str: str) -> TrainingTaskType:
        """转换任务类型字符串为枚举"""
        try:
            return TrainingTaskType(task_type_str)
        except ValueError:
            raise ValueError(f"不支持的任务类型: {task_type_str}")
    
    def handle_multimodal(self, request, gpu_id_list) -> dict:
        """处理多模态训练任务"""
        params = self._extract_params(request)
        base_kwargs = self._build_base_kwargs(request, gpu_id_list, params)
        
        # 获取配置
        config = self._get_config()
        
        # 合并配置和请求参数
        task_config = config.merge_with_request(params, TrainingTaskType.MULTIMODAL)
        
        # 移除配置相关的参数，只保留训练参数
        training_kwargs = {**base_kwargs}
        for key, value in task_config.items():
            if key != "description":
                training_kwargs[key] = value
        
        return training_kwargs
    
    def handle_language_model(self, request, gpu_id_list) -> dict:
        """处理语言模型训练任务"""
        params = self._extract_params(request)
        base_kwargs = self._build_base_kwargs(request, gpu_id_list, params)
        
        # 获取配置
        config = self._get_config()
        
        # 合并配置和请求参数
        task_config = config.merge_with_request(params, TrainingTaskType.LANGUAGE_MODEL)
        
        # 移除配置相关的参数，只保留训练参数
        training_kwargs = {**base_kwargs}
        for key, value in task_config.items():
            if key not in ["profile", "description"]:
                training_kwargs[key] = value
        
        return training_kwargs
    
    def handle_deploy(self, request, gpu_id_list) -> dict:
        """处理部署任务"""
        params = self._extract_params(request)
        base_kwargs = self._build_base_kwargs(request, gpu_id_list, params)
        
        # 获取配置
        config = self._get_config()
        
        # 合并配置和请求参数
        task_config = config.merge_with_request(params, TrainingTaskType.DEPLOY)
        
        # 移除配置相关的参数，只保留训练参数
        training_kwargs = {**base_kwargs}
        for key, value in task_config.items():
            if key not in ["profile", "description"]:
                training_kwargs[key] = value
        
        return training_kwargs
    
    def get_gpu_count_for_task(self, request) -> int:
        """获取任务需要的GPU数量"""
        params = self._extract_params(request)
        config = self._get_config()
        
        # 优先使用请求中的GPU数量
        if "gpu_count" in params:
            return params["gpu_count"]
        
        # 使用配置中的GPU数量
        return config.base.gpu_count
    
    def clear_config_cache(self):
        """清除配置缓存"""
        self._config_cache.clear()
    
    def get_available_profiles(self) -> list:
        """获取可用的配置预设"""
        return ["default"] # 不再区分 profile
    
    def get_profile_info(self, profile: str) -> dict:
        """获取配置预设信息"""
        config = self._get_config()
        return {
            "profile": "default", # 不再区分 profile
            "environment": settings.ENVIRONMENT,
            "description": config.description or "默认配置",
            "gpu_count": config.base.gpu_count,
            "base_config": config.base.model_dump(),
            "multimodal_config": config.multimodal.model_dump() if config.multimodal else None,
            "language_model_config": config.language_model.model_dump() if config.language_model else None,
        }
    
    def get_environment_info(self) -> dict:
        """获取环境配置信息"""
        return {
            "environment": settings.ENVIRONMENT,
            "available_profiles": self.get_available_profiles(),
            "environment_overrides": TrainingConfigManager.get_environment_overrides(settings.ENVIRONMENT),
            "cache_size": len(self._config_cache),
        }
    
    def refresh_config_cache(self):
        """刷新配置缓存"""
        self._config_cache.clear()
        return {"message": "配置缓存已刷新", "cleared_count": len(self._config_cache)}
    
    def update_runtime_config(self, profile: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """运行时更新配置"""
        try:
            # 获取当前配置
            current_config = self._get_config()
            
            # 创建更新的配置
            config_dict = current_config.model_dump()
            
            # 应用更新
            def update_nested_dict(d: dict, updates: dict):
                for key, value in updates.items():
                    if isinstance(value, dict) and key in d:
                        update_nested_dict(d[key], value)
                    else:
                        d[key] = value
            
            update_nested_dict(config_dict, updates)
            
            # 创建新的配置对象
            new_config = TrainingConfiguration(**config_dict)
            
            # 更新缓存
            profile_key = "default" # 不再区分 profile
            self._config_cache[profile_key] = new_config
            
            return {
                "success": True,
                "message": "运行时配置更新成功",
                "updated_config": new_config.model_dump(),
                "applied_updates": updates
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"运行时配置更新失败: {str(e)}",
                "error": str(e)
            }
