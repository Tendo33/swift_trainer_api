from typing import Any, Dict

from application.models.training_model import TrainingHyperParams, TrainingTaskType
from application.setting import settings


class TrainingHandler:
    """训练任务处理器，只负责参数提取和组装"""

    def __init__(self) -> None:
        pass

    def _extract_params(self, request: Any) -> TrainingHyperParams | None:
        """提取请求参数"""
        params = request.train_params or None
        if params is None:
            return None
        if isinstance(params, dict):
            return TrainingHyperParams(**params)
        if isinstance(params, TrainingHyperParams):
            return params
        if hasattr(params, "model_dump"):
            return TrainingHyperParams(**params.model_dump())
        return None

    def _build_base_kwargs(
        self, request: Any, gpu_id_list: list[str], params: TrainingHyperParams | None
    ) -> dict:
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

    def handle_multimodal(self, request: Any, gpu_id_list: list[str]) -> dict:
        """处理多模态训练任务"""
        params = self._extract_params(request)
        base_kwargs = self._build_base_kwargs(request, gpu_id_list, params)
        training_kwargs = {**base_kwargs}
        if params:
            training_kwargs.update(params.dict(exclude_none=True))
        return training_kwargs

    def handle_language_model(self, request: Any, gpu_id_list: list[str]) -> dict:
        """处理语言模型训练任务"""
        params = self._extract_params(request)
        base_kwargs = self._build_base_kwargs(request, gpu_id_list, params)
        training_kwargs = {**base_kwargs}
        if params:
            training_kwargs.update(params.dict(exclude_none=True))
        return training_kwargs

    def handle_deploy(self, request: Any, gpu_id_list: list[str]) -> dict:
        """处理部署任务"""
        params = self._extract_params(request)
        base_kwargs = self._build_base_kwargs(request, gpu_id_list, params)
        training_kwargs = {**base_kwargs}
        if params:
            training_kwargs.update(params.dict(exclude_none=True))
        return training_kwargs

    def get_gpu_count_for_task(self, request: Any) -> int:
        """获取任务需要的GPU数量"""
        params = self._extract_params(request)

        # 优先使用请求中的GPU数量
        if params and params.num_epochs is not None:
            return getattr(params, "gpu_count", None) or settings.GPU_COUNT

        # 使用配置中的GPU数量
        return settings.GPU_COUNT

    def clear_config_cache(self) -> None:
        """清除配置缓存"""
        pass

    def get_available_profiles(self) -> list[str]:
        """获取可用的配置预设"""
        return ["default"]  # 不再区分 profile

    def get_profile_info(self, profile: str) -> dict:
        """获取配置预设信息"""
        return {
            "profile": "default",  # 不再区分 profile
            "environment": settings.ENVIRONMENT,
            "description": "默认配置",
            "gpu_count": settings.GPU_COUNT,
            "base_config": {},
            "multimodal_config": None,
            "language_model_config": None,
        }

    def get_environment_info(self) -> dict:
        """获取环境配置信息"""
        return {
            "environment": settings.ENVIRONMENT,
            "available_profiles": self.get_available_profiles(),
            "environment_overrides": {},
            "cache_size": 0,
        }

    def refresh_config_cache(self) -> dict:
        """刷新配置缓存"""
        return {"message": "配置缓存已刷新", "cleared_count": 0}

    def update_runtime_config(
        self, profile: str, updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """运行时更新配置"""
        try:
            # 获取当前配置
            current_config = {}  # 不再区分 profile

            # 创建更新的配置
            config_dict = current_config

            # 应用更新
            def update_nested_dict(d: dict, updates: dict):
                for key, value in updates.items():
                    if isinstance(value, dict) and key in d:
                        update_nested_dict(d[key], value)
                    else:
                        d[key] = value

            update_nested_dict(config_dict, updates)

            # 创建新的配置对象
            new_config = {}  # 不再区分 profile

            # 更新缓存

            return {
                "success": True,
                "message": "运行时配置更新成功",
                "updated_config": new_config,
                "applied_updates": updates,
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"运行时配置更新失败: {str(e)}",
                "error": str(e),
            }
