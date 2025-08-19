from typing import Any, Dict, List

from application.models.base_trainer import (
    BaseTrainer,
    TrainerFactory,
)
from application.models.training_model import (
    TrainingHyperParams,
    TrainingJobCreateRequest,
    TrainingTaskType,
    determine_trainer_type,
    resolve_training_params,
)
from application.setting import settings


class TrainingHandler:
    """训练任务处理器，基于新的训练器基类架构"""

    def __init__(self) -> None:
        pass

    def create_trainer(
        self, request: TrainingJobCreateRequest, gpu_id_list: List[str]
    ) -> BaseTrainer:
        """创建训练器实例"""

        # 解析训练参数
        params = resolve_training_params(request)
        if not params:
            raise ValueError("无法解析训练参数")

        # 确定训练器类型
        trainer_type = determine_trainer_type(request)
        if not trainer_type:
            raise ValueError(f"不支持的任务类型: {request.task_type}")

        # 验证必要参数
        if not request.data_path:
            raise ValueError("数据集路径不能为空")
        if not request.model_path:
            raise ValueError("模型路径不能为空")
        if not request.output_dir:
            raise ValueError("输出目录不能为空")

        # 创建训练器
        trainer = TrainerFactory.create_trainer(
            trainer_type=trainer_type,
            gpu_ids=gpu_id_list,
            data_path=request.data_path,
            model_path=request.model_path,
            output_dir=request.output_dir,
            params=params,
        )

        # 验证训练器参数
        if not trainer.validate_params():
            raise ValueError("训练参数验证失败")

        return trainer

    def build_training_kwargs(
        self, request: TrainingJobCreateRequest, gpu_id_list: List[str]
    ) -> dict:
        """构建训练任务参数（兼容原有接口）"""
        try:
            trainer = self.create_trainer(request, gpu_id_list)
            params = trainer.params

            # 构建基础参数
            training_kwargs = {
                "gpu_id": ",".join(gpu_id_list),
                "data_path": request.data_path,
                "model_path": request.model_path,
                "output_dir": request.output_dir,
                "task_type": request.task_type,
                "trainer_type": trainer.get_trainer_type(),
                "train_params": params,
            }

            # 添加参数字典（向后兼容）
            params_dict = params.model_dump(exclude_none=True)
            training_kwargs.update(params_dict)

            return training_kwargs

        except Exception as e:
            raise ValueError(f"构建训练参数失败: {str(e)}")

    # 保留原有方法以确保向后兼容
    def handle_multimodal(self, request: Any, gpu_id_list: List[str]) -> dict:
        """处理多模态训练任务（向后兼容）"""
        if not isinstance(request, TrainingJobCreateRequest):
            # 转换为新的请求格式
            request = self._convert_legacy_request(request, TrainingTaskType.MULTIMODAL)
        return self.build_training_kwargs(request, gpu_id_list)

    def handle_language_model(self, request: Any, gpu_id_list: List[str]) -> dict:
        """处理语言模型训练任务（向后兼容）"""
        if not isinstance(request, TrainingJobCreateRequest):
            # 转换为新的请求格式
            request = self._convert_legacy_request(
                request, TrainingTaskType.LANGUAGE_MODEL
            )
        return self.build_training_kwargs(request, gpu_id_list)

    def handle_deploy(self, request: Any, gpu_id_list: List[str]) -> dict:
        """处理部署任务（向后兼容）"""
        if not isinstance(request, TrainingJobCreateRequest):
            # 转换为新的请求格式
            request = self._convert_legacy_request(request, TrainingTaskType.DEPLOY)
        return self.build_training_kwargs(request, gpu_id_list)

    def _convert_legacy_request(
        self, request: Any, task_type: TrainingTaskType
    ) -> TrainingJobCreateRequest:
        """转换旧的请求格式"""
        # 提取旧的参数
        params = getattr(request, "train_params", None)

        return TrainingJobCreateRequest(
            task_type=task_type,
            data_path=getattr(request, "data_path", None),
            model_path=getattr(request, "model_path", None),
            output_dir=getattr(request, "output_dir", None),
            priority=getattr(request, "priority", 0),
            train_params=params,
        )

    def get_gpu_count_for_task(self, request: Any) -> int:
        """获取任务需要的GPU数量"""
        # 如果是新的请求格式，直接使用设置中的GPU数量
        if isinstance(request, TrainingJobCreateRequest):
            return settings.GPU_COUNT

        # 兼容旧的请求格式
        params = self._extract_params_legacy(request)
        if params and hasattr(params, "num_epochs") and params.num_epochs is not None:
            return getattr(params, "gpu_count", None) or settings.GPU_COUNT

        return settings.GPU_COUNT

    def _extract_params_legacy(self, request: Any) -> TrainingHyperParams | None:
        """提取请求参数（旧格式兼容）"""
        params = getattr(request, "train_params", None)
        if params is None:
            return None
        if isinstance(params, dict):
            return TrainingHyperParams(**params)
        if isinstance(params, TrainingHyperParams):
            return params
        if hasattr(params, "model_dump"):
            return TrainingHyperParams(**params.model_dump())
        return None

    def clear_config_cache(self) -> None:
        """清除配置缓存"""
        pass

    def get_available_profiles(self) -> List[str]:
        """获取可用的配置预设"""
        return ["default"]  # 不再区分 profile

    def get_profile_info(self, profile: str) -> Dict[str, Any]:
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

    def get_environment_info(self) -> Dict[str, Any]:
        """获取环境配置信息"""
        return {
            "environment": settings.ENVIRONMENT,
            "available_profiles": self.get_available_profiles(),
            "environment_overrides": {},
            "cache_size": 0,
        }

    def refresh_config_cache(self) -> Dict[str, Any]:
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
