# =============================================
# 训练器基类和专用参数定义
# 创建时间：2024-12-19
# 重构目标：分离LLM和VLLM训练，提高扩展性
# =============================================

import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field


class TrainerType(str, Enum):
    """训练器类型枚举"""

    LLM = "llm"
    MLLM = "mllm"


class BaseTrainingParams(BaseModel, ABC):
    """训练参数基类"""

    # 通用参数
    num_epochs: int = Field(default=1, description="训练轮数")
    batch_size: int = Field(default=1, description="批次大小")
    learning_rate: float = Field(default=1e-4, description="学习率")
    gradient_accumulation_steps: int = Field(default=4, description="梯度累积步数")
    eval_steps: int = Field(default=100, description="评估步数")
    save_steps: int = Field(default=100, description="保存步数")
    save_total_limit: int = Field(default=2, description="保存总数限制")
    logging_steps: int = Field(default=5, description="日志步数")
    max_length: int = Field(default=8192, description="最大长度")
    warmup_ratio: float = Field(default=0.05, description="预热比例")
    dataloader_num_workers: int = Field(default=4, description="数据加载器进程数")
    dataset_num_proc: int = Field(default=4, description="数据集处理进程数")
    save_only_model: bool = Field(default=True, description="仅保存模型")
    torch_dtype: str = Field(default="bfloat16", description="PyTorch数据类型")

    @abstractmethod
    def get_trainer_type(self) -> TrainerType:
        """获取训练器类型"""
        pass

    @abstractmethod
    def to_command_args(self) -> List[str]:
        """转换为命令行参数"""
        pass


class LLMTrainingParams(BaseTrainingParams):
    """大语言模型训练参数"""

    train_type: str = Field(default="lora", description="训练类型")
    lora_rank: int = Field(default=16, description="LoRA rank")
    lora_alpha: int = Field(default=32, description="LoRA alpha")

    def get_trainer_type(self) -> TrainerType:
        return TrainerType.LLM

    def to_command_args(self) -> List[str]:
        """转换为LLM训练命令行参数"""
        args = [
            "--train_type",
            self.train_type,
            "--torch_dtype",
            self.torch_dtype,
            "--num_train_epochs",
            str(self.num_epochs),
            "--per_device_train_batch_size",
            str(self.batch_size),
            "--per_device_eval_batch_size",
            str(self.batch_size),
            "--learning_rate",
            str(self.learning_rate),
            "--lora_rank",
            str(self.lora_rank),
            "--lora_alpha",
            str(self.lora_alpha),
            "--gradient_accumulation_steps",
            str(self.gradient_accumulation_steps),
            "--eval_steps",
            str(self.eval_steps),
            "--save_steps",
            str(self.save_steps),
            "--save_total_limit",
            str(self.save_total_limit),
            "--logging_steps",
            str(self.logging_steps),
            "--max_length",
            str(self.max_length),
            "--warmup_ratio",
            str(self.warmup_ratio),
            "--dataloader_num_workers",
            str(self.dataloader_num_workers),
            "--dataset_num_proc",
            str(self.dataset_num_proc),
        ]

        if self.save_only_model:
            args.append("--save_only_model")

        return args


class MLLMTrainingParams(BaseTrainingParams):
    """多模态大语言模型训练参数"""

    train_type: str = Field(default="lora", description="训练类型")
    vit_lr: float = Field(default=1e-5, description="ViT学习率")
    aligner_lr: float = Field(default=1e-5, description="Aligner学习率")
    lora_rank: int = Field(default=16, description="LoRA rank")
    lora_alpha: int = Field(default=32, description="LoRA alpha")

    def get_trainer_type(self) -> TrainerType:
        return TrainerType.MLLM

    def to_command_args(self) -> List[str]:
        """转换为VLLM训练命令行参数"""
        args = [
            "--train_type",
            self.train_type,
            "--torch_dtype",
            self.torch_dtype,
            "--num_train_epochs",
            str(self.num_epochs),
            "--per_device_train_batch_size",
            str(self.batch_size),
            "--per_device_eval_batch_size",
            str(self.batch_size),
            "--learning_rate",
            str(self.learning_rate),
            "--vit_lr",
            str(self.vit_lr),
            "--aligner_lr",
            str(self.aligner_lr),
            "--lora_rank",
            str(self.lora_rank),
            "--lora_alpha",
            str(self.lora_alpha),
            "--gradient_accumulation_steps",
            str(self.gradient_accumulation_steps),
            "--eval_steps",
            str(self.eval_steps),
            "--save_steps",
            str(self.save_steps),
            "--save_total_limit",
            str(self.save_total_limit),
            "--logging_steps",
            str(self.logging_steps),
            "--max_length",
            str(self.max_length),
            "--warmup_ratio",
            str(self.warmup_ratio),
            "--dataloader_num_workers",
            str(self.dataloader_num_workers),
            "--dataset_num_proc",
            str(self.dataset_num_proc),
        ]

        if self.save_only_model:
            args.append("--save_only_model")

        return args


class BaseTrainer(ABC):
    """训练器基类"""

    def __init__(
        self,
        gpu_ids: List[str],
        data_path: str,
        model_path: str,
        output_dir: str,
        params: BaseTrainingParams,
    ):
        self.gpu_ids = gpu_ids
        self.data_path = data_path
        self.model_path = model_path
        self.output_dir = output_dir
        self.params = params

    @abstractmethod
    def build_command(self) -> List[str]:
        """构建训练命令"""
        pass

    @abstractmethod
    def build_environment(self) -> Dict[str, str]:
        """构建环境变量"""
        pass

    @abstractmethod
    def validate_params(self) -> bool:
        """验证训练参数"""
        pass

    @abstractmethod
    def get_trainer_type(self) -> TrainerType:
        """获取训练器类型"""
        pass

    def get_gpu_count(self) -> int:
        """获取GPU数量"""
        return len(self.gpu_ids)

    def get_gpu_ids_str(self) -> str:
        """获取GPU ID字符串"""
        return ",".join(self.gpu_ids)


class LLMTrainer(BaseTrainer):
    """大语言模型训练器"""

    def __init__(
        self,
        gpu_ids: List[str],
        data_path: str,
        model_path: str,
        output_dir: str,
        params: LLMTrainingParams,
    ):
        super().__init__(gpu_ids, data_path, model_path, output_dir, params)
        self.params: LLMTrainingParams = params

    def build_command(self) -> List[str]:
        """构建LLM训练命令"""
        command = [
            "swift",
            "sft",
            "--model",
            self.model_path,
            "--dataset",
            self.data_path,
            "--output_dir",
            self.output_dir,
        ]

        # 添加训练参数
        command.extend(self.params.to_command_args())

        return command

    def build_environment(self) -> Dict[str, str]:
        """构建LLM训练环境变量"""
        env = os.environ.copy()

        # 设置GPU环境变量
        env["CUDA_VISIBLE_DEVICES"] = self.get_gpu_ids_str()

        # 设置NCCL环境变量（用于多GPU训练）
        if self.get_gpu_count() > 1:
            env["NCCL_P2P_DISABLE"] = "1"
            env["NCCL_IB_DISABLE"] = "1"
            env["NPROC_PER_NODE"] = str(self.get_gpu_count())

        return env

    def validate_params(self) -> bool:
        """验证LLM训练参数"""
        if not self.data_path or not os.path.exists(self.data_path):
            return False
        if not self.model_path:
            return False
        if not self.output_dir:
            return False
        if self.params.num_epochs <= 0:
            return False
        if self.params.batch_size <= 0:
            return False
        if self.params.learning_rate <= 0:
            return False
        return True

    def get_trainer_type(self) -> TrainerType:
        return TrainerType.LLM


class MLLMTrainer(BaseTrainer):
    """多模态大语言模型训练器"""

    def __init__(
        self,
        gpu_ids: List[str],
        data_path: str,
        model_path: str,
        output_dir: str,
        params: MLLMTrainingParams,
    ):
        super().__init__(gpu_ids, data_path, model_path, output_dir, params)
        self.params: MLLMTrainingParams = params

    def build_command(self) -> List[str]:
        """构建MLLM训练命令"""
        command = [
            "swift",
            "sft",
            "--model",
            self.model_path,
            "--dataset",
            self.data_path,
            "--output_dir",
            self.output_dir,
        ]

        # 添加训练参数
        command.extend(self.params.to_command_args())

        return command

    def build_environment(self) -> Dict[str, str]:
        """构建MLLM训练环境变量"""
        env = os.environ.copy()

        # 设置GPU环境变量
        env["CUDA_VISIBLE_DEVICES"] = self.get_gpu_ids_str()

        # 设置NCCL环境变量（用于多GPU训练）
        if self.get_gpu_count() > 1:
            env["NCCL_P2P_DISABLE"] = "1"
            env["NCCL_IB_DISABLE"] = "1"
            env["NPROC_PER_NODE"] = str(self.get_gpu_count())

        # MLLM特定的环境变量
        env["VISION_MODEL_ENABLED"] = "1"

        return env

    def validate_params(self) -> bool:
        """验证MLLM训练参数"""
        if not self.data_path or not os.path.exists(self.data_path):
            return False
        if not self.model_path:
            return False
        if not self.output_dir:
            return False
        if self.params.num_epochs <= 0:
            return False
        if self.params.batch_size <= 0:
            return False
        if self.params.learning_rate <= 0:
            return False
        if self.params.vit_lr <= 0:
            return False
        if self.params.aligner_lr <= 0:
            return False
        return True

    def get_trainer_type(self) -> TrainerType:
        return TrainerType.MLLM


class TrainerFactory:
    """训练器工厂类"""

    @staticmethod
    def create_trainer(
        trainer_type: TrainerType,
        gpu_ids: List[str],
        data_path: str,
        model_path: str,
        output_dir: str,
        params: BaseTrainingParams,
    ) -> BaseTrainer:
        """创建训练器实例"""

        if trainer_type == TrainerType.LLM:
            if not isinstance(params, LLMTrainingParams):
                raise ValueError("LLM训练器需要LLMTrainingParams参数")
            return LLMTrainer(gpu_ids, data_path, model_path, output_dir, params)

        elif trainer_type == TrainerType.MLLM:
            if not isinstance(params, MLLMTrainingParams):
                raise ValueError("MLLM训练器需要MLLMTrainingParams参数")
            return MLLMTrainer(gpu_ids, data_path, model_path, output_dir, params)

        else:
            raise ValueError(f"不支持的训练器类型: {trainer_type}")

    @staticmethod
    def create_params_from_dict(
        trainer_type: TrainerType, params_dict: Dict[str, Any]
    ) -> BaseTrainingParams:
        """从字典创建参数对象"""

        if trainer_type == TrainerType.LLM:
            return LLMTrainingParams(**params_dict)

        elif trainer_type == TrainerType.MLLM:
            return MLLMTrainingParams(**params_dict)

        else:
            raise ValueError(f"不支持的训练器类型: {trainer_type}")


# 训练参数联合类型
TrainingParams = Union[LLMTrainingParams, MLLMTrainingParams]
