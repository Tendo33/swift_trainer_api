import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class TrainingStatus(str, Enum):
    """训练任务状态枚举"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingTaskType(str, Enum):
    MULTIMODAL = "multimodal"
    LANGUAGE_MODEL = "language_model"
    DEPLOY = "deploy"


class MultiModalTrainParams(BaseModel):
    num_epochs: int = Field(default=1, description="训练轮数")
    batch_size: int = Field(default=1, description="批次大小")
    learning_rate: float = Field(default=1e-4, description="学习率")
    vit_lr: float = Field(default=1e-5, description="ViT学习率")
    aligner_lr: float = Field(default=1e-5, description="Aligner学习率")
    lora_rank: int = Field(default=16, description="LoRA rank")
    lora_alpha: int = Field(default=32, description="LoRA alpha")
    gradient_accumulation_steps: int = Field(default=4, description="梯度累积步数")
    eval_steps: int = Field(default=100, description="评估步数")
    save_steps: int = Field(default=100, description="保存步数")
    save_total_limit: int = Field(default=2, description="保存总数限制")
    logging_steps: int = Field(default=5, description="日志步数")
    max_length: int = Field(default=8192, description="最大长度")
    warmup_ratio: float = Field(default=0.05, description="预热比例")
    dataloader_num_workers: int = Field(default=4, description="数据加载器工作进程数")
    dataset_num_proc: int = Field(default=4, description="数据集处理进程数")
    save_only_model: bool = Field(default=True, description="仅保存模型")
    train_type: str = Field(default="lora", description="训练类型")
    torch_dtype: str = Field(default="bfloat16", description="PyTorch数据类型")


class LanguageModelTrainParams(BaseModel):
    num_epochs: int = Field(default=1, description="训练轮数")
    batch_size: int = Field(default=1, description="批次大小")
    learning_rate: float = Field(default=1e-4, description="学习率")
    gradient_accumulation_steps: int = Field(default=4, description="梯度累积步数")
    eval_steps: int = Field(default=100, description="评估步数")
    save_steps: int = Field(default=100, description="保存步数")
    save_total_limit: int = Field(default=2, description="保存总数限制")
    logging_steps: int = Field(default=5, description="日志步数")
    max_length: int = Field(default=2048, description="最大长度")
    warmup_ratio: float = Field(default=0.05, description="预热比例")
    dataloader_num_workers: int = Field(default=4, description="数据加载器工作进程数")
    dataset_num_proc: int = Field(default=4, description="数据集处理进程数")
    save_only_model: bool = Field(default=True, description="仅保存模型")
    train_type: str = Field(default="standard", description="训练类型")
    torch_dtype: str = Field(default="bfloat16", description="PyTorch数据类型")


class DeployParams(BaseModel):
    model_path: str = Field(..., description="待部署模型路径")
    deploy_target: str = Field(..., description="部署目标环境，如 local/k8s/cloud")
    version: Optional[str] = Field(default=None, description="部署版本号")
    resources: Optional[dict] = Field(default=None, description="资源需求")
    port: Optional[int] = Field(default=None, description="分配的部署端口")
    deploy_type: str = Field(default="llm", description="部署类型，如 llm/mllm")


class TrainingJobCreateRequest(BaseModel):
    """创建训练任务请求模型"""

    task_type: TrainingTaskType = Field(
        default=TrainingTaskType.MULTIMODAL, description="任务类型"
    )
    data_path: Optional[str] = Field(default=None, description="数据集路径")
    model_path: Optional[str] = Field(default=None, description="模型路径")
    output_dir: Optional[str] = Field(default=None, description="输出目录")
    priority: int = Field(
        default=0, ge=0, le=10, description="任务优先级，0-10，数字越大优先级越高"
    )
    train_params: Optional[
        Union[MultiModalTrainParams, LanguageModelTrainParams, DeployParams, dict]
    ] = Field(default=None, description="训练超参数配置")


class TrainingJob(BaseModel):
    """训练任务模型"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="任务ID")
    status: TrainingStatus = Field(
        default=TrainingStatus.PENDING, description="任务状态"
    )
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    started_at: Optional[datetime] = Field(default=None, description="开始时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")
    task_type: TrainingTaskType = Field(
        default=TrainingTaskType.MULTIMODAL, description="任务类型"
    )

    # 训练参数
    gpu_id: str = Field(..., description="GPU ID")
    data_path: str = Field(..., description="数据集路径")
    model_path: str = Field(..., description="模型路径")
    output_dir: str = Field(..., description="输出目录")

    # 训练配置
    num_epochs: int = Field(default=1, description="训练轮数")
    batch_size: int = Field(default=1, description="批次大小")
    learning_rate: float = Field(default=1e-4, description="学习率")
    vit_lr: float = Field(default=1e-5, description="ViT学习率")
    aligner_lr: float = Field(default=1e-5, description="Aligner学习率")
    lora_rank: int = Field(default=16, description="LoRA rank")
    lora_alpha: int = Field(default=32, description="LoRA alpha")
    gradient_accumulation_steps: int = Field(default=4, description="梯度累积步数")
    eval_steps: int = Field(default=100, description="评估步数")
    save_steps: int = Field(default=100, description="保存步数")
    save_total_limit: int = Field(default=2, description="保存总数限制")
    logging_steps: int = Field(default=5, description="日志步数")
    max_length: int = Field(default=8192, description="最大长度")
    warmup_ratio: float = Field(default=0.05, description="预热比例")
    dataloader_num_workers: int = Field(default=4, description="数据加载器工作进程数")
    dataset_num_proc: int = Field(default=4, description="数据集处理进程数")
    # deepspeed: str = Field(default="zero2", description="DeepSpeed配置")
    save_only_model: bool = Field(default=True, description="仅保存模型")
    train_type: str = Field(default="lora", description="训练类型")
    torch_dtype: str = Field(default="bfloat16", description="PyTorch数据类型")
    deploy_port: Optional[int] = Field(default=None, description="部署分配端口")

    # 运行时信息
    process_id: Optional[int] = Field(default=None, description="进程ID")
    log_file_path: Optional[str] = Field(default=None, description="日志文件路径")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    progress: float = Field(default=0.0, description="训练进度 (0-100)")

    # 训练结果
    final_loss: Optional[float] = Field(default=None, description="最终损失")
    training_time: Optional[float] = Field(default=None, description="训练时间(秒)")
    checkpoint_path: Optional[str] = Field(default=None, description="检查点路径")

    # 导出相关
    export_completed: bool = Field(default=False, description="导出是否完成")
    export_time: Optional[float] = Field(default=None, description="导出时间(秒)")
    export_path: Optional[str] = Field(default=None, description="导出模型路径")
    export_error: Optional[str] = Field(default=None, description="导出错误信息")


class TrainingJobResponse(BaseModel):
    """训练任务响应模型"""

    job_id: str = Field(..., description="任务ID")
    status: TrainingStatus = Field(..., description="任务状态")
    message: str = Field(..., description="响应消息")


class DeleteAllJobsResponse(BaseModel):
    """删除所有训练任务响应模型"""

    success: bool = Field(..., description="操作是否成功")
    deleted_count: int = Field(..., description="成功删除的任务数量")
    failed_count: int = Field(..., description="删除失败的任务数量")
    total_count: int = Field(..., description="总任务数量")
    message: str = Field(..., description="响应消息")


class TrainingJobListResponse(BaseModel):
    """训练任务列表响应模型"""

    jobs: List[TrainingJob] = Field(..., description="任务列表")
    total: int = Field(..., description="总数")
    page: int = Field(default=1, description="当前页")
    size: int = Field(default=10, description="每页大小")


class TrainingLogEntry(BaseModel):
    """训练日志条目模型"""

    timestamp: datetime = Field(..., description="时间戳")
    level: str = Field(..., description="日志级别")
    message: str = Field(..., description="日志消息")
    job_id: str = Field(..., description="任务ID")


class TrainingStatusResponse(BaseModel):
    """训练状态响应模型"""

    job_id: str = Field(..., description="任务ID")
    status: TrainingStatus = Field(..., description="任务状态")
    progress: float = Field(..., description="训练进度")
    current_epoch: Optional[int] = Field(default=None, description="当前轮数")
    current_step: Optional[int] = Field(default=None, description="当前步数")
    loss: Optional[float] = Field(default=None, description="当前损失")
    learning_rate: Optional[float] = Field(default=None, description="当前学习率")
    gpu_memory_usage: Optional[Dict[str, float]] = Field(
        default=None, description="GPU内存使用情况"
    )
    estimated_time_remaining: Optional[float] = Field(
        default=None, description="预计剩余时间(秒)"
    )
    created_at: datetime = Field(..., description="创建时间")
    started_at: Optional[datetime] = Field(default=None, description="开始时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")

    # 导出信息
    export_completed: Optional[bool] = Field(default=None, description="导出是否完成")
    export_time: Optional[float] = Field(default=None, description="导出时间(秒)")
    export_path: Optional[str] = Field(default=None, description="导出模型路径")
    export_error: Optional[str] = Field(default=None, description="导出错误信息")


class TrainingProgressResponse(BaseModel):
    """简化的训练进度响应模型"""

    job_id: str = Field(..., description="任务ID")
    progress: float = Field(..., description="训练进度 (0-100)")
    estimated_time_remaining: Optional[str] = Field(
        default=None, description="预计剩余时间"
    )


class GPUQueueStatusResponse(BaseModel):
    """GPU队列状态响应模型"""

    total_queued: int = Field(..., description="队列中的任务总数")
    queue_items: List[Dict[str, Any]] = Field(..., description="队列中的任务列表")


class TrainingJobQueuedResponse(BaseModel):
    """训练任务排队响应模型"""

    job_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    message: str = Field(..., description="响应消息")
    queue_position: Optional[int] = Field(default=None, description="队列位置")
    estimated_wait_time: Optional[str] = Field(default=None, description="预计等待时间")
