import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class TrainingStatus(str, Enum):
    """训练任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingJobCreateRequest(BaseModel):
    """创建训练任务请求模型"""
    gpu_id: str = Field(..., description="GPU ID，如 '0' 或 '0,1,2'")
    data_path: str = Field(..., description="数据集路径")
    model_path: str = Field(..., description="模型路径")
    output_dir: str = Field(..., description="输出目录")
    
    # 可选参数，使用默认值
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
    deepspeed: str = Field(default="zero2", description="DeepSpeed配置")
    save_only_model: bool = Field(default=True, description="仅保存模型")
    
    # 高级参数
    train_type: str = Field(default="custom", description="训练类型")
    external_plugins: str = Field(
        default="examples/train/multimodal/lora_llm_full_vit/custom_plugin.py",
        description="外部插件路径"
    )
    torch_dtype: str = Field(default="bfloat16", description="PyTorch数据类型")
    
    @field_validator('gpu_id')
    def validate_gpu_id(cls, v):
        """验证GPU ID格式"""
        if not v:
            raise ValueError("GPU ID不能为空")
        # 检查是否为有效的GPU ID格式（数字，用逗号分隔）
        gpu_ids = v.split(',')
        for gpu_id in gpu_ids:
            if not gpu_id.strip().isdigit():
                raise ValueError(f"无效的GPU ID: {gpu_id}")
        return v


class TrainingJob(BaseModel):
    """训练任务模型"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="任务ID")
    status: TrainingStatus = Field(default=TrainingStatus.PENDING, description="任务状态")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    started_at: Optional[datetime] = Field(default=None, description="开始时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")
    
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
    deepspeed: str = Field(default="zero2", description="DeepSpeed配置")
    save_only_model: bool = Field(default=True, description="仅保存模型")
    train_type: str = Field(default="custom", description="训练类型")
    external_plugins: str = Field(
        default="examples/train/multimodal/lora_llm_full_vit/custom_plugin.py",
        description="外部插件路径"
    )
    torch_dtype: str = Field(default="bfloat16", description="PyTorch数据类型")
    
    # 运行时信息
    process_id: Optional[int] = Field(default=None, description="进程ID")
    log_file_path: Optional[str] = Field(default=None, description="日志文件路径")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    progress: float = Field(default=0.0, description="训练进度 (0-100)")
    
    # 训练结果
    final_loss: Optional[float] = Field(default=None, description="最终损失")
    training_time: Optional[float] = Field(default=None, description="训练时间(秒)")
    checkpoint_path: Optional[str] = Field(default=None, description="检查点路径")


class TrainingJobResponse(BaseModel):
    """训练任务响应模型"""
    job_id: str = Field(..., description="任务ID")
    status: TrainingStatus = Field(..., description="任务状态")
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
    gpu_memory_usage: Optional[Dict[str, float]] = Field(default=None, description="GPU内存使用情况")
    estimated_time_remaining: Optional[float] = Field(default=None, description="预计剩余时间(秒)")
    created_at: datetime = Field(..., description="创建时间")
    started_at: Optional[datetime] = Field(default=None, description="开始时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间") 