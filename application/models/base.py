import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class BaseTrainingJob(BaseModel):
    """训练任务基础模型"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="任务ID")
    status: str = Field(default="pending", description="任务状态")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    started_at: Optional[datetime] = Field(default=None, description="开始时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")
    
    # 基础训练参数
    gpu_id: str = Field(..., description="GPU ID")
    output_dir: str = Field(..., description="输出目录")
    
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