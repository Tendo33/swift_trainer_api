from datetime import datetime
from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field


class DeployJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DeployJobCreateRequest(BaseModel):
    model_path: str = Field(..., description="待部署模型路径")
    deploy_target: str = Field(..., description="部署目标环境，如 local/k8s/cloud")
    version: Optional[str] = Field(default=None, description="部署版本号")
    resources: Optional[Dict] = Field(default=None, description="资源需求")
    port: Optional[int] = Field(default=None, description="分配的部署端口")
    deploy_type: str = Field(default="llm", description="部署类型，如 llm/mllm")


class DeployJob(BaseModel):
    id: str = Field(..., description="部署任务ID")
    status: DeployJobStatus = Field(
        default=DeployJobStatus.PENDING, description="任务状态"
    )
    model_path: str = Field(..., description="待部署模型路径")
    deploy_target: str = Field(..., description="部署目标环境")
    version: Optional[str] = Field(default=None, description="部署版本号")
    resources: Optional[Dict] = Field(default=None, description="资源需求")
    port: Optional[int] = Field(default=None, description="分配的部署端口")
    deploy_type: str = Field(default="llm", description="部署类型")
    process_id: Optional[int] = Field(default=None, description="部署进程ID")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    started_at: Optional[datetime] = Field(default=None, description="开始时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")
    error_message: Optional[str] = Field(default=None, description="错误信息")
