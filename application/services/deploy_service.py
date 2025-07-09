import os
import subprocess
import uuid
from datetime import datetime

from application.models.deploy_model import (
    DeployJob,
    DeployJobCreateRequest,
    DeployJobStatus,
)
from application.services.redis_service import get_redis_service
from application.utils.logger import get_system_logger

from .deploy_handler import DeployHandler

logger = get_system_logger()


class DeployService:
    def __init__(self):
        self.logger = logger
        self.deploy_handler = DeployHandler()
        self.queue_processor_running = False
        self.queue_processor_thread = None
        self.redis_service = get_redis_service()

    def create_deploy_job(self, request: DeployJobCreateRequest) -> DeployJob:
        job_id = str(uuid.uuid4())
        port = request.port or self.deploy_handler.handle(request)
        job = DeployJob(
            id=job_id,
            status=DeployJobStatus.PENDING,
            model_path=request.model_path,
            deploy_target=request.deploy_target,
            version=request.version,
            resources=request.resources,
            port=port,
            deploy_type=request.deploy_type,
        )
        self.redis_service.save_deploy_job(job)
        self.redis_service.add_deploy_event(
            job.id, "deploy_job_created", f"部署任务 {job.id} 已创建，分配端口 {port}"
        )
        return job

    def start_deploy(self, job_id: str) -> bool:
        job = self.redis_service.get_deploy_job(job_id)
        if not job:
            raise ValueError(f"部署任务 {job_id} 不存在")
        if job.status != DeployJobStatus.PENDING:
            raise ValueError(f"部署任务 {job_id} 状态不正确: {job.status}")
        if job.deploy_type == "llm":
            self._start_llm_deploy(job)
        elif job.deploy_type == "mllm":
            self._start_mllm_deploy(job)
        else:
            raise ValueError(f"不支持的部署类型: {job.deploy_type}")
        return True

    def _start_llm_deploy(self, job: DeployJob):
        cmd = [
            "swift",
            "deploy",
            "--model",
            job.model_path,
            "--infer_backend",
            "vllm",
            "--gpu_memory_utilization",
            "0.9",
            "--max_model_len",
            "8192",
            "--max_new_tokens",
            "2048",
            "--agent_template",
            "hermes",
            "--served_model_name",
            job.model_path.split("/")[-1] if "/" in job.model_path else job.model_path,
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
        process = subprocess.Popen(cmd, env=env)
        job.process_id = process.pid
        job.status = DeployJobStatus.RUNNING
        job.started_at = datetime.now()
        self.redis_service.save_deploy_job(job)
        self.redis_service.add_deploy_event(
            job.id, "deploy_started", f"LLM部署进程已启动，PID={process.pid}"
        )
        self.logger.info(f"LLM部署进程已启动，PID={process.pid}")

    def _start_mllm_deploy(self, job: DeployJob):
        cmd = [
            "swift",
            "deploy",
            "--model",
            job.model_path,
            "--infer_backend",
            "vllm",
            "--gpu_memory_utilization",
            "0.9",
            "--max_model_len",
            "8192",
            "--max_new_tokens",
            "2048",
            "--agent_template",
            "hermes",
            "--served_model_name",
            job.model_path.split("/")[-1] if "/" in job.model_path else job.model_path,
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
        process = subprocess.Popen(cmd, env=env)
        job.process_id = process.pid
        job.status = DeployJobStatus.RUNNING
        job.started_at = datetime.now()
        self.redis_service.save_deploy_job(job)
        self.redis_service.add_deploy_event(
            job.id, "deploy_started", f"MLLM部署进程已启动，PID={process.pid}"
        )
        self.logger.info(f"MLLM部署进程已启动，PID={process.pid}")

    def stop_deploy(self, job_id: str) -> bool:
        job = self.redis_service.get_deploy_job(job_id)
        if not job:
            raise ValueError(f"部署任务 {job_id} 不存在")
        if job.status not in [DeployJobStatus.RUNNING, DeployJobStatus.PENDING]:
            raise ValueError(f"部署任务 {job_id} 状态不正确: {job.status}")
        if job.process_id:
            try:
                import psutil

                p = psutil.Process(job.process_id)
                p.terminate()
            except Exception as e:
                self.logger.warning(f"终止进程失败: {e}")
        job.status = DeployJobStatus.CANCELLED
        job.completed_at = datetime.now()
        self.redis_service.save_deploy_job(job)
        self.redis_service.add_deploy_event(
            job.id, "deploy_cancelled", f"部署任务 {job.id} 已取消"
        )
        self.logger.info(f"部署任务 {job.id} 已取消")
        return True

    def get_deploy_status(self, job_id: str):
        job = self.redis_service.get_deploy_job(job_id)
        if not job:
            self.logger.warning(f"部署任务 {job_id} 不存在")
            return None
        return {
            "job_id": job_id,
            "status": job.status,
            "deploy_port": job.port,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "process_id": job.process_id,
            "deploy_type": job.deploy_type,
        }

    def get_queue_status(self):
        return self.redis_service.get_deploy_queue_status()

    def add_job_to_queue(self, job_id: str, port: int, priority: int):
        return self.redis_service.add_job_to_deploy_queue(job_id, port, priority)

    def remove_job_from_queue(self, job_id: str) -> bool:
        return self.redis_service.remove_job_from_deploy_queue(job_id)
