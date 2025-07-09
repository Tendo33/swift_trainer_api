from datetime import datetime

from application.models.training_model import TrainingJob, TrainingStatus
from application.services.redis_service import get_redis_service
from application.utils.logger import get_system_logger

from .deploy_handler import DeployHandler

logger = get_system_logger()

class DeployService:
    def __init__(self):
        self.redis_service = get_redis_service()
        self.logger = logger
        self.deploy_handler = DeployHandler()
        self.queue_processor_running = False
        self.queue_processor_thread = None

    def create_deploy_job(self, deploy_params):
        if hasattr(deploy_params, "dict"):
            params = deploy_params.model_dump()
        else:
            params = deploy_params
        port = self.deploy_handler.handle(params)
        job = TrainingJob(task_type="deploy", deploy_port=port, status=TrainingStatus.PENDING)
        self.save_deploy_job(job)
        self.redis_service.add_training_event(job.id, "deploy_job_created", f"部署任务 {job.id} 已创建，分配端口 {port}")
        return job

    def save_deploy_job(self, job: TrainingJob):
        self.redis_service.save_training_job(job)

    def start_deploy(self, job_id: str) -> bool:
        job = self.redis_service.get_training_job(job_id)
        if not job:
            raise ValueError(f"部署任务 {job_id} 不存在")
        if job.status != TrainingStatus.PENDING:
            raise ValueError(f"部署任务 {job_id} 状态不正确: {job.status}")
        job.status = TrainingStatus.RUNNING
        job.started_at = datetime.now()
        self.save_deploy_job(job)
        self.redis_service.add_training_event(job_id, "deploy_started", f"部署任务 {job_id} 已启动")
        self.logger.info(f"部署任务 {job_id} 已启动")
        return True

    def stop_deploy(self, job_id: str) -> bool:
        job = self.redis_service.get_training_job(job_id)
        if not job:
            raise ValueError(f"部署任务 {job_id} 不存在")
        if job.status not in [TrainingStatus.RUNNING, TrainingStatus.PENDING]:
            raise ValueError(f"部署任务 {job_id} 状态不正确: {job.status}")
        job.status = TrainingStatus.CANCELLED
        job.completed_at = datetime.now()
        self.save_deploy_job(job)
        self.redis_service.add_training_event(job_id, "deploy_cancelled", f"部署任务 {job_id} 已取消")
        self.logger.info(f"部署任务 {job_id} 已取消")
        return True

    def get_deploy_status(self, job_id: str):
        job = self.redis_service.get_training_job(job_id)
        if not job:
            self.logger.warning(f"部署任务 {job_id} 不存在")
            return None
        return {
            "job_id": job_id,
            "status": job.status,
            "deploy_port": job.deploy_port,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
        }

    # 队列相关接口（模拟实现，后续可完善）
    def get_queue_status(self):
        return self.redis_service.get_gpu_queue_status()  # 可自定义部署队列

    def remove_job_from_queue(self, job_id: str) -> bool:
        return self.redis_service.remove_job_from_gpu_queue(job_id)

    def start_queue_processor(self) -> bool:
        if self.queue_processor_running:
            self.logger.warning("部署队列处理器已经在运行")
            return True
        self.queue_processor_running = True
        # 可扩展线程处理逻辑
        self.logger.info("部署队列处理器已启动")
        return True

    def stop_queue_processor(self) -> bool:
        if not self.queue_processor_running:
            self.logger.warning("部署队列处理器未在运行")
            return True
        self.queue_processor_running = False
        self.logger.info("部署队列处理器已停止")
        return True

    def process_deploy_queue(self):
        # 模拟处理队列
        self.logger.info("处理部署队列（模拟）")
        return {"processed": 0, "started": 0, "failed": 0, "details": []} 