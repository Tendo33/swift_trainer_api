import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis

from application.config import settings
from application.models.training import LLMTrainingJob, TrainingJob, TrainingStatus
from application.utils.logger import get_system_logger

logger = get_system_logger()


class RedisService:
    """Redis服务类，用于管理训练任务状态和缓存"""
    
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        self.logger = logger
    
    def ping(self) -> bool:
        """测试Redis连接"""
        try:
            return self.redis_client.ping()
        except Exception as e:
            self.logger.error(f"Redis连接失败: {str(e)}")
            return False
    
    def save_training_job(self, job: TrainingJob) -> bool:
        """保存VLM训练任务到Redis"""
        try:
            job_data = job.model_dump_json()
            key = f"training_job:{job.id}"
            self.redis_client.setex(key, 86400 * 7, job_data)  # 保存7天
            self.logger.info(f"保存VLM训练任务 {job.id} 到Redis")
            return True
        except Exception as e:
            self.logger.error(f"保存VLM训练任务失败: {str(e)}")
            return False
    
    def save_llm_training_job(self, job: LLMTrainingJob) -> bool:
        """保存LLM训练任务到Redis"""
        try:
            job_data = job.model_dump_json()
            key = f"llm_training_job:{job.id}"
            self.redis_client.setex(key, 86400 * 7, job_data)  # 保存7天
            self.logger.info(f"保存LLM训练任务 {job.id} 到Redis")
            return True
        except Exception as e:
            self.logger.error(f"保存LLM训练任务失败: {str(e)}")
            return False
    
    def get_training_job(self, job_id: str) -> Optional[TrainingJob]:
        """从Redis获取VLM训练任务"""
        try:
            key = f"training_job:{job_id}"
            job_data = self.redis_client.get(key)
            if job_data:
                return TrainingJob.model_validate_json(job_data)
            return None
        except Exception as e:
            self.logger.error(f"获取VLM训练任务失败: {str(e)}")
            return None
    
    def get_llm_training_job(self, job_id: str) -> Optional[LLMTrainingJob]:
        """从Redis获取LLM训练任务"""
        try:
            key = f"llm_training_job:{job_id}"
            job_data = self.redis_client.get(key)
            if job_data:
                return LLMTrainingJob.model_validate_json(job_data)
            return None
        except Exception as e:
            self.logger.error(f"获取LLM训练任务失败: {str(e)}")
            return None
    
    def update_training_job(self, job_id: str, **kwargs) -> bool:
        """更新VLM训练任务"""
        try:
            job = self.get_training_job(job_id)
            if job is None:
                return False
            
            # 更新字段
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            
            # 保存更新后的任务
            return self.save_training_job(job)
        except Exception as e:
            self.logger.error(f"更新VLM训练任务失败: {str(e)}")
            return False
    
    def update_llm_training_job(self, job_id: str, **kwargs) -> bool:
        """更新LLM训练任务"""
        try:
            job = self.get_llm_training_job(job_id)
            if job is None:
                return False
            
            # 更新字段
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            
            # 保存更新后的任务
            return self.save_llm_training_job(job)
        except Exception as e:
            self.logger.error(f"更新LLM训练任务失败: {str(e)}")
            return False
    
    def delete_training_job(self, job_id: str) -> bool:
        """删除VLM训练任务"""
        try:
            key = f"training_job:{job_id}"
            result = self.redis_client.delete(key)
            self.logger.info(f"删除VLM训练任务 {job_id}")
            return result > 0
        except Exception as e:
            self.logger.error(f"删除VLM训练任务失败: {str(e)}")
            return False
    
    def delete_llm_training_job(self, job_id: str) -> bool:
        """删除LLM训练任务"""
        try:
            key = f"llm_training_job:{job_id}"
            result = self.redis_client.delete(key)
            self.logger.info(f"删除LLM训练任务 {job_id}")
            return result > 0
        except Exception as e:
            self.logger.error(f"删除LLM训练任务失败: {str(e)}")
            return False
    
    def get_all_training_jobs(self) -> List[TrainingJob | LLMTrainingJob]:
        """获取所有训练任务（VLM和LLM）"""
        try:
            vlm_jobs = self._get_all_vlm_training_jobs()
            llm_jobs = self._get_all_llm_training_jobs()
            
            # 合并任务列表
            all_jobs = []
            for job in vlm_jobs:
                all_jobs.append(job)
            for job in llm_jobs:
                all_jobs.append(job)
            
            # 按创建时间排序
            all_jobs.sort(key=lambda x: x.created_at, reverse=True)
            return all_jobs
        except Exception as e:
            self.logger.error(f"获取所有训练任务失败: {str(e)}")
            return []
    
    def _get_all_vlm_training_jobs(self) -> List[TrainingJob]:
        """获取所有VLM训练任务"""
        try:
            pattern = "training_job:*"
            keys = self.redis_client.keys(pattern)
            jobs = []
            
            for key in keys:
                job_data = self.redis_client.get(key)
                if job_data:
                    job = TrainingJob.model_validate_json(job_data)
                    jobs.append(job)
            
            # 按创建时间排序
            jobs.sort(key=lambda x: x.created_at, reverse=True)
            return jobs
        except Exception as e:
            self.logger.error(f"获取所有VLM训练任务失败: {str(e)}")
            return []
    
    def _get_all_llm_training_jobs(self) -> List[LLMTrainingJob]:
        """获取所有LLM训练任务"""
        try:
            pattern = "llm_training_job:*"
            keys = self.redis_client.keys(pattern)
            jobs = []
            
            for key in keys:
                job_data = self.redis_client.get(key)
                if job_data:
                    job = LLMTrainingJob.model_validate_json(job_data)
                    jobs.append(job)
            
            # 按创建时间排序
            jobs.sort(key=lambda x: x.created_at, reverse=True)
            return jobs
        except Exception as e:
            self.logger.error(f"获取所有LLM训练任务失败: {str(e)}")
            return []
    
    def get_jobs_by_status(self, status: TrainingStatus) -> List[TrainingJob | LLMTrainingJob]:
        """根据状态获取训练任务（VLM和LLM）"""
        try:
            all_jobs = self.get_all_training_jobs()
            return [job for job in all_jobs if job.status == status]
        except Exception as e:
            self.logger.error(f"根据状态获取训练任务失败: {str(e)}")
            return []
    
    def save_training_log(self, job_id: str, log_entry: Dict[str, Any]) -> bool:
        """保存训练日志"""
        try:
            key = f"training_log:{job_id}"
            log_entry['timestamp'] = datetime.now().isoformat()
            log_data = json.dumps(log_entry, ensure_ascii=False)
            
            # 使用列表存储日志条目
            self.redis_client.lpush(key, log_data)
            # 限制日志条目数量，保留最新的1000条
            self.redis_client.ltrim(key, 0, 999)
            # 设置过期时间
            self.redis_client.expire(key, 86400 * 7)  # 7天
            
            return True
        except Exception as e:
            self.logger.error(f"保存训练日志失败: {str(e)}")
            return False
    
    def get_training_logs(self, job_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取训练日志"""
        try:
            key = f"training_log:{job_id}"
            log_data_list = self.redis_client.lrange(key, 0, limit - 1)
            
            logs = []
            for log_data in log_data_list:
                try:
                    log_entry = json.loads(log_data)
                    logs.append(log_entry)
                except json.JSONDecodeError:
                    continue
            
            return logs
        except Exception as e:
            self.logger.error(f"获取训练日志失败: {str(e)}")
            return []
    
    def save_training_progress(self, job_id: str, progress: float, **kwargs) -> bool:
        """保存训练进度"""
        try:
            key = f"training_progress:{job_id}"
            progress_data = {
                'progress': progress,
                'timestamp': datetime.now().isoformat(),
                **kwargs
            }
            
            self.redis_client.setex(key, 86400 * 7, json.dumps(progress_data))
            return True
        except Exception as e:
            self.logger.error(f"保存训练进度失败: {str(e)}")
            return False
    
    def get_training_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        """获取训练进度"""
        try:
            key = f"training_progress:{job_id}"
            progress_data = self.redis_client.get(key)
            if progress_data:
                return json.loads(progress_data)
            return None
        except Exception as e:
            self.logger.error(f"获取训练进度失败: {str(e)}")
            return None
    
    def set_gpu_status(self, gpu_id: str, status: str, job_id: Optional[str] = None) -> bool:
        """设置GPU状态"""
        try:
            key = f"gpu_status:{gpu_id}"
            status_data = {
                'status': status,
                'job_id': job_id,
                'timestamp': datetime.now().isoformat()
            }
            
            self.redis_client.setex(key, 3600, json.dumps(status_data))  # 1小时过期
            return True
        except Exception as e:
            self.logger.error(f"设置GPU状态失败: {str(e)}")
            return False
    
    def get_gpu_status(self, gpu_id: str) -> Optional[Dict[str, Any]]:
        """获取GPU状态"""
        try:
            key = f"gpu_status:{gpu_id}"
            status_data = self.redis_client.get(key)
            if status_data:
                return json.loads(status_data)
            return None
        except Exception as e:
            self.logger.error(f"获取GPU状态失败: {str(e)}")
            return None
    
    def get_all_gpu_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有GPU状态"""
        try:
            pattern = "gpu_status:*"
            keys = self.redis_client.keys(pattern)
            gpu_status = {}
            
            for key in keys:
                gpu_id = key.split(':')[1]
                status_data = self.redis_client.get(key)
                if status_data:
                    gpu_status[gpu_id] = json.loads(status_data)
            
            return gpu_status
        except Exception as e:
            self.logger.error(f"获取所有GPU状态失败: {str(e)}")
            return {}
    
    def add_training_event(self, job_id: str, event_type: str, message: str, **kwargs) -> bool:
        """添加训练事件"""
        try:
            key = f"training_events:{job_id}"
            event_data = {
                'type': event_type,
                'message': message,
                'timestamp': datetime.now().isoformat(),
                **kwargs
            }
            
            self.redis_client.lpush(key, json.dumps(event_data, ensure_ascii=False))
            self.redis_client.ltrim(key, 0, 999)  # 保留最新1000条事件
            self.redis_client.expire(key, 86400 * 7)  # 7天过期
            
            return True
        except Exception as e:
            self.logger.error(f"添加训练事件失败: {str(e)}")
            return False
    
    def get_training_events(self, job_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """获取训练事件"""
        try:
            key = f"training_events:{job_id}"
            event_data_list = self.redis_client.lrange(key, 0, limit - 1)
            
            events = []
            for event_data in event_data_list:
                try:
                    event = json.loads(event_data)
                    events.append(event)
                except json.JSONDecodeError:
                    continue
            
            return events
        except Exception as e:
            self.logger.error(f"获取训练事件失败: {str(e)}")
            return []
    
    def cleanup_expired_data(self) -> int:
        """清理过期数据"""
        try:
            # 这里可以实现清理逻辑，比如删除过期的任务、日志等
            # 目前Redis会自动处理过期数据
            return 0
        except Exception as e:
            self.logger.error(f"清理过期数据失败: {str(e)}")
            return 0


# 创建全局Redis服务实例
redis_service = RedisService()


def get_redis_service() -> RedisService:
    """获取Redis服务实例"""
    return redis_service 