import json
from datetime import datetime
from typing import Any, Dict, List

import redis

from application.models.deploy_model import DeployJob
from application.models.training_model import TrainingJob, TrainingStatus
from application.setting import settings
from application.utils.gpu_utils import get_gpu_manager
from application.utils.logger import get_system_logger

logger = get_system_logger()


class RedisService:
    """Redis服务类，用于管理训练任务状态和缓存"""

    def __init__(self) -> None:
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        self.logger = logger

    def ping(self) -> bool:
        """测试Redis连接"""
        try:
            result = self.redis_client.ping()
            self.logger.debug("Redis连接测试成功")
            return result
        except Exception as e:
            self.logger.error(f"Redis连接失败: {str(e)}", exc_info=True)
            return False

    def save_training_job(self, job: TrainingJob) -> bool:
        """保存训练任务到Redis"""
        try:
            job_data = job.model_dump_json()
            key = f"training_job:{job.id}"
            self.redis_client.setex(key, 86400 * 30, job_data)  # 保存30天
            # self.logger.info(f"保存训练任务 {job.id} 到Redis")
            return True
        except (ConnectionError, TimeoutError) as e:
            self.logger.error(f"Redis连接失败，无法保存训练任务 {job.id}: {str(e)}")
            return False
        except (ValueError, TypeError) as e:
            self.logger.error(f"训练任务数据序列化失败 {job.id}: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"保存训练任务失败 {job.id}: {str(e)}", exc_info=True)
            return False

    def get_training_job(self, job_id: str) -> TrainingJob | None:
        """从Redis获取训练任务"""
        try:
            key = f"training_job:{job_id}"
            job_data = self.redis_client.get(key)
            if job_data:
                try:
                    return TrainingJob.model_validate_json(job_data)
                except Exception as e:
                    self.logger.error(f"解析训练任务数据失败 {job_id}: {str(e)}")
                    return None
            else:
                self.logger.debug(f"训练任务 {job_id} 在Redis中不存在")
                return None
        except Exception as e:
            self.logger.error(f"获取训练任务失败 {job_id}: {str(e)}", exc_info=True)
            return None

    def update_training_job(self, job_id: str, **kwargs: Any) -> bool:
        """更新训练任务"""
        try:
            job = self.get_training_job(job_id)
            if job is None:
                self.logger.warning(f"更新训练任务失败，任务不存在 {job_id}")
                return False

            # 更新字段
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
                else:
                    self.logger.warning(f"训练任务 {job_id} 没有字段 {key}")

            # 保存更新后的任务
            return self.save_training_job(job)
        except Exception as e:
            self.logger.error(f"更新训练任务失败 {job_id}: {str(e)}", exc_info=True)
            return False

    def delete_training_job(self, job_id: str) -> bool:
        """删除训练任务"""
        try:
            key = f"training_job:{job_id}"
            result = self.redis_client.delete(key)
            self.logger.info(f"删除训练任务 {job_id}")
            return result > 0
        except Exception as e:
            self.logger.error(f"删除训练任务失败 {job_id}: {str(e)}", exc_info=True)
            return False

    def delete_all_training_jobs(self) -> dict[str, Any]:
        """删除所有训练任务"""
        try:
            pattern = "training_job:*"
            keys = self.redis_client.keys(pattern)

            if not keys:
                self.logger.info("没有找到需要删除的训练任务")
                return {
                    "success": True,
                    "deleted_count": 0,
                    "failed_count": 0,
                    "total_count": 0,
                    "message": "没有找到需要删除的训练任务",
                }

            # 删除所有训练任务
            deleted_count = 0
            failed_count = 0

            for key in keys:
                try:
                    result = self.redis_client.delete(key)
                    if result > 0:
                        deleted_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    self.logger.error(f"删除训练任务失败 {key}: {str(e)}")
                    failed_count += 1

            # 同时删除相关的日志、进度、事件数据
            self._cleanup_related_data(keys)

            self.logger.info(
                f"批量删除训练任务完成: 成功 {deleted_count} 个, 失败 {failed_count} 个"
            )

            return {
                "success": True,
                "deleted_count": deleted_count,
                "failed_count": failed_count,
                "total_count": len(keys),
                "message": f"成功删除 {deleted_count} 个训练任务",
            }

        except Exception as e:
            self.logger.error(f"批量删除训练任务失败: {str(e)}", exc_info=True)
            return {
                "success": False,
                "deleted_count": 0,
                "failed_count": 0,
                "total_count": 0,
                "message": f"批量删除训练任务失败: {str(e)}",
            }

    def _cleanup_related_data(self, job_keys: List[str]) -> None:
        """清理与训练任务相关的数据（日志、进度、事件等）"""
        try:
            for key in job_keys:
                job_id = key.split(":", 1)[1]  # 提取job_id

                # 删除相关数据
                related_patterns = [
                    f"training_log:{job_id}",
                    f"training_progress:{job_id}",
                    f"training_events:{job_id}",
                ]

                for pattern in related_patterns:
                    try:
                        self.redis_client.delete(pattern)
                    except Exception as e:
                        self.logger.warning(f"清理相关数据失败 {pattern}: {str(e)}")

        except Exception as e:
            self.logger.error(f"清理相关数据失败: {str(e)}")

    def get_all_training_jobs(self) -> List[TrainingJob]:
        """获取所有训练任务"""
        try:
            pattern = "training_job:*"
            keys = self.redis_client.keys(pattern)
            jobs = []

            for key in keys:
                try:
                    job_data = self.redis_client.get(key)
                    if job_data:
                        try:
                            job = TrainingJob.model_validate_json(job_data)
                            jobs.append(job)
                        except Exception as e:
                            self.logger.error(f"解析训练任务数据失败 {key}: {str(e)}")
                            continue
                except Exception as e:
                    self.logger.error(f"获取训练任务数据失败 {key}: {str(e)}")
                    continue

            # 按创建时间排序
            jobs.sort(key=lambda x: x.created_at, reverse=True)
            return jobs
        except Exception as e:
            self.logger.error(f"获取所有训练任务失败: {str(e)}", exc_info=True)
            return []

    def get_jobs_by_status(self, status: TrainingStatus) -> List[TrainingJob]:
        """根据状态获取训练任务"""
        try:
            all_jobs = self.get_all_training_jobs()
            filtered_jobs = [job for job in all_jobs if job.status == status]
            self.logger.debug(f"根据状态 {status} 过滤得到 {len(filtered_jobs)} 个任务")
            return filtered_jobs
        except Exception as e:
            self.logger.error(
                f"根据状态获取训练任务失败 {status}: {str(e)}", exc_info=True
            )
            return []

    def save_training_log(self, job_id: str, log_entry: Dict[str, Any]) -> bool:
        """保存训练日志"""
        try:
            key = f"training_log:{job_id}"
            log_entry["timestamp"] = datetime.now().isoformat()
            log_data = json.dumps(log_entry, ensure_ascii=False)

            # 使用列表存储日志条目
            self.redis_client.lpush(key, log_data)
            # 限制日志条目数量，保留最新的1000条
            self.redis_client.ltrim(key, 0, 999)
            # 设置过期时间
            self.redis_client.expire(key, 86400 * 30)  # 30天

            return True
        except Exception as e:
            self.logger.error(f"保存训练日志失败 {job_id}: {str(e)}", exc_info=True)
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
                except json.JSONDecodeError as e:
                    self.logger.warning(f"解析训练日志数据失败 {job_id}: {str(e)}")
                    continue

            return logs
        except Exception as e:
            self.logger.error(f"获取训练日志失败 {job_id}: {str(e)}", exc_info=True)
            return []

    def save_training_progress(
        self, job_id: str, progress: float, **kwargs: Any
    ) -> bool:
        """保存训练进度"""
        try:
            key = f"training_progress:{job_id}"
            progress_data = {
                "progress": progress,
                "timestamp": datetime.now().isoformat(),
                **kwargs,
            }

            self.redis_client.setex(key, 86400 * 30, json.dumps(progress_data))
            return True
        except Exception as e:
            self.logger.error(f"保存训练进度失败 {job_id}: {str(e)}", exc_info=True)
            return False

    def get_training_progress(self, job_id: str) -> Dict[str, Any] | None:
        """获取训练进度"""
        try:
            key = f"training_progress:{job_id}"
            progress_data = self.redis_client.get(key)
            if progress_data:
                try:
                    return json.loads(progress_data)
                except json.JSONDecodeError as e:
                    self.logger.error(f"解析训练进度数据失败 {job_id}: {str(e)}")
                    return None
            else:
                self.logger.debug(f"训练进度 {job_id} 在Redis中不存在")
                return None
        except Exception as e:
            self.logger.error(f"获取训练进度失败 {job_id}: {str(e)}", exc_info=True)
            return None

    def set_gpu_status(
        self, gpu_id: str, status: str, job_id: str | None = None
    ) -> bool:
        """设置GPU状态"""
        try:
            key = f"gpu_status:{gpu_id}"
            status_data = {
                "status": status,
                "job_id": job_id,
                "timestamp": datetime.now().isoformat(),
            }

            self.redis_client.setex(key, 3600, json.dumps(status_data))  # 1小时过期
            return True
        except Exception as e:
            self.logger.error(f"设置GPU状态失败 {gpu_id}: {str(e)}", exc_info=True)
            return False

    def get_gpu_status(self, gpu_id: str) -> Dict[str, Any] | None:
        """获取GPU状态"""
        try:
            key = f"gpu_status:{gpu_id}"
            status_data = self.redis_client.get(key)
            if status_data:
                try:
                    return json.loads(status_data)
                except json.JSONDecodeError as e:
                    self.logger.error(f"解析GPU状态数据失败 {gpu_id}: {str(e)}")
                    return None
            else:
                self.logger.debug(f"GPU状态 {gpu_id} 在Redis中不存在")
                return None
        except Exception as e:
            self.logger.error(f"获取GPU状态失败 {gpu_id}: {str(e)}", exc_info=True)
            return None

    def get_all_gpu_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有GPU状态"""
        try:
            pattern = "gpu_status:*"
            keys = self.redis_client.keys(pattern)
            gpu_status = {}

            for key in keys:
                gpu_id = key.split(":")[1]
                status_data = self.redis_client.get(key)
                if status_data:
                    gpu_status[gpu_id] = json.loads(status_data)

            return gpu_status
        except Exception as e:
            self.logger.error(f"获取所有GPU状态失败: {str(e)}")
            return {}

    def add_training_event(
        self, job_id: str, event_type: str, message: str, **kwargs: Any
    ) -> bool:
        """添加训练事件"""
        try:
            key = f"training_events:{job_id}"
            event_data = {
                "type": event_type,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                **kwargs,
            }

            self.redis_client.lpush(key, json.dumps(event_data, ensure_ascii=False))
            self.redis_client.ltrim(key, 0, 999)  # 保留最新1000条事件
            self.redis_client.expire(key, 86400 * 30)  # 30天过期

            return True
        except Exception as e:
            self.logger.error(f"添加训练事件失败 {job_id}: {str(e)}", exc_info=True)
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
                except json.JSONDecodeError as e:
                    self.logger.warning(f"解析训练事件数据失败 {job_id}: {str(e)}")
                    continue

            return events
        except Exception as e:
            self.logger.error(f"获取训练事件失败 {job_id}: {str(e)}", exc_info=True)
            return []

    def cleanup_expired_data(self) -> int:
        """清理过期数据"""
        try:
            # 这里可以实现清理逻辑，比如删除过期的任务、日志等
            # 目前Redis会自动处理过期数据
            self.logger.debug("清理过期数据完成")
            return 0
        except Exception as e:
            self.logger.error(f"清理过期数据失败: {str(e)}", exc_info=True)
            return 0

    # GPU队列管理相关方法
    def add_job_to_gpu_queue(
        self, job_id: str, gpu_ids: List[str], priority: int = 0
    ) -> bool:
        """将任务添加到GPU队列"""
        try:
            queue_data = {
                "job_id": job_id,
                "gpu_ids": gpu_ids,
                "priority": priority,
                "created_at": datetime.now().isoformat(),
                "status": "queued",
            }

            # 使用有序集合存储队列，按优先级排序
            score = priority * 1000000 + int(datetime.now().timestamp())
            self.redis_client.zadd("gpu_queue", {job_id: score})

            # 存储队列详细信息
            self.redis_client.setex(
                f"gpu_queue_detail:{job_id}",
                86400 * 30,  # 30天过期
                json.dumps(queue_data),
            )

            self.logger.info(
                f"任务 {job_id} 已添加到GPU队列，优先级: {priority}, GPU: {gpu_ids}"
            )
            return True
        except Exception as e:
            self.logger.error(f"添加任务到GPU队列失败 {job_id}: {str(e)}")
            return False

    def get_job_from_gpu_queue(self) -> Dict[str, Any] | None:
        """从GPU队列获取下一个可执行的任务"""
        try:
            # 获取队列中优先级最高的任务
            queue_items = self.redis_client.zrange("gpu_queue", 0, 0, withscores=True)
            if not queue_items:
                return None

            job_id = queue_items[0][0]
            score = queue_items[0][1]

            # 获取任务详细信息
            detail_key = f"gpu_queue_detail:{job_id}"
            detail_data = self.redis_client.get(detail_key)
            if not detail_data:
                # 清理无效的队列项
                self.redis_client.zrem("gpu_queue", job_id)
                return None

            queue_data = json.loads(detail_data)

            # 检查任务是否仍然存在且状态为pending
            job = self.get_training_job(job_id)
            if not job or job.status != TrainingStatus.PENDING:
                # 清理无效的队列项
                self.redis_client.zrem("gpu_queue", job_id)
                self.redis_client.delete(detail_key)
                return None

            return {"job_id": job_id, "queue_data": queue_data, "score": score}
        except Exception as e:
            self.logger.error(f"从GPU队列获取任务失败: {str(e)}")
            return None

    def remove_job_from_gpu_queue(self, job_id: str) -> bool:
        """从GPU队列中移除任务"""
        try:
            # 从有序集合中移除
            self.redis_client.zrem("gpu_queue", job_id)

            # 删除详细信息
            self.redis_client.delete(f"gpu_queue_detail:{job_id}")

            self.logger.info(f"任务 {job_id} 已从GPU队列中移除")
            return True
        except Exception as e:
            self.logger.error(f"从GPU队列移除任务失败 {job_id}: {str(e)}")
            return False

    def get_gpu_queue_status(self) -> Dict[str, Any]:
        """获取GPU队列状态"""
        try:
            # 获取队列中的所有任务
            queue_items = self.redis_client.zrange("gpu_queue", 0, -1, withscores=True)

            queue_details = []
            for job_id, score in queue_items:
                detail_key = f"gpu_queue_detail:{job_id}"
                detail_data = self.redis_client.get(detail_key)
                if detail_data:
                    queue_data = json.loads(detail_data)
                    # 获取任务基本信息
                    job = self.get_training_job(job_id)
                    if job:
                        queue_details.append(
                            {
                                "job_id": job_id,
                                "gpu_ids": queue_data.get("gpu_ids", []),
                                "priority": queue_data.get("priority", 0),
                                "created_at": queue_data.get("created_at"),
                                "queue_position": len(queue_details) + 1,
                                "job_status": job.status.value,
                                "job_created_at": job.created_at.isoformat(),
                            }
                        )

            return {"total_queued": len(queue_details), "queue_items": queue_details}
        except Exception as e:
            self.logger.error(f"获取GPU队列状态失败: {str(e)}")
            return {"total_queued": 0, "queue_items": []}

    def check_gpu_availability_for_queue(self, gpu_ids: List[str]) -> Dict[str, Any]:
        """检查指定GPU是否可用于队列中的任务"""
        try:
            gpu_manager = get_gpu_manager()
            available_gpus = []
            unavailable_gpus = []

            for gpu_id in gpu_ids:
                # 使用GPU管理器检查GPU可用性（包括显存使用率检查）
                if gpu_manager.check_gpu_availability(gpu_id):
                    available_gpus.append(gpu_id)
                else:
                    unavailable_gpus.append(gpu_id)

            return {
                "available_gpus": available_gpus,
                "unavailable_gpus": unavailable_gpus,
                "can_start": len(available_gpus) == len(gpu_ids),
            }
        except Exception as e:
            self.logger.error(f"检查GPU可用性失败: {str(e)}")
            return {
                "available_gpus": [],
                "unavailable_gpus": gpu_ids,
                "can_start": False,
            }

    def save_deploy_job(self, job: DeployJob) -> None:
        key = f"deploy:job:{job.id}"
        self.redis_client.set(key, job.json())

    def get_deploy_job(self, job_id: str) -> DeployJob | None:
        key = f"deploy:job:{job_id}"
        data = self.redis_client.get(key)
        if data:
            return DeployJob.model_validate_json(data)
        return None

    def add_deploy_event(self, job_id: str, event_type: str, message: str) -> None:
        key = f"deploy:event:{job_id}"
        event = {
            "type": event_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        self.redis_client.rpush(key, json.dumps(event))

    def add_job_to_deploy_queue(self, job_id: str, port: int, priority: int) -> None:
        queue_key = "deploy:queue"
        queue_data = json.dumps(
            {
                "job_id": job_id,
                "port": port,
                "priority": priority,
                "created_at": datetime.now().isoformat(),
            }
        )
        self.redis_client.rpush(queue_key, queue_data)

    def get_deploy_queue_status(self) -> List[Dict[str, Any]]:
        queue_key = "deploy:queue"
        items = self.redis_client.lrange(queue_key, 0, -1)
        return [json.loads(item) for item in items]

    def remove_job_from_deploy_queue(self, job_id: str) -> bool:
        queue_key = "deploy:queue"
        items = self.redis_client.lrange(queue_key, 0, -1)
        for item in items:
            data = json.loads(item)
            if data["job_id"] == job_id:
                self.redis_client.lrem(queue_key, 0, item)
                return True
        return False


# 创建全局Redis服务实例
redis_service = RedisService()


def get_redis_service() -> RedisService:
    """获取Redis服务实例"""
    return redis_service
