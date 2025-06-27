import time
from contextlib import contextmanager
from typing import Dict, List, Set

import redis

from application.config import settings
from application.exceptions import ResourceUnavailableError
from application.utils.logger import get_system_logger

logger = get_system_logger()


class ResourceManager:
    """分布式资源管理器"""
    
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD,
            decode_responses=True
        )
        self.logger = logger
        self._local_locks: Set[str] = set()  # 本地锁记录
    
    def acquire_gpu_lock(self, gpu_id: str, job_id: str, timeout: int = 30) -> bool:
        """获取GPU资源锁"""
        lock_key = f"gpu_lock:{gpu_id}"
        lock_value = f"{job_id}:{int(time.time())}"
        
        try:
            # 使用Redis SET NX EX 实现分布式锁
            result = self.redis_client.set(lock_key, lock_value, ex=timeout, nx=True)
            if result:
                self._local_locks.add(gpu_id)
                self.logger.info(f"成功获取GPU {gpu_id} 锁，任务ID: {job_id}")
                return True
            else:
                # 检查锁是否被当前任务持有
                current_value = self.redis_client.get(lock_key)
                if current_value and current_value.startswith(f"{job_id}:"):
                    self.logger.info(f"GPU {gpu_id} 锁已被当前任务 {job_id} 持有")
                    return True
                else:
                    self.logger.warning(f"GPU {gpu_id} 被其他任务占用")
                    return False
        except Exception as e:
            self.logger.error(f"获取GPU锁失败: {str(e)}")
            return False
    
    def release_gpu_lock(self, gpu_id: str, job_id: str) -> bool:
        """释放GPU资源锁"""
        lock_key = f"gpu_lock:{gpu_id}"
        
        try:
            # 使用Lua脚本确保原子性释放
            lua_script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            result = self.redis_client.eval(lua_script, 1, lock_key, f"{job_id}:")
            
            if result:
                self._local_locks.discard(gpu_id)
                self.logger.info(f"成功释放GPU {gpu_id} 锁，任务ID: {job_id}")
                return True
            else:
                self.logger.warning(f"释放GPU {gpu_id} 锁失败，可能已被其他任务占用")
                return False
        except Exception as e:
            self.logger.error(f"释放GPU锁失败: {str(e)}")
            return False
    
    def get_gpu_status(self, gpu_id: str) -> Dict[str, any]:
        """获取GPU状态信息"""
        lock_key = f"gpu_lock:{gpu_id}"
        
        try:
            lock_info = self.redis_client.get(lock_key)
            if lock_info:
                job_id, timestamp = lock_info.split(":", 1)
                return {
                    "gpu_id": gpu_id,
                    "status": "occupied",
                    "job_id": job_id,
                    "locked_at": int(timestamp),
                    "lock_duration": int(time.time()) - int(timestamp)
                }
            else:
                return {
                    "gpu_id": gpu_id,
                    "status": "available",
                    "job_id": None,
                    "locked_at": None,
                    "lock_duration": None
                }
        except Exception as e:
            self.logger.error(f"获取GPU状态失败: {str(e)}")
            return {
                "gpu_id": gpu_id,
                "status": "unknown",
                "error": str(e)
            }
    
    def get_all_gpu_status(self) -> List[Dict[str, any]]:
        """获取所有GPU状态"""
        try:
            # 获取所有GPU锁键
            lock_keys = self.redis_client.keys("gpu_lock:*")
            gpu_statuses = []
            
            for key in lock_keys:
                gpu_id = key.split(":", 1)[1]
                status = self.get_gpu_status(gpu_id)
                gpu_statuses.append(status)
            
            return gpu_statuses
        except Exception as e:
            self.logger.error(f"获取所有GPU状态失败: {str(e)}")
            return []
    
    def cleanup_expired_locks(self) -> int:
        """清理过期的锁"""
        try:
            lock_keys = self.redis_client.keys("gpu_lock:*")
            cleaned_count = 0
            
            for key in lock_keys:
                # 检查锁是否过期（超过1小时）
                ttl = self.redis_client.ttl(key)
                if ttl == -1:  # 没有设置过期时间
                    self.redis_client.delete(key)
                    cleaned_count += 1
                    self.logger.info(f"清理无过期时间的锁: {key}")
            
            return cleaned_count
        except Exception as e:
            self.logger.error(f"清理过期锁失败: {str(e)}")
            return 0
    
    @contextmanager
    def gpu_lock(self, gpu_id: str, job_id: str, timeout: int = 30):
        """GPU锁上下文管理器"""
        if not self.acquire_gpu_lock(gpu_id, job_id, timeout):
            raise ResourceUnavailableError("GPU", gpu_id, "无法获取GPU锁")
        
        try:
            yield
        finally:
            self.release_gpu_lock(gpu_id, job_id)
    
    def validate_gpu_ids(self, gpu_ids: str) -> List[str]:
        """验证GPU ID格式并返回清理后的列表"""
        if not gpu_ids:
            raise ValueError("GPU ID不能为空")
        
        cleaned_ids = []
        for gpu_id in gpu_ids.split(','):
            gpu_id = gpu_id.strip()
            if not gpu_id.isdigit():
                raise ValueError(f"无效的GPU ID: {gpu_id}")
            cleaned_ids.append(gpu_id)
        
        return cleaned_ids


# 全局资源管理器实例
resource_manager = ResourceManager()


def get_resource_manager() -> ResourceManager:
    """获取资源管理器实例"""
    return resource_manager 