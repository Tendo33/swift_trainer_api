import os
import sys

from application.config import settings
from loguru import logger


class TrainingLogger:
    """训练日志管理器"""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.log_file_path = os.path.join(settings.LOG_DIR, f"training_{job_id}.log")
        self.json_log_file_path = os.path.join(settings.LOG_DIR, f"training_{job_id}.jsonl")
        
        # 配置日志记录器
        self._setup_logger()
    
    def _setup_logger(self):
        """设置日志记录器"""
        # 移除默认的日志处理器
        logger.remove()
        
        # 添加控制台输出
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=settings.LOG_LEVEL,
            colorize=True
        )
        
        # 添加文件输出
        logger.add(
            self.log_file_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=settings.LOG_LEVEL,
            rotation="100 MB",
            retention="30 days",
            compression="zip"
        )
        
        # 添加JSON格式日志（用于结构化日志分析）
        logger.add(
            self.json_log_file_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level=settings.LOG_LEVEL,
            serialize=True,
            rotation="100 MB",
            retention="30 days"
        )
    
    def info(self, message: str, **kwargs):
        """记录信息日志"""
        extra_data = {"job_id": self.job_id, **kwargs}
        logger.bind(**extra_data).info(message)
    
    def warning(self, message: str, **kwargs):
        """记录警告日志"""
        extra_data = {"job_id": self.job_id, **kwargs}
        logger.bind(**extra_data).warning(message)
    
    def error(self, message: str, **kwargs):
        """记录错误日志"""
        extra_data = {"job_id": self.job_id, **kwargs}
        logger.bind(**extra_data).error(message)
    
    def debug(self, message: str, **kwargs):
        """记录调试日志"""
        extra_data = {"job_id": self.job_id, **kwargs}
        logger.bind(**extra_data).debug(message)
    
    def training_progress(self, epoch: int, step: int, loss: float, learning_rate: float, **kwargs):
        """记录训练进度"""
        message = f"Epoch {epoch}, Step {step}, Loss: {loss:.6f}, LR: {learning_rate:.2e}"
        extra_data = {
            "job_id": self.job_id,
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "learning_rate": learning_rate,
            "log_type": "training_progress",
            **kwargs
        }
        logger.bind(**extra_data).info(message)
    
    def gpu_usage(self, gpu_id: str, memory_used: float, memory_total: float, **kwargs):
        """记录GPU使用情况"""
        memory_percent = (memory_used / memory_total) * 100
        message = f"GPU {gpu_id}: {memory_used:.1f}MB / {memory_total:.1f}MB ({memory_percent:.1f}%)"
        extra_data = {
            "job_id": self.job_id,
            "gpu_id": gpu_id,
            "memory_used": memory_used,
            "memory_total": memory_total,
            "memory_percent": memory_percent,
            "log_type": "gpu_usage",
            **kwargs
        }
        logger.bind(**extra_data).info(message)
    
    def checkpoint_saved(self, checkpoint_path: str, step: int, **kwargs):
        """记录检查点保存"""
        message = f"Checkpoint saved at step {step}: {checkpoint_path}"
        extra_data = {
            "job_id": self.job_id,
            "checkpoint_path": checkpoint_path,
            "step": step,
            "log_type": "checkpoint_saved",
            **kwargs
        }
        logger.bind(**extra_data).info(message)
    
    def training_completed(self, final_loss: float, training_time: float, **kwargs):
        """记录训练完成"""
        message = f"Training completed. Final loss: {final_loss:.6f}, Time: {training_time:.2f}s"
        extra_data = {
            "job_id": self.job_id,
            "final_loss": final_loss,
            "training_time": training_time,
            "log_type": "training_completed",
            **kwargs
        }
        logger.bind(**extra_data).info(message)
    
    def training_failed(self, error_message: str, **kwargs):
        """记录训练失败"""
        message = f"Training failed: {error_message}"
        extra_data = {
            "job_id": self.job_id,
            "error_message": error_message,
            "log_type": "training_failed",
            **kwargs
        }
        logger.bind(**extra_data).error(message)


class SystemLogger:
    """系统日志管理器"""
    
    def __init__(self):
        self.log_file_path = os.path.join(settings.LOG_DIR, "system.log")
        self._setup_logger()
    
    def _setup_logger(self):
        """设置系统日志记录器"""
        # 移除默认的日志处理器
        logger.remove()
        
        # 添加控制台输出
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=settings.LOG_LEVEL,
            colorize=True
        )
        
        # 添加系统日志文件
        logger.add(
            self.log_file_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=settings.LOG_LEVEL,
            rotation="100 MB",
            retention="30 days",
            compression="zip"
        )
    
    def info(self, message: str, **kwargs):
        """记录信息日志"""
        logger.bind(**kwargs).info(message)
    
    def warning(self, message: str, **kwargs):
        """记录警告日志"""
        logger.bind(**kwargs).warning(message)
    
    def error(self, message: str, **kwargs):
        """记录错误日志"""
        logger.bind(**kwargs).error(message)
    
    def debug(self, message: str, **kwargs):
        """记录调试日志"""
        logger.bind(**kwargs).debug(message)


# 创建全局系统日志实例
system_logger = SystemLogger()


def get_training_logger(job_id: str) -> TrainingLogger:
    """获取训练日志记录器"""
    return TrainingLogger(job_id)


def get_system_logger() -> SystemLogger:
    """获取系统日志记录器"""
    return system_logger 