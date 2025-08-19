import inspect
import os
import sys
from datetime import datetime

from loguru import logger

from application.setting import settings


class TrainingLogger:
    """训练日志管理器"""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.log_file_path = os.path.join(settings.LOG_DIR, f"training_{job_id}.log")
        self.json_log_file_path = os.path.join(
            settings.LOG_DIR, f"training_{job_id}.jsonl"
        )

        # 配置日志记录器
        self._setup_logger()

    def _setup_logger(self):
        """设置日志记录器"""
        # 移除默认的日志处理器
        logger.remove()

        # 添加控制台输出
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <5}</level> | <cyan>{extra[caller_file]}</cyan>:<cyan>{extra[caller_function]}</cyan>:<cyan>{extra[caller_line]}</cyan> - <level>{message}</level>",
            level=settings.LOG_LEVEL,
            colorize=True,
        )

        # 添加文件输出
        logger.add(
            self.log_file_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <5} | {extra[caller_file]}:{extra[caller_function]}:{extra[caller_line]} - {message}",
            level=settings.LOG_LEVEL,
            rotation="100 MB",
            retention="30 days",
            compression="zip",
        )

        # 添加JSON格式日志（用于结构化日志分析）
        logger.add(
            self.json_log_file_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level=settings.LOG_LEVEL,
            serialize=True,
            rotation="100 MB",
            retention="30 days",
        )

    def _get_caller_info(self):
        """获取调用者信息"""
        # 获取调用栈，跳过当前方法
        frame = inspect.currentframe()
        try:
            # 跳过当前方法(_get_caller_info)和日志方法(info/warning/error/debug)
            caller_frame = frame.f_back.f_back
            if caller_frame:
                filename = os.path.basename(caller_frame.f_code.co_filename)
                function = caller_frame.f_code.co_name
                line = caller_frame.f_lineno
                return filename, function, line
        except Exception:
            pass
        finally:
            del frame
        return "unknown", "unknown", 0

    def _save_to_redis(self, level: str, message: str, **kwargs):
        """保存日志到Redis"""
        try:
            # 在方法内部导入Redis服务，避免循环导入
            from application.services.redis_service import get_redis_service

            redis_service = get_redis_service()

            filename, function, line = self._get_caller_info()
            log_entry = {
                "level": level,
                "message": message,
                "job_id": self.job_id,
                "caller_file": filename,
                "caller_function": function,
                "caller_line": line,
                "timestamp": datetime.now().isoformat(),
                **kwargs,
            }
            redis_service.save_training_log(self.job_id, log_entry)
        except (ImportError, ConnectionError) as e:
            # 如果Redis连接失败，不影响文件日志记录
            logger.warning(f"Redis连接失败，跳过日志保存: {e}")
        except Exception as e:
            # 其他异常也记录但不影响主流程
            logger.error(f"保存日志到Redis失败: {str(e)}")

    def info(self, message: str, **kwargs):
        """记录信息日志"""
        filename, function, line = self._get_caller_info()
        extra_data = {
            "job_id": self.job_id,
            "caller_file": filename,
            "caller_function": function,
            "caller_line": line,
            **kwargs,
        }
        logger.bind(**extra_data).info(message)

        # 同时保存到Redis
        self._save_to_redis("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs):
        """记录警告日志"""
        filename, function, line = self._get_caller_info()
        extra_data = {
            "job_id": self.job_id,
            "caller_file": filename,
            "caller_function": function,
            "caller_line": line,
            **kwargs,
        }
        logger.bind(**extra_data).warning(message)

        # 同时保存到Redis
        self._save_to_redis("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs):
        """记录错误日志"""
        filename, function, line = self._get_caller_info()
        extra_data = {
            "job_id": self.job_id,
            "caller_file": filename,
            "caller_function": function,
            "caller_line": line,
            **kwargs,
        }
        logger.bind(**extra_data).error(message)

        # 同时保存到Redis
        self._save_to_redis("ERROR", message, **kwargs)

    def debug(self, message: str, **kwargs):
        """记录调试日志"""
        filename, function, line = self._get_caller_info()
        extra_data = {
            "job_id": self.job_id,
            "caller_file": filename,
            "caller_function": function,
            "caller_line": line,
            **kwargs,
        }
        logger.bind(**extra_data).debug(message)

        # 同时保存到Redis
        self._save_to_redis("DEBUG", message, **kwargs)

    def training_progress(
        self, epoch: int, step: int, loss: float, learning_rate: float, **kwargs
    ):
        """记录训练进度"""
        message = (
            f"Epoch {epoch}, Step {step}, Loss: {loss:.6f}, LR: {learning_rate:.2e}"
        )
        filename, function, line = self._get_caller_info()
        extra_data = {
            "job_id": self.job_id,
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "learning_rate": learning_rate,
            "log_type": "training_progress",
            "caller_file": filename,
            "caller_function": function,
            "caller_line": line,
            **kwargs,
        }
        logger.bind(**extra_data).info(message)

        # 同时保存到Redis
        self._save_to_redis("INFO", message, **kwargs)

    def gpu_usage(self, gpu_id: str, memory_used: float, memory_total: float, **kwargs):
        """记录GPU使用情况"""
        memory_percent = (memory_used / memory_total) * 100
        message = f"GPU {gpu_id}: {memory_used:.1f}MB / {memory_total:.1f}MB ({memory_percent:.1f}%)"
        filename, function, line = self._get_caller_info()
        extra_data = {
            "job_id": self.job_id,
            "gpu_id": gpu_id,
            "memory_used": memory_used,
            "memory_total": memory_total,
            "memory_percent": memory_percent,
            "log_type": "gpu_usage",
            "caller_file": filename,
            "caller_function": function,
            "caller_line": line,
            **kwargs,
        }
        logger.bind(**extra_data).info(message)

        # 同时保存到Redis
        self._save_to_redis("INFO", message, **kwargs)

    def checkpoint_saved(self, checkpoint_path: str, step: int, **kwargs):
        """记录检查点保存"""
        message = f"Checkpoint saved at step {step}: {checkpoint_path}"
        filename, function, line = self._get_caller_info()
        extra_data = {
            "job_id": self.job_id,
            "checkpoint_path": checkpoint_path,
            "step": step,
            "log_type": "checkpoint_saved",
            "caller_file": filename,
            "caller_function": function,
            "caller_line": line,
            **kwargs,
        }
        logger.bind(**extra_data).info(message)

        # 同时保存到Redis
        self._save_to_redis("INFO", message, **kwargs)

    def training_completed(self, final_loss: float, training_time: float, **kwargs):
        """记录训练完成"""
        message = f"Training completed. Final loss: {final_loss:.6f}, Time: {training_time:.2f}s"
        filename, function, line = self._get_caller_info()
        extra_data = {
            "job_id": self.job_id,
            "final_loss": final_loss,
            "training_time": training_time,
            "log_type": "training_completed",
            "caller_file": filename,
            "caller_function": function,
            "caller_line": line,
            **kwargs,
        }
        logger.bind(**extra_data).info(message)

        # 同时保存到Redis
        self._save_to_redis("INFO", message, **kwargs)

    def training_failed(self, error_message: str, **kwargs):
        """记录训练失败"""
        message = f"Training failed: {error_message}"
        filename, function, line = self._get_caller_info()
        extra_data = {
            "job_id": self.job_id,
            "error_message": error_message,
            "log_type": "training_failed",
            "caller_file": filename,
            "caller_function": function,
            "caller_line": line,
            **kwargs,
        }
        logger.bind(**extra_data).error(message)

        # 同时保存到Redis
        self._save_to_redis("ERROR", message, **kwargs)


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
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <5}</level> | <cyan>{extra[caller_file]}</cyan>:<cyan>{extra[caller_function]}</cyan>:<cyan>{extra[caller_line]}</cyan> - <level>{message}</level>",
            level=settings.LOG_LEVEL,
            colorize=True,
        )

        # 添加系统日志文件
        logger.add(
            self.log_file_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <5} | {extra[caller_file]}:{extra[caller_function]}:{extra[caller_line]} - {message}",
            level=settings.LOG_LEVEL,
            rotation="100 MB",
            retention="30 days",
            compression="zip",
        )

    def _get_caller_info(self):
        """获取调用者信息"""
        # 获取调用栈，跳过当前方法
        frame = inspect.currentframe()
        try:
            # 跳过当前方法(_get_caller_info)和日志方法(info/warning/error/debug)
            caller_frame = frame.f_back.f_back
            if caller_frame:
                filename = os.path.basename(caller_frame.f_code.co_filename)
                function = caller_frame.f_code.co_name
                line = caller_frame.f_lineno
                return filename, function, line
        except Exception:
            pass
        finally:
            del frame
        return "unknown", "unknown", 0

    def info(self, message: str, **kwargs):
        """记录信息日志"""
        filename, function, line = self._get_caller_info()
        extra_data = {
            "caller_file": filename,
            "caller_function": function,
            "caller_line": line,
            **kwargs,
        }
        logger.bind(**extra_data).info(message)

    def warning(self, message: str, **kwargs):
        """记录警告日志"""
        filename, function, line = self._get_caller_info()
        extra_data = {
            "caller_file": filename,
            "caller_function": function,
            "caller_line": line,
            **kwargs,
        }
        logger.bind(**extra_data).warning(message)

    def error(self, message: str, **kwargs):
        """记录错误日志"""
        filename, function, line = self._get_caller_info()
        extra_data = {
            "caller_file": filename,
            "caller_function": function,
            "caller_line": line,
            **kwargs,
        }
        logger.bind(**extra_data).error(message)

    def debug(self, message: str, **kwargs):
        """记录调试日志"""
        filename, function, line = self._get_caller_info()
        extra_data = {
            "caller_file": filename,
            "caller_function": function,
            "caller_line": line,
            **kwargs,
        }
        logger.bind(**extra_data).debug(message)


# 创建全局系统日志实例
system_logger = SystemLogger()


def get_training_logger(job_id: str) -> TrainingLogger:
    """获取训练日志记录器"""
    return TrainingLogger(job_id)


def get_system_logger() -> SystemLogger:
    """获取系统日志记录器"""
    return system_logger
