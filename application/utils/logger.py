import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from application.config import settings


class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        # 基础日志信息
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # 添加额外字段
        if hasattr(record, 'job_id'):
            log_data["job_id"] = record.job_id
        if hasattr(record, 'gpu_id'):
            log_data["gpu_id"] = record.gpu_id
        if hasattr(record, 'operation'):
            log_data["operation"] = record.operation
        if hasattr(record, 'duration'):
            log_data["duration"] = record.duration
        
        # 添加自定义字段
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data, ensure_ascii=False)


class TrainingLoggerAdapter(logging.LoggerAdapter):
    """训练任务日志适配器"""
    
    def __init__(self, logger: logging.Logger, job_id: str):
        super().__init__(logger, {"job_id": job_id})
        self.job_id = job_id
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """处理日志消息，添加任务ID"""
        extra = kwargs.get('extra', {})
        extra['job_id'] = self.job_id
        kwargs['extra'] = extra
        return msg, kwargs
    
    def log_with_context(self, level: int, message: str, **kwargs):
        """带上下文的日志记录"""
        extra = kwargs.get('extra', {})
        extra['job_id'] = self.job_id
        extra.update({k: v for k, v in kwargs.items() if k != 'extra'})
        
        self.log(level, message, extra=extra)


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 创建格式化器
    formatter = StructuredFormatter()
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件）
    if log_file:
        # 确保日志目录存在
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用RotatingFileHandler进行日志轮转
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_system_logger() -> logging.Logger:
    """获取系统日志记录器"""
    log_file = os.path.join(settings.LOG_DIR, "system.log")
    return setup_logger("swift_trainer.system", log_file, settings.LOG_LEVEL)


def get_training_logger(job_id: str) -> TrainingLoggerAdapter:
    """获取训练任务日志记录器"""
    log_file = os.path.join(settings.LOG_DIR, f"training_{job_id}.log")
    logger = setup_logger(f"swift_trainer.training.{job_id}", log_file, settings.LOG_LEVEL)
    return TrainingLoggerAdapter(logger, job_id)


def get_api_logger() -> logging.Logger:
    """获取API日志记录器"""
    log_file = os.path.join(settings.LOG_DIR, "api.log")
    return setup_logger("swift_trainer.api", log_file, settings.LOG_LEVEL)


def get_resource_logger() -> logging.Logger:
    """获取资源管理日志记录器"""
    log_file = os.path.join(settings.LOG_DIR, "resource.log")
    return setup_logger("swift_trainer.resource", log_file, settings.LOG_LEVEL)


class LogContext:
    """日志上下文管理器"""
    
    def __init__(self, logger: logging.Logger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(
            f"开始执行操作: {self.operation}",
            extra={
                'operation': self.operation,
                'extra_fields': self.context
            }
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(
                f"操作完成: {self.operation}",
                extra={
                    'operation': self.operation,
                    'duration': duration,
                    'extra_fields': self.context
                }
            )
        else:
            self.logger.error(
                f"操作失败: {self.operation} - {str(exc_val)}",
                extra={
                    'operation': self.operation,
                    'duration': duration,
                    'error_type': exc_type.__name__,
                    'error_message': str(exc_val),
                    'extra_fields': self.context
                },
                exc_info=True
            )


def log_operation(operation: str, **context):
    """操作日志装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_system_logger()
            with LogContext(logger, operation, **context):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# 创建默认日志记录器
system_logger = get_system_logger()
api_logger = get_api_logger()
resource_logger = get_resource_logger() 