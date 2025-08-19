import os
from typing import Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置类"""

    # 环境配置
    ENVIRONMENT: str = Field(default="dev", env="ENVIRONMENT")

    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        allowed_environments = ["dev", "test", "staging", "prod"]
        if v not in allowed_environments:
            raise ValueError(f"Environment must be one of {allowed_environments}")
        return v

    # Redis配置
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")

    @validator("REDIS_PORT")
    def validate_redis_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("Redis port must be between 1 and 65535")
        return v

    @validator("REDIS_DB")
    def validate_redis_db(cls, v):
        if not 0 <= v <= 15:
            raise ValueError("Redis DB must be between 0 and 15")
        return v

    # 日志配置
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_DIR: str = Field(default="logs", env="LOG_DIR")

    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of {allowed_levels}")
        return v.upper()

    # 应用配置
    API_PREFIX: str = Field(default="/api/v1", env="API_PREFIX")
    APP_HOST: str = Field(default="0.0.0.0", env="APP_HOST")
    APP_PORT: int = Field(default=8000, env="APP_PORT")

    @validator("APP_PORT")
    def validate_app_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("App port must be between 1 and 65535")
        return v

    # GPU配置
    GPU_MEMORY_THRESHOLD: float = Field(default=0.3, env="GPU_MEMORY_THRESHOLD")
    GPU_MEMORY_FREE_THRESHOLD_GB: int = Field(
        default=20, env="GPU_MEMORY_FREE_THRESHOLD_GB"
    )

    @validator("GPU_MEMORY_THRESHOLD")
    def validate_gpu_memory_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("GPU memory threshold must be between 0.0 and 1.0")
        return v

    @validator("GPU_MEMORY_FREE_THRESHOLD_GB")
    def validate_gpu_memory_free_threshold(cls, v):
        if v < 0:
            raise ValueError("GPU memory free threshold must be non-negative")
        return v

    # 训练配置
    DEFAULT_TRAINING_TIMEOUT: int = Field(
        default=86400, env="DEFAULT_TRAINING_TIMEOUT"
    )  # 24小时
    QUEUE_CHECK_INTERVAL: int = Field(default=30, env="QUEUE_CHECK_INTERVAL")  # 30秒

    @validator("DEFAULT_TRAINING_TIMEOUT")
    def validate_training_timeout(cls, v):
        if v < 0:
            raise ValueError("Training timeout must be non-negative")
        return v

    @validator("QUEUE_CHECK_INTERVAL")
    def validate_queue_check_interval(cls, v):
        if v < 1:
            raise ValueError("Queue check interval must be at least 1 second")
        return v

    class Config:
        env_file = f"env/.env.{os.getenv('ENVIRONMENT', 'dev')}"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 确保日志目录存在
        os.makedirs(self.LOG_DIR, exist_ok=True)


# 创建全局配置实例
settings = Settings()


# 打印配置信息
def print_config_info():
    """打印配置文件和配置信息"""
    import logging

    logger = logging.getLogger(__name__)

    env_file_path = f"env/.env.{os.getenv('ENVIRONMENT', 'dev')}"
    logger.info("=== 配置文件信息 ===")
    logger.info(f"读取的配置文件: {env_file_path}")
    logger.info(f"配置文件是否存在: {os.path.exists(env_file_path)}")
    logger.info(f"当前环境: {settings.ENVIRONMENT}")
    logger.info("=== 配置项详情 ===")
    logger.info(f"ENVIRONMENT: {settings.ENVIRONMENT}")
    logger.info(f"REDIS_HOST: {settings.REDIS_HOST}")
    logger.info(f"REDIS_PORT: {settings.REDIS_PORT}")
    logger.info(f"REDIS_DB: {settings.REDIS_DB}")
    logger.info(f"REDIS_PASSWORD: {'***' if settings.REDIS_PASSWORD else 'None'}")
    logger.info(f"LOG_LEVEL: {settings.LOG_LEVEL}")
    logger.info(f"LOG_DIR: {settings.LOG_DIR}")
    logger.info(f"API_PREFIX: {settings.API_PREFIX}")
    logger.info(f"APP_HOST: {settings.APP_HOST}")
    logger.info(f"APP_PORT: {settings.APP_PORT}")
    logger.info("=" * 50)


# 确保必要的目录存在
os.makedirs(settings.LOG_DIR, exist_ok=True)
