import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置类"""

    # 环境配置
    ENVIRONMENT: str = Field(default="dev", env="ENVIRONMENT")

    # Redis配置
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")

    # 日志配置
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_DIR: str = Field(default="logs", env="LOG_DIR")

    # 应用配置
    API_PREFIX: str = Field(default="/api/v1", env="API_PREFIX")
    APP_HOST: str = Field(default="0.0.0.0", env="APP_HOST")
    APP_PORT: int = Field(default=8000, env="APP_PORT")

    class Config:
        env_file = f"env/.env.{os.getenv('ENVIRONMENT', 'dev')}"
        case_sensitive = False


# 创建全局配置实例
settings = Settings()


# 打印配置信息
def print_config_info():
    """打印配置文件和配置信息"""
    env_file_path = f"env/.env.{os.getenv('ENVIRONMENT', 'dev')}"
    print("=== 配置文件信息 ===")
    print(f"读取的配置文件: {env_file_path}")
    print(f"配置文件是否存在: {os.path.exists(env_file_path)}")
    print(f"当前环境: {settings.ENVIRONMENT}")
    print()

    print("=== 配置项详情 ===")
    print(f"ENVIRONMENT: {settings.ENVIRONMENT}")
    print(f"REDIS_HOST: {settings.REDIS_HOST}")
    print(f"REDIS_PORT: {settings.REDIS_PORT}")
    print(f"REDIS_DB: {settings.REDIS_DB}")
    print(f"REDIS_PASSWORD: {'***' if settings.REDIS_PASSWORD else 'None'}")
    print(f"LOG_LEVEL: {settings.LOG_LEVEL}")
    print(f"LOG_DIR: {settings.LOG_DIR}")
    print(f"API_PREFIX: {settings.API_PREFIX}")
    print(f"APP_HOST: {settings.APP_HOST}")
    print(f"APP_PORT: {settings.APP_PORT}")
    print("=" * 50)


# 确保必要的目录存在
os.makedirs(settings.LOG_DIR, exist_ok=True)
