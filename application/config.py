import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置类"""
    
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
    DEBUG: bool = Field(default=True, env="DEBUG")
    
    # 训练配置
    OUTPUT_DIR: str = Field(default="output", env="OUTPUT_DIR")
    DATA_DIR: str = Field(default="data", env="DATA_DIR")
    
    # Swift训练默认参数 - 只保留模型和数据集配置
    DEFAULT_MODEL: str = Field(default="Qwen/Qwen2.5-VL-7B-Instruct", env="DEFAULT_MODEL")
    DEFAULT_DATASET: str = Field(default="AI-ModelScope/coco#20000", env="DEFAULT_DATASET")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# 创建全局配置实例
settings = Settings()

# 确保必要的目录存在
os.makedirs(settings.LOG_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.DATA_DIR, exist_ok=True) 