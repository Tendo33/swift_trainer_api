#!/usr/bin/env python3
"""
Swift Trainer API 启动脚本
"""

import os
import sys
from pathlib import Path

import uvicorn
from application.config import print_config_info, settings
from application.utils.logger import get_system_logger

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))



logger = get_system_logger()


def main():
    """主函数"""
    logger.info("=" * 50)
    logger.info("Swift Trainer API 启动中...")
    logger.info("=" * 50)
    
    # 检查必要的目录
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    
    logger.info(f"日志目录: {settings.LOG_DIR}")
    print_config_info()
    # 启动服务器
    uvicorn.run(
        "application.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=False,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )


if __name__ == "__main__":
    main() 