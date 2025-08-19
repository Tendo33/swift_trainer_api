# API Package

# 导出路由器
from .training import router as training_router
from .training_v2 import router as training_v2_router

__all__ = [
    "training_router",
    "training_v2_router",
]
