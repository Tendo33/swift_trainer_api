import os
from contextlib import asynccontextmanager

from application.api.training import router as training_router
from application.config import settings
from application.utils.logger import get_system_logger
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = get_system_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("Swift Trainer API 正在启动...")
    
    # 检查必要的目录
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    
    # 检查Redis连接
    from application.services.redis_service import get_redis_service
    redis_service = get_redis_service()
    if not redis_service.ping():
        logger.warning("Redis连接失败，某些功能可能不可用")
    else:
        logger.info("Redis连接成功")
    
    # 检查GPU管理器
    from application.utils.gpu_utils import get_gpu_manager
    gpu_manager = get_gpu_manager()
    gpu_count = len(gpu_manager.get_gpu_info())
    logger.info(f"检测到 {gpu_count} 个GPU")
    
    yield
    
    # 关闭时执行
    logger.info("Swift Trainer API 正在关闭...")


# 创建FastAPI应用
app = FastAPI(
    title="Swift Trainer API",
    description="基于FastAPI的Swift训练任务管理API系统",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 全局异常处理
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """HTTP异常处理"""
    logger.error(f"HTTP异常: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": request.url.path
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """请求验证异常处理"""
    logger.error(f"请求验证失败: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "请求参数验证失败",
            "details": exc.errors(),
            "path": request.url.path
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理"""
    logger.error(f"未处理的异常: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "服务器内部错误",
            "message": str(exc),
            "path": request.url.path
        }
    )


# 注册路由
app.include_router(training_router, prefix=settings.API_PREFIX)


@app.get("/", summary="根路径")
async def root():
    """API根路径"""
    return {
        "message": "Swift Trainer API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": f"{settings.API_PREFIX}/training/health"
    }


@app.get("/info", summary="API信息")
async def api_info():
    """获取API详细信息"""
    return {
        "name": "Swift Trainer API",
        "version": "1.0.0",
        "description": "基于FastAPI的Swift训练任务管理API系统",
        "features": [
            "Swift训练任务管理",
            "Redis状态存储",
            "GPU资源管理",
            "详细日志记录",
            "实时进度监控"
        ],
        "endpoints": {
            "training": f"{settings.API_PREFIX}/training",
            "health": f"{settings.API_PREFIX}/training/health",
            "gpu_info": f"{settings.API_PREFIX}/training/gpus",
            "system_status": f"{settings.API_PREFIX}/training/system/status"
        },
        "config": {
            "redis_host": settings.REDIS_HOST,
            "redis_port": settings.REDIS_PORT,
            "log_level": settings.LOG_LEVEL,
            "output_dir": settings.OUTPUT_DIR
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("启动Swift Trainer API服务器...")
    logger.info(f"主机: {settings.APP_HOST}")
    logger.info(f"端口: {settings.APP_PORT}")
    logger.info(f"API前缀: {settings.API_PREFIX}")
    
    uvicorn.run(
        "application.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    ) 