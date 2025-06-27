import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from application.api.training import router as training_router
from application.config import settings
from application.exceptions import SwiftTrainerException
from application.utils.logger import LogContext, get_system_logger
from application.utils.response import (
    error_response,
    exception_response,
    validation_error_response,
)

logger = get_system_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    with LogContext(logger, "应用启动", app_name="Swift Trainer API"):
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
        
        # 清理过期的资源锁
        from application.utils.resource_manager import get_resource_manager
        resource_manager = get_resource_manager()
        cleaned_count = resource_manager.cleanup_expired_locks()
        if cleaned_count > 0:
            logger.info(f"清理了 {cleaned_count} 个过期的资源锁")
    
    yield
    
    # 关闭时执行
    with LogContext(logger, "应用关闭", app_name="Swift Trainer API"):
        # 清理资源
        from application.utils.resource_manager import get_resource_manager
        resource_manager = get_resource_manager()
        # 这里可以添加其他清理逻辑


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
    logger.error(f"HTTP异常: {exc.status_code} - {exc.detail}", extra={
        'status_code': exc.status_code,
        'path': request.url.path,
        'method': request.method
    })
    return error_response(
        message=exc.detail,
        error_code="HTTP_ERROR",
        error_details={"status_code": exc.status_code, "path": request.url.path},
        status_code=exc.status_code
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """请求验证异常处理"""
    logger.error(f"请求验证失败: {exc.errors()}", extra={
        'path': request.url.path,
        'method': request.method,
        'validation_errors': exc.errors()
    })
    return validation_error_response(exc.errors())


@app.exception_handler(SwiftTrainerException)
async def swift_trainer_exception_handler(request: Request, exc: SwiftTrainerException):
    """Swift Trainer 自定义异常处理"""
    logger.error(f"Swift Trainer 异常: {exc.message}", extra={
        'error_code': exc.error_code,
        'error_details': exc.details,
        'path': request.url.path,
        'method': request.method
    })
    return exception_response(exc)


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理"""
    logger.error(f"未处理的异常: {str(exc)}", extra={
        'exception_type': type(exc).__name__,
        'path': request.url.path,
        'method': request.method
    }, exc_info=True)
    return error_response(
        message="服务器内部错误",
        error_code="INTERNAL_ERROR",
        error_details={"exception_type": type(exc).__name__},
        status_code=500
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
        reload=True if settings.DEBUG else False,
        log_level=settings.LOG_LEVEL.lower()
    ) 