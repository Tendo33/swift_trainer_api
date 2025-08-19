from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from application.models.training_model import (
    DeleteAllJobsResponse,
    GPUQueueStatusResponse,
    TrainingJob,
    TrainingJobCreateRequest,
    TrainingJobListResponse,
    TrainingJobQueuedResponse,
    TrainingJobResponse,
    TrainingProgressResponse,
)
from application.services.redis_service import get_redis_service
from application.services.training_service import get_training_service
from application.utils.gpu_utils import get_gpu_manager
from application.utils.logger import get_system_logger

logger = get_system_logger()

# 创建不同的路由器，按功能分类
training_job_router = APIRouter(prefix="/training", tags=["训练任务管理"])
gpu_resource_router = APIRouter(prefix="/training", tags=["GPU资源管理"])
gpu_queue_router = APIRouter(prefix="/training", tags=["GPU队列管理"])
system_monitor_router = APIRouter(prefix="/training", tags=["系统监控"])

# ==================== 训练任务管理接口 ====================


@training_job_router.post("/jobs", summary="创建训练任务")
async def create_training_job(request: TrainingJobCreateRequest):
    """创建新的Swift训练任务"""
    try:
        training_service = get_training_service()
        job = training_service.create_training_job(request)

        # 检查任务是否在队列中
        queue_status = training_service.get_queue_status()
        job_in_queue = any(
            item["job_id"] == job.id for item in queue_status["queue_items"]
        )

        if job_in_queue:
            # 任务在队列中，返回排队响应
            queue_item = next(
                item for item in queue_status["queue_items"] if item["job_id"] == job.id
            )
            logger.info(f"创建训练任务成功并加入队列: {job.id}")

            return TrainingJobQueuedResponse(
                job_id=job.id,
                status="queued",
                message="训练任务已创建并加入GPU队列",
                queue_position=queue_item["queue_position"],
                estimated_wait_time="根据队列位置和GPU使用情况估算",
            )
        else:
            # 任务直接创建，返回正常响应
            logger.info(f"创建训练任务成功: {job.id}")

            return TrainingJobResponse(
                job_id=job.id, status=job.status, message="训练任务创建成功"
            )

    except ValueError as e:
        logger.warning(f"创建训练任务参数错误: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except (ConnectionError, TimeoutError) as e:
        logger.error(f"创建训练任务时服务连接失败: {str(e)}")
        raise HTTPException(status_code=503, detail="服务暂时不可用，请稍后重试")
    except RuntimeError as e:
        logger.error(f"创建训练任务时运行时错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"创建训练任务失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="创建训练任务失败，请检查日志")


@training_job_router.post(
    "/jobs/{job_id}/start", response_model=TrainingJobResponse, summary="启动训练任务"
)
async def start_training_job(job_id: str):
    """启动指定的训练任务"""
    try:
        training_service = get_training_service()
        success = training_service.start_training(job_id)

        if success:
            logger.info(f"启动训练任务成功: {job_id}")
            return TrainingJobResponse(
                job_id=job_id, status="running", message="训练任务启动成功"
            )
        else:
            raise HTTPException(status_code=500, detail="启动训练任务失败")
    except ValueError as e:
        logger.warning(f"启动训练任务参数错误: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"启动训练任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"启动训练任务失败: {str(e)}")


@training_job_router.post(
    "/jobs/{job_id}/stop", response_model=TrainingJobResponse, summary="停止训练任务"
)
async def stop_training_job(job_id: str):
    """停止指定的训练任务"""
    try:
        training_service = get_training_service()
        success = training_service.stop_training(job_id)

        if success:
            logger.info(f"停止训练任务成功: {job_id}")
            return TrainingJobResponse(
                job_id=job_id, status="cancelled", message="训练任务停止成功"
            )
        else:
            raise HTTPException(status_code=500, detail="停止训练任务失败")
    except ValueError as e:
        logger.warning(f"停止训练任务参数错误: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"停止训练任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"停止训练任务失败: {str(e)}")


@training_job_router.get(
    "/jobs", response_model=TrainingJobListResponse, summary="获取训练任务列表"
)
async def get_training_jobs(
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(10, ge=1, le=100, description="每页大小"),
    status: Optional[str] = Query(None, description="任务状态过滤"),
):
    """获取训练任务列表，支持分页和状态过滤"""
    try:
        redis_service = get_redis_service()

        if status:
            # 根据状态过滤
            from application.models.training_model import TrainingStatus

            try:
                status_enum = TrainingStatus(status)
                jobs = redis_service.get_jobs_by_status(status_enum)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"无效的状态值: {status}")
        else:
            # 获取所有任务
            jobs = redis_service.get_all_training_jobs()

        # 分页处理
        total = len(jobs)
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        paginated_jobs = jobs[start_idx:end_idx]

        return TrainingJobListResponse(
            jobs=paginated_jobs, total=total, page=page, size=size
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取训练任务列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取训练任务列表失败: {str(e)}")


@training_job_router.get(
    "/jobs/{job_id}", response_model=TrainingJob, summary="获取训练任务详情"
)
async def get_training_job(job_id: str):
    """获取指定训练任务的详细信息"""
    try:
        redis_service = get_redis_service()
        job = redis_service.get_training_job(job_id)

        if job is None:
            raise HTTPException(status_code=404, detail="训练任务不存在")

        return job
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取训练任务详情失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取训练任务详情失败: {str(e)}")


@training_job_router.get(
    "/jobs/{job_id}/progress",
    response_model=TrainingProgressResponse,
    summary="获取训练进度",
)
async def get_training_progress(job_id: str):
    """获取指定训练任务的进度和剩余时间"""
    try:
        training_service = get_training_service()
        status_data = training_service.get_training_status(job_id)

        if status_data is None:
            raise HTTPException(status_code=404, detail="训练任务不存在")

        # 确保进度值保留两位小数
        progress = round(status_data["progress"], 2)

        # 只返回进度和剩余时间
        response = TrainingProgressResponse(
            job_id=job_id,
            progress=progress,
            estimated_time_remaining=status_data.get("estimated_time_remaining"),
        )

        logger.info(f"获取训练进度成功: {job_id}, 进度: {progress}%")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取训练进度失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取训练进度失败: {str(e)}")


@training_job_router.get("/jobs/{job_id}/logs", summary="获取训练日志")
async def get_training_logs(
    job_id: str, limit: int = Query(100, ge=1, le=1000, description="日志条数限制")
):
    """获取指定训练任务的日志"""
    try:
        redis_service = get_redis_service()

        # 检查任务是否存在
        job = redis_service.get_training_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="训练任务不存在")

        logs = redis_service.get_training_logs(job_id, limit)

        return {"job_id": job_id, "logs": logs, "total": len(logs)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取训练日志失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取训练日志失败: {str(e)}")


@training_job_router.get("/jobs/{job_id}/events", summary="获取训练事件")
async def get_training_events(
    job_id: str, limit: int = Query(50, ge=1, le=500, description="事件条数限制")
):
    """获取指定训练任务的事件历史"""
    try:
        redis_service = get_redis_service()

        # 检查任务是否存在
        job = redis_service.get_training_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="训练任务不存在")

        events = redis_service.get_training_events(job_id, limit)

        return {"job_id": job_id, "events": events, "total": len(events)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取训练事件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取训练事件失败: {str(e)}")


@training_job_router.delete(
    "/jobs/{job_id}", response_model=TrainingJobResponse, summary="删除训练任务"
)
async def delete_training_job(job_id: str):
    """删除指定的训练任务"""
    try:
        redis_service = get_redis_service()

        # 检查任务是否存在
        job = redis_service.get_training_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="训练任务不存在")

        # 如果任务正在运行，先停止
        if job.status == "running":
            training_service = get_training_service()
            training_service.stop_training(job_id)

        # 删除任务
        success = redis_service.delete_training_job(job_id)

        if success:
            logger.info(f"删除训练任务成功: {job_id}")
            return TrainingJobResponse(
                job_id=job_id, status="deleted", message="训练任务删除成功"
            )
        else:
            raise HTTPException(status_code=500, detail="删除训练任务失败")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除训练任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除训练任务失败: {str(e)}")


@training_job_router.delete(
    "/jobs", response_model=DeleteAllJobsResponse, summary="删除所有训练任务"
)
async def delete_all_training_jobs():
    """删除所有训练任务"""
    try:
        redis_service = get_redis_service()

        # 获取所有任务，检查是否有正在运行的任务
        all_jobs = redis_service.get_all_training_jobs()
        running_jobs = [job for job in all_jobs if job.status == "running"]

        if running_jobs:
            running_job_ids = [job.id for job in running_jobs]
            raise HTTPException(
                status_code=400,
                detail=f"无法删除正在运行的任务: {', '.join(running_job_ids)}",
            )

        # 执行批量删除
        result = redis_service.delete_all_training_jobs()

        if result["success"]:
            logger.info(f"批量删除训练任务成功: {result['message']}")
            return DeleteAllJobsResponse(**result)
        else:
            raise HTTPException(status_code=500, detail=result["message"])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除所有训练任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除所有训练任务失败: {str(e)}")


@training_job_router.post(
    "/jobs/{job_id}/export",
    response_model=TrainingJobResponse,
    summary="手动触发模型导出",
)
async def export_training_model(job_id: str):
    """手动触发指定训练任务的模型导出和合并"""
    try:
        training_service = get_training_service()
        success = training_service.export_model(job_id)

        if success:
            logger.info(f"开始导出训练模型: {job_id}")
            return TrainingJobResponse(
                job_id=job_id, status="exporting", message="模型导出已开始"
            )
        else:
            raise HTTPException(status_code=500, detail="开始模型导出失败")
    except ValueError as e:
        logger.warning(f"导出模型参数错误: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"导出模型失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"导出模型失败: {str(e)}")


# ==================== GPU资源管理接口 ====================


@gpu_resource_router.get("/gpus", summary="获取GPU信息")
async def get_gpu_info():
    """获取所有GPU的信息和状态"""
    try:
        gpu_manager = get_gpu_manager()
        gpus = gpu_manager.get_gpu_info()

        # 获取GPU状态
        redis_service = get_redis_service()
        gpu_status = redis_service.get_all_gpu_status()

        # 合并信息
        gpu_info_list = []
        for gpu in gpus:
            gpu_id = str(gpu["index"])
            # 确保status_info不为None，如果为None则使用空字典
            status_info = gpu_status.get(gpu_id) or {}

            gpu_info = {
                "id": gpu_id,
                "name": gpu["name"],
                "memory_total": gpu["memory_total"],
                "memory_used": gpu["memory_used"],
                "memory_free": gpu["memory_free"],
                "utilization": gpu["utilization"],
                "temperature": gpu["temperature"],
                "status": status_info.get("status", "available"),
                "job_id": status_info.get("job_id"),
                "available": gpu_manager.check_gpu_availability(gpu_id),
            }
            gpu_info_list.append(gpu_info)

        return {"gpus": gpu_info_list, "total": len(gpu_info_list)}
    except Exception as e:
        logger.error(f"获取GPU信息失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取GPU信息失败: {str(e)}")


@gpu_resource_router.get("/system/status", summary="获取系统状态")
async def get_system_status():
    """获取系统资源状态"""
    try:
        import psutil

        gpu_manager = get_gpu_manager()

        # 获取系统信息
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        gpu_info = gpu_manager.get_gpu_info()

        # 计算GPU总内存使用情况
        total_gpu_memory = 0
        used_gpu_memory = 0
        for gpu in gpu_info:
            total_gpu_memory += gpu["memory_total"]
            used_gpu_memory += gpu["memory_used"]

        return {
            "system": {
                "memory": {
                    "total": memory.total,
                    "used": memory.used,
                    "available": memory.available,
                    "percent": memory.percent,
                },
                "cpu": {
                    "usage_percent": cpu_percent,
                    "count": cpu_count,
                },
            },
            "gpu": {
                "count": len(gpu_info),
                "total_memory": total_gpu_memory,
                "used_memory": used_gpu_memory,
                "memory_usage_percent": (used_gpu_memory / total_gpu_memory * 100)
                if total_gpu_memory > 0
                else 0,
            },
        }
    except Exception as e:
        logger.error(f"获取系统状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")


# ==================== GPU队列管理接口 ====================


@gpu_queue_router.get(
    "/queue", response_model=GPUQueueStatusResponse, summary="获取GPU队列状态"
)
async def get_gpu_queue_status():
    """获取GPU队列状态和队列中的任务列表"""
    try:
        training_service = get_training_service()
        queue_status = training_service.get_queue_status()

        return GPUQueueStatusResponse(**queue_status)
    except Exception as e:
        logger.error(f"获取GPU队列状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取GPU队列状态失败: {str(e)}")


@gpu_queue_router.post("/queue/process", summary="处理GPU队列")
async def process_gpu_queue():
    """手动触发处理GPU队列，尝试启动队列中的任务"""
    try:
        training_service = get_training_service()
        result = training_service.process_gpu_queue()

        return {
            "message": f"队列处理完成，处理了 {result['processed']} 个任务，启动了 {result['started']} 个任务",
            "result": result,
        }
    except Exception as e:
        logger.error(f"处理GPU队列失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理GPU队列失败: {str(e)}")


@gpu_queue_router.delete("/queue/{job_id}", summary="从队列中移除任务")
async def remove_job_from_queue(job_id: str):
    """从GPU队列中移除指定的任务"""
    try:
        training_service = get_training_service()
        success = training_service.remove_job_from_queue(job_id)

        if success:
            return {"message": f"任务 {job_id} 已从队列中移除"}
        else:
            raise HTTPException(status_code=500, detail="从队列移除任务失败")
    except ValueError as e:
        logger.warning(f"从队列移除任务参数错误: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"从队列移除任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"从队列移除任务失败: {str(e)}")


@gpu_queue_router.get("/queue/{job_id}/status", summary="获取任务队列状态")
async def get_job_queue_status(job_id: str):
    """获取指定任务在队列中的状态"""
    try:
        training_service = get_training_service()
        queue_status = training_service.get_queue_status()

        # 查找指定任务
        job_item = None
        for item in queue_status["queue_items"]:
            if item["job_id"] == job_id:
                job_item = item
                break

        if not job_item:
            raise HTTPException(status_code=404, detail="任务不在队列中")

        return {
            "job_id": job_id,
            "queue_position": job_item["queue_position"],
            "priority": job_item["priority"],
            "gpu_ids": job_item["gpu_ids"],
            "created_at": job_item["created_at"],
            "estimated_wait_time": "根据队列位置和GPU使用情况估算",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务队列状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取任务队列状态失败: {str(e)}")


# ==================== 系统监控接口 ====================


@system_monitor_router.get("/health", summary="健康检查")
async def health_check():
    """API健康检查"""
    try:
        # 检查Redis连接
        redis_service = get_redis_service()
        redis_ok = redis_service.ping()

        # 检查GPU管理器
        gpu_manager = get_gpu_manager()
        gpu_count = len(gpu_manager.get_gpu_info())

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "redis": "ok" if redis_ok else "error",
                "gpu_manager": "ok" if gpu_count >= 0 else "error",
            },
            "gpu_count": gpu_count,
        }
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return JSONResponse(
            status_code=503, content={"status": "unhealthy", "error": str(e)}
        )


# 创建主路由器，包含所有子路由器
router = APIRouter()
router.include_router(training_job_router)
router.include_router(gpu_resource_router)
router.include_router(gpu_queue_router)
router.include_router(system_monitor_router)
