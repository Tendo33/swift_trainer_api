from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from application.models.training import (
    TrainingJob,
    TrainingJobCreateRequest,
    TrainingJobListResponse,
    TrainingJobResponse,
    TrainingStatusResponse,
)
from application.services.redis_service import get_redis_service
from application.services.training_service import get_training_service
from application.utils.gpu_utils import get_gpu_manager
from application.utils.logger import get_system_logger

logger = get_system_logger()
router = APIRouter(prefix="/training", tags=["训练任务管理"])


@router.post("/jobs", response_model=TrainingJobResponse, summary="创建训练任务")
async def create_training_job(request: TrainingJobCreateRequest):
    """创建新的Swift训练任务"""
    try:
        training_service = get_training_service()
        job = training_service.create_training_job(request)
        
        logger.info(f"创建训练任务成功: {job.id}")
        
        return TrainingJobResponse(
            job_id=job.id,
            status=job.status,
            message="训练任务创建成功"
        )
    except ValueError as e:
        logger.warning(f"创建训练任务参数错误: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"创建训练任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建训练任务失败: {str(e)}")


@router.post("/jobs/{job_id}/start", response_model=TrainingJobResponse, summary="启动训练任务")
async def start_training_job(job_id: str):
    """启动指定的训练任务"""
    try:
        training_service = get_training_service()
        success = training_service.start_training(job_id)
        
        if success:
            logger.info(f"启动训练任务成功: {job_id}")
            return TrainingJobResponse(
                job_id=job_id,
                status="running",
                message="训练任务启动成功"
            )
        else:
            raise HTTPException(status_code=500, detail="启动训练任务失败")
    except ValueError as e:
        logger.warning(f"启动训练任务参数错误: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"启动训练任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"启动训练任务失败: {str(e)}")


@router.post("/jobs/{job_id}/stop", response_model=TrainingJobResponse, summary="停止训练任务")
async def stop_training_job(job_id: str):
    """停止指定的训练任务"""
    try:
        training_service = get_training_service()
        success = training_service.stop_training(job_id)
        
        if success:
            logger.info(f"停止训练任务成功: {job_id}")
            return TrainingJobResponse(
                job_id=job_id,
                status="cancelled",
                message="训练任务停止成功"
            )
        else:
            raise HTTPException(status_code=500, detail="停止训练任务失败")
    except ValueError as e:
        logger.warning(f"停止训练任务参数错误: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"停止训练任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"停止训练任务失败: {str(e)}")


@router.post("/jobs/{job_id}/export", response_model=TrainingJobResponse, summary="手动触发模型导出")
async def export_training_model(job_id: str):
    """手动触发指定训练任务的模型导出和合并"""
    try:
        training_service = get_training_service()
        success = training_service.export_model(job_id)
        
        if success:
            logger.info(f"开始导出训练模型: {job_id}")
            return TrainingJobResponse(
                job_id=job_id,
                status="exporting",
                message="模型导出已开始"
            )
        else:
            raise HTTPException(status_code=500, detail="开始模型导出失败")
    except ValueError as e:
        logger.warning(f"导出模型参数错误: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"导出模型失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"导出模型失败: {str(e)}")


@router.get("/jobs/{job_id}/status", response_model=TrainingStatusResponse, summary="获取训练状态")
async def get_training_status(job_id: str):
    """获取指定训练任务的详细状态"""
    try:
        training_service = get_training_service()
        status_data = training_service.get_training_status(job_id)
        
        if status_data is None:
            raise HTTPException(status_code=404, detail="训练任务不存在")
        
        return TrainingStatusResponse(**status_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取训练状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取训练状态失败: {str(e)}")


@router.get("/jobs", response_model=TrainingJobListResponse, summary="获取训练任务列表")
async def get_training_jobs(
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(10, ge=1, le=100, description="每页大小"),
    status: Optional[str] = Query(None, description="任务状态过滤")
):
    """获取训练任务列表，支持分页和状态过滤"""
    try:
        redis_service = get_redis_service()
        
        if status:
            # 根据状态过滤
            from application.models.training import TrainingStatus
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
            jobs=paginated_jobs,
            total=total,
            page=page,
            size=size
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取训练任务列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取训练任务列表失败: {str(e)}")


@router.get("/jobs/{job_id}", response_model=TrainingJob, summary="获取训练任务详情")
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


@router.delete("/jobs/{job_id}", response_model=TrainingJobResponse, summary="删除训练任务")
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
                job_id=job_id,
                status="deleted",
                message="训练任务删除成功"
            )
        else:
            raise HTTPException(status_code=500, detail="删除训练任务失败")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除训练任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除训练任务失败: {str(e)}")


@router.get("/jobs/{job_id}/logs", summary="获取训练日志")
async def get_training_logs(
    job_id: str,
    limit: int = Query(100, ge=1, le=1000, description="日志条数限制")
):
    """获取指定训练任务的日志"""
    try:
        redis_service = get_redis_service()
        
        # 检查任务是否存在
        job = redis_service.get_training_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="训练任务不存在")
        
        logs = redis_service.get_training_logs(job_id, limit)
        
        return {
            "job_id": job_id,
            "logs": logs,
            "total": len(logs)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取训练日志失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取训练日志失败: {str(e)}")


@router.get("/jobs/{job_id}/events", summary="获取训练事件")
async def get_training_events(
    job_id: str,
    limit: int = Query(50, ge=1, le=500, description="事件条数限制")
):
    """获取指定训练任务的事件历史"""
    try:
        redis_service = get_redis_service()
        
        # 检查任务是否存在
        job = redis_service.get_training_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="训练任务不存在")
        
        events = redis_service.get_training_events(job_id, limit)
        
        return {
            "job_id": job_id,
            "events": events,
            "total": len(events)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取训练事件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取训练事件失败: {str(e)}")


@router.get("/gpus", summary="获取GPU信息")
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
            gpu_id = str(gpu['index'])
            status_info = gpu_status.get(gpu_id, {})
            
            gpu_info = {
                "id": gpu_id,
                "name": gpu['name'],
                "memory_total": gpu['memory_total'],
                "memory_used": gpu['memory_used'],
                "memory_free": gpu['memory_free'],
                "utilization": gpu['utilization'],
                "temperature": gpu['temperature'],
                "status": status_info.get('status', 'available'),
                "job_id": status_info.get('job_id'),
                "available": gpu_manager.check_gpu_availability(gpu_id)
            }
            gpu_info_list.append(gpu_info)
        
        return {
            "gpus": gpu_info_list,
            "total": len(gpu_info_list)
        }
    except Exception as e:
        logger.error(f"获取GPU信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取GPU信息失败: {str(e)}")


@router.get("/system/status", summary="获取系统状态")
async def get_system_status():
    """获取系统资源状态"""
    try:
        gpu_manager = get_gpu_manager()
        
        # 获取系统信息
        memory_info = gpu_manager.get_system_memory_info()
        cpu_info = gpu_manager.get_system_cpu_info()
        gpu_info = gpu_manager.get_gpu_info()
        
        # 计算GPU总内存使用情况
        total_gpu_memory = 0
        used_gpu_memory = 0
        for gpu in gpu_info:
            total_gpu_memory += gpu['memory_total']
            used_gpu_memory += gpu['memory_used']
        
        return {
            "system": {
                "memory": {
                    "total": memory_info['total'],
                    "used": memory_info['used'],
                    "available": memory_info['available'],
                    "percent": memory_info['percent']
                },
                "cpu": {
                    "usage_percent": cpu_info['usage_percent'],
                    "count": cpu_info['count']
                }
            },
            "gpu": {
                "count": len(gpu_info),
                "total_memory": total_gpu_memory,
                "used_memory": used_gpu_memory,
                "memory_usage_percent": (used_gpu_memory / total_gpu_memory * 100) if total_gpu_memory > 0 else 0
            }
        }
    except Exception as e:
        logger.error(f"获取系统状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")


@router.get("/health", summary="健康检查")
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
            "timestamp": "2024-01-01T00:00:00Z",
            "services": {
                "redis": "ok" if redis_ok else "error",
                "gpu_manager": "ok" if gpu_count >= 0 else "error"
            },
            "gpu_count": gpu_count
        }
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        ) 