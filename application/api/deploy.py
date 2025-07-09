from fastapi import APIRouter, HTTPException

from application.models.deploy_model import DeployJobCreateRequest
from application.services.deploy_service import DeployService

router = APIRouter(prefix="/deploy", tags=["部署任务管理"])
deploy_service = DeployService()


@router.post("/jobs", summary="创建部署任务")
def create_deploy_job(request: DeployJobCreateRequest):
    try:
        job = deploy_service.create_deploy_job(request)
        return {"job_id": job.id, "status": job.status, "port": job.port}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/start", summary="启动部署任务")
def start_deploy_job(job_id: str):
    try:
        deploy_service.start_deploy(job_id)
        return {"job_id": job_id, "message": "部署任务启动成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/stop", summary="停止部署任务")
def stop_deploy_job(job_id: str):
    try:
        deploy_service.stop_deploy(job_id)
        return {"job_id": job_id, "message": "部署任务停止成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/status", summary="查询部署任务状态")
def get_deploy_status(job_id: str):
    status = deploy_service.get_deploy_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="部署任务不存在")
    return status


@router.get("/queue", summary="查询部署队列状态")
def get_deploy_queue():
    return deploy_service.get_queue_status()


@router.post("/queue/{job_id}/add", summary="将任务加入部署队列")
def add_job_to_queue(job_id: str, port: int, priority: int):
    deploy_service.add_job_to_queue(job_id, port, priority)
    return {"message": f"任务 {job_id} 已加入部署队列"}


@router.delete("/queue/{job_id}", summary="从部署队列移除任务")
def remove_job_from_queue(job_id: str):
    success = deploy_service.remove_job_from_queue(job_id)
    if success:
        return {"message": f"任务 {job_id} 已从部署队列移除"}
    else:
        raise HTTPException(status_code=404, detail="任务不在队列中")
