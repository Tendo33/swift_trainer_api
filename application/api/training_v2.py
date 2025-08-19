# =============================================
# 新版训练API - 支持基类分离架构
# 创建时间：2024-12-19
# 支持LLM和VLLM专用参数和端点
# =============================================

from typing import Optional

from fastapi import APIRouter, HTTPException

from application.models.base_trainer import (
    LLMTrainingParams,
    MLLMTrainingParams,
    TrainerType,
)
from application.models.training_model import (
    TrainingJobCreateRequest,
    TrainingJobResponse,
    TrainingTaskType,
)
from application.services.training_service import get_training_service
from application.utils.logger import get_system_logger

logger = get_system_logger()

# 创建专用的路由器
llm_router = APIRouter(prefix="/training/llm", tags=["LLM训练"])
mllm_router = APIRouter(prefix="/training/mllm", tags=["MLLM训练"])
v2_router = APIRouter(prefix="/training/v2", tags=["训练V2接口"])


# ==================== LLM专用训练接口 ====================


@llm_router.post("/jobs", summary="创建LLM训练任务")
async def create_llm_training_job(
    data_path: str,
    model_path: str,
    output_dir: str,
    params: Optional[LLMTrainingParams] = None,
    priority: int = 0,
):
    """创建LLM专用训练任务"""
    try:
        request = TrainingJobCreateRequest(
            task_type=TrainingTaskType.LLM,
            data_path=data_path,
            model_path=model_path,
            output_dir=output_dir,
            priority=priority,
            llm_params=params or LLMTrainingParams(),
        )

        training_service = get_training_service()
        job = training_service.create_training_job(request)

        logger.info(f"创建LLM训练任务成功: {job.id}")

        return TrainingJobResponse(
            job_id=job.id, status=job.status, message="LLM训练任务创建成功"
        )

    except ValueError as e:
        logger.warning(f"创建LLM训练任务参数错误: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"创建LLM训练任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建LLM训练任务失败: {str(e)}")


@llm_router.get("/params/default", summary="获取LLM默认参数")
async def get_llm_default_params():
    """获取LLM训练的默认参数"""
    try:
        default_params = LLMTrainingParams()
        return {
            "trainer_type": TrainerType.LLM,
            "default_params": default_params.model_dump(),
            "description": "LLM训练默认参数配置",
        }
    except Exception as e:
        logger.error(f"获取LLM默认参数失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取默认参数失败: {str(e)}")


# ==================== MLLM专用训练接口 ====================


@mllm_router.post("/jobs", summary="创建MLLM训练任务")
async def create_mllm_training_job(
    data_path: str,
    model_path: str,
    output_dir: str,
    params: Optional[MLLMTrainingParams] = None,
    priority: int = 0,
):
    """创建MLLM专用训练任务"""
    try:
        request = TrainingJobCreateRequest(
            task_type=TrainingTaskType.MLLM,
            data_path=data_path,
            model_path=model_path,
            output_dir=output_dir,
            priority=priority,
            mllm_params=params or MLLMTrainingParams(),
        )

        training_service = get_training_service()
        job = training_service.create_training_job(request)

        logger.info(f"创建MLLM训练任务成功: {job.id}")

        return TrainingJobResponse(
            job_id=job.id, status=job.status, message="MLLM训练任务创建成功"
        )

    except ValueError as e:
        logger.warning(f"创建MLLM训练任务参数错误: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"创建MLLM训练任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建MLLM训练任务失败: {str(e)}")


@mllm_router.get("/params/default", summary="获取MLLM默认参数")
async def get_mllm_default_params():
    """获取MLLM训练的默认参数"""
    try:
        default_params = MLLMTrainingParams()
        return {
            "trainer_type": TrainerType.MLLM,
            "default_params": default_params.model_dump(),
            "description": "MLLM训练默认参数配置",
        }
    except Exception as e:
        logger.error(f"获取MLLM默认参数失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取默认参数失败: {str(e)}")


# ==================== V2通用接口 ====================


@v2_router.post("/jobs", summary="创建训练任务V2")
async def create_training_job_v2(request: TrainingJobCreateRequest):
    """
    创建训练任务V2版本
    支持新的参数格式和训练器类型
    """
    try:
        training_service = get_training_service()
        job = training_service.create_training_job(request)

        logger.info(f"创建训练任务V2成功: {job.id}, 类型: {job.trainer_type}")

        return TrainingJobResponse(
            job_id=job.id,
            status=job.status,
            message=f"训练任务创建成功，训练器类型: {job.trainer_type}",
        )

    except ValueError as e:
        logger.warning(f"创建训练任务V2参数错误: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"创建训练任务V2失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建训练任务失败: {str(e)}")


@v2_router.get("/supported-types", summary="获取支持的训练器类型")
async def get_supported_trainer_types():
    """获取支持的训练器类型和对应的参数模型"""
    try:
        return {
            "supported_types": [
                {
                    "type": TrainerType.LLM,
                    "task_types": [
                        TrainingTaskType.LLM,
                        TrainingTaskType.LANGUAGE_MODEL,
                    ],
                    "params_model": "LLMTrainingParams",
                    "description": "大语言模型训练",
                },
                {
                    "type": TrainerType.MLLM,
                    "task_types": [TrainingTaskType.MLLM, TrainingTaskType.MULTIMODAL],
                    "params_model": "MLLMTrainingParams",
                    "description": "多模态大语言模型训练",
                },
            ],
            "backward_compatibility": {
                "language_model": "自动转换为LLM类型",
                "multimodal": "自动转换为MLLM类型",
            },
        }
    except Exception as e:
        logger.error(f"获取支持的训练器类型失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取训练器类型失败: {str(e)}")


# ==================== 参数比较工具接口 ====================


@v2_router.post("/params/compare", summary="比较训练参数")
async def compare_training_params(
    llm_params: LLMTrainingParams, mllm_params: MLLMTrainingParams
):
    """比较LLM和MLLM训练参数的差异"""
    try:
        llm_dict = llm_params.model_dump()
        mllm_dict = mllm_params.model_dump()

        # 找出通用参数
        common_params = {}
        llm_specific = {}
        mllm_specific = {}

        all_keys = set(llm_dict.keys()) | set(mllm_dict.keys())

        for key in all_keys:
            if key in llm_dict and key in mllm_dict:
                common_params[key] = {
                    "llm_value": llm_dict[key],
                    "mllm_value": mllm_dict[key],
                    "same": llm_dict[key] == mllm_dict[key],
                }
            elif key in llm_dict:
                llm_specific[key] = llm_dict[key]
            else:
                mllm_specific[key] = mllm_dict[key]

        return {
            "common_params": common_params,
            "llm_specific_params": llm_specific,
            "mllm_specific_params": mllm_specific,
            "summary": {
                "total_common": len(common_params),
                "llm_specific_count": len(llm_specific),
                "mllm_specific_count": len(mllm_specific),
            },
        }

    except Exception as e:
        logger.error(f"比较训练参数失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"参数比较失败: {str(e)}")


# 创建主路由器，包含所有子路由器
router = APIRouter()
router.include_router(llm_router)
router.include_router(mllm_router)
router.include_router(v2_router)
