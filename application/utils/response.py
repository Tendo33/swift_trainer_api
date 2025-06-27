from typing import Any, Dict, List, Optional

from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from application.exceptions import SwiftTrainerException


class APIResponse(BaseModel):
    """统一API响应格式"""
    success: bool = Field(..., description="请求是否成功")
    message: str = Field(..., description="响应消息")
    data: Optional[Any] = Field(default=None, description="响应数据")
    error_code: Optional[str] = Field(default=None, description="错误代码")
    error_details: Optional[Dict[str, Any]] = Field(default=None, description="错误详情")
    timestamp: float = Field(default_factory=lambda: __import__('time').time(), description="响应时间戳")


class PaginatedResponse(BaseModel):
    """分页响应格式"""
    items: List[Any] = Field(..., description="数据项列表")
    total: int = Field(..., description="总数量")
    page: int = Field(..., description="当前页码")
    size: int = Field(..., description="每页大小")
    pages: int = Field(..., description="总页数")
    has_next: bool = Field(..., description="是否有下一页")
    has_prev: bool = Field(..., description="是否有上一页")


def success_response(
    data: Any = None, 
    message: str = "操作成功", 
    status_code: int = 200
) -> JSONResponse:
    """成功响应"""
    response = APIResponse(
        success=True,
        message=message,
        data=data
    )
    return JSONResponse(
        content=response.model_dump(),
        status_code=status_code
    )


def error_response(
    message: str,
    error_code: str = "INTERNAL_ERROR",
    error_details: Optional[Dict[str, Any]] = None,
    status_code: int = 500
) -> JSONResponse:
    """错误响应"""
    response = APIResponse(
        success=False,
        message=message,
        error_code=error_code,
        error_details=error_details
    )
    return JSONResponse(
        content=response.model_dump(),
        status_code=status_code
    )


def paginated_response(
    items: List[Any],
    total: int,
    page: int,
    size: int,
    message: str = "获取数据成功"
) -> JSONResponse:
    """分页响应"""
    pages = (total + size - 1) // size
    paginated_data = PaginatedResponse(
        items=items,
        total=total,
        page=page,
        size=size,
        pages=pages,
        has_next=page < pages,
        has_prev=page > 1
    )
    
    return success_response(
        data=paginated_data.model_dump(),
        message=message
    )


def exception_response(exception: SwiftTrainerException, status_code: int = 500) -> JSONResponse:
    """异常响应"""
    return error_response(
        message=exception.message,
        error_code=exception.error_code,
        error_details=exception.details,
        status_code=status_code
    )


def validation_error_response(errors: List[Dict[str, Any]]) -> JSONResponse:
    """参数验证错误响应"""
    return error_response(
        message="参数验证失败",
        error_code="VALIDATION_ERROR",
        error_details={"validation_errors": errors},
        status_code=422
    )


def not_found_response(resource_type: str, resource_id: str) -> JSONResponse:
    """资源未找到响应"""
    return error_response(
        message=f"{resource_type} with id '{resource_id}' not found",
        error_code="RESOURCE_NOT_FOUND",
        error_details={"resource_type": resource_type, "resource_id": resource_id},
        status_code=404
    )


def resource_unavailable_response(resource_type: str, resource_id: str, reason: str) -> JSONResponse:
    """资源不可用响应"""
    return error_response(
        message=f"{resource_type} '{resource_id}' is unavailable: {reason}",
        error_code="RESOURCE_UNAVAILABLE",
        error_details={"resource_type": resource_type, "resource_id": resource_id, "reason": reason},
        status_code=409
    ) 