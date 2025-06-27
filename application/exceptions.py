from typing import Any, Dict, Optional


class SwiftTrainerException(Exception):
    """Swift Trainer 基础异常类"""
    
    def __init__(self, message: str, error_code: str = "INTERNAL_ERROR", details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(SwiftTrainerException):
    """参数验证错误"""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value
        super().__init__(message, "VALIDATION_ERROR", details)


class ResourceNotFoundError(SwiftTrainerException):
    """资源未找到错误"""
    
    def __init__(self, resource_type: str, resource_id: str):
        message = f"{resource_type} with id '{resource_id}' not found"
        super().__init__(message, "RESOURCE_NOT_FOUND", {"resource_type": resource_type, "resource_id": resource_id})


class ResourceUnavailableError(SwiftTrainerException):
    """资源不可用错误"""
    
    def __init__(self, resource_type: str, resource_id: str, reason: str):
        message = f"{resource_type} '{resource_id}' is unavailable: {reason}"
        super().__init__(message, "RESOURCE_UNAVAILABLE", {
            "resource_type": resource_type, 
            "resource_id": resource_id, 
            "reason": reason
        })


class TrainingError(SwiftTrainerException):
    """训练相关错误"""
    
    def __init__(self, message: str, job_id: Optional[str] = None, operation: Optional[str] = None):
        details = {}
        if job_id:
            details["job_id"] = job_id
        if operation:
            details["operation"] = operation
        super().__init__(message, "TRAINING_ERROR", details)


class ProcessError(SwiftTrainerException):
    """进程管理错误"""
    
    def __init__(self, message: str, process_id: Optional[int] = None, operation: Optional[str] = None):
        details = {}
        if process_id:
            details["process_id"] = process_id
        if operation:
            details["operation"] = operation
        super().__init__(message, "PROCESS_ERROR", details)


class ExportError(SwiftTrainerException):
    """模型导出错误"""
    
    def __init__(self, message: str, job_id: Optional[str] = None, checkpoint_path: Optional[str] = None):
        details = {}
        if job_id:
            details["job_id"] = job_id
        if checkpoint_path:
            details["checkpoint_path"] = checkpoint_path
        super().__init__(message, "EXPORT_ERROR", details) 