import subprocess

from application.setting import settings
from application.utils.logger import get_system_logger

logger = get_system_logger()


class GPUManager:
    """GPU资源管理器"""

    def __init__(self) -> None:
        self.logger = logger
        # GPU可用性检查的配置参数 - 从配置中读取
        self.memory_usage_threshold = settings.GPU_MEMORY_THRESHOLD
        self.memory_free_threshold_gb = settings.GPU_MEMORY_FREE_THRESHOLD_GB

    def _run_nvidia_smi(self, query: str) -> str | None:
        """执行nvidia-smi命令的通用方法"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                self.logger.warning("nvidia-smi命令执行失败")
                return None

            return result.stdout.strip()

        except subprocess.TimeoutExpired:
            self.logger.error("nvidia-smi命令执行超时")
            return None
        except Exception as e:
            self.logger.error(f"nvidia-smi命令执行异常: {str(e)}")
            return None

    def get_gpu_info(self) -> list[dict]:
        """获取所有GPU信息"""
        query = "index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu"
        output = self._run_nvidia_smi(query)

        if not output:
            return []

        gpus = []
        for line in output.split("\n"):
            if line.strip():
                parts = line.split(", ")
                if len(parts) >= 7:
                    gpu_info = {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total": int(parts[2]),
                        "memory_used": int(parts[3]),
                        "memory_free": int(parts[4]),
                        "utilization": int(parts[5]),
                        "temperature": int(parts[6]),
                    }
                    gpus.append(gpu_info)

        return gpus

    def _is_gpu_available_by_info(self, gpu_info: dict) -> bool:
        """根据GPU信息判断是否可用（内部方法）"""
        memory_usage_ratio = gpu_info["memory_used"] / gpu_info["memory_total"]
        memory_free_gb = gpu_info["memory_free"] / 1024  # 转换为GB

        return (
            memory_usage_ratio < self.memory_usage_threshold
            and memory_free_gb >= self.memory_free_threshold_gb
        )

    def check_gpu_availability(self, gpu_id: str) -> bool:
        """检查指定GPU是否可用"""
        gpus = self.get_gpu_info()
        for gpu in gpus:
            if str(gpu["index"]) == gpu_id:
                return self._is_gpu_available_by_info(gpu)
        return False

    def get_available_gpus(self) -> list[str]:
        """获取所有可用的GPU ID列表"""
        gpus = self.get_gpu_info()
        return [
            str(gpu["index"]) for gpu in gpus if self._is_gpu_available_by_info(gpu)
        ]

    def get_gpu_memory_usage(self, gpu_id: str) -> dict | None:
        """获取指定GPU的内存使用情况"""
        gpus = self.get_gpu_info()
        for gpu in gpus:
            if str(gpu["index"]) == gpu_id:
                return {
                    "gpu_id": gpu_id,
                    "memory_used": gpu["memory_used"],
                    "memory_total": gpu["memory_total"],
                    "memory_free": gpu["memory_free"],
                    "utilization": gpu["utilization"],
                    "temperature": gpu["temperature"],
                }
        return None

    def get_gpu_processes(self) -> list[dict]:
        """获取GPU上运行的进程"""
        query = "compute-apps.gpu_uuid,compute-apps.pid,compute-apps.process_name,compute-apps.used_memory"
        output = self._run_nvidia_smi(query)

        if not output:
            return []

        processes = []
        for line in output.split("\n"):
            if line.strip():
                parts = line.split(", ")
                if len(parts) >= 4:
                    process_info = {
                        "gpu_uuid": parts[0],
                        "pid": int(parts[1]),
                        "process_name": parts[2],
                        "used_memory": int(parts[3]),
                    }
                    processes.append(process_info)

        return processes

    def kill_gpu_process(self, pid: int) -> bool:
        """终止指定进程"""
        try:
            # 验证PID是否有效
            if not isinstance(pid, int) or pid <= 0:
                self.logger.warning(f"无效的进程ID: {pid}")
                return False

            # 先尝试优雅终止
            try:
                subprocess.run(["kill", str(pid)], check=True, timeout=5)
                self.logger.info(f"成功终止进程 {pid}")
                return True
            except subprocess.TimeoutExpired:
                # 优雅终止失败，强制终止
                subprocess.run(["kill", "-9", str(pid)], check=True, timeout=5)
                self.logger.info(f"强制终止进程 {pid}")
                return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"终止进程 {pid} 失败: {e}")
            return False
        except Exception as e:
            self.logger.error(f"终止进程 {pid} 时发生未知错误: {e}")
            return False

    def get_gpu_count(self) -> int:
        """获取GPU数量"""
        return len(self.get_gpu_info())

    def validate_gpu_id(self, gpu_id: str) -> bool:
        """验证GPU ID是否有效"""
        if not gpu_id or not gpu_id.strip().isdigit():
            return False

        gpu_id_int = int(gpu_id)
        gpus = self.get_gpu_info()
        return any(gpu["index"] == gpu_id_int for gpu in gpus)

    def validate_gpu_ids(self, gpu_ids: str) -> bool:
        """验证GPU ID列表格式"""
        if not gpu_ids or not gpu_ids.strip():
            return False

        try:
            ids = gpu_ids.split(",")
            return all(self.validate_gpu_id(gpu_id.strip()) for gpu_id in ids)
        except (AttributeError, ValueError) as e:
            self.logger.warning(f"GPU ID格式验证失败: {gpu_ids}, 错误: {e}")
            return False


# 创建全局GPU管理器实例
gpu_manager = GPUManager()


def get_gpu_manager() -> GPUManager:
    """获取GPU管理器实例"""
    return gpu_manager


def get_gpu_count() -> int:
    """获取GPU数量"""
    return gpu_manager.get_gpu_count()


def is_gpu_available(gpu_id: str) -> bool:
    """检查GPU是否可用"""
    return gpu_manager.check_gpu_availability(gpu_id)


def get_available_gpus() -> list[str]:
    """获取所有可用的GPU ID列表"""
    return gpu_manager.get_available_gpus()


def validate_gpu_ids(gpu_ids: str) -> bool:
    """验证GPU ID格式"""
    return gpu_manager.validate_gpu_ids(gpu_ids)
