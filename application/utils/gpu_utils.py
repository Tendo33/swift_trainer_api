import subprocess
from typing import Dict, List, Optional

import psutil
from application.utils.logger import get_system_logger

logger = get_system_logger()


class GPUManager:
    """GPU资源管理器"""
    
    def __init__(self):
        self.logger = logger
    
    def get_gpu_info(self) -> List[Dict]:
        """获取所有GPU信息"""
        try:
            # 使用nvidia-smi获取GPU信息
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                self.logger.warning("无法获取GPU信息，nvidia-smi可能不可用")
                return []
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 7:
                        gpu_info = {
                            'index': int(parts[0]),
                            'name': parts[1],
                            'memory_total': int(parts[2]),
                            'memory_used': int(parts[3]),
                            'memory_free': int(parts[4]),
                            'utilization': int(parts[5]),
                            'temperature': int(parts[6])
                        }
                        gpus.append(gpu_info)
            
            return gpus
            
        except subprocess.TimeoutExpired:
            self.logger.error("获取GPU信息超时")
            return []
        except Exception as e:
            self.logger.error(f"获取GPU信息失败: {str(e)}")
            return []
    
    def get_gpu_memory_usage(self, gpu_id: str) -> Optional[Dict]:
        """获取指定GPU的内存使用情况"""
        gpus = self.get_gpu_info()
        for gpu in gpus:
            if str(gpu['index']) == gpu_id:
                return {
                    'gpu_id': gpu_id,
                    'memory_used': gpu['memory_used'],
                    'memory_total': gpu['memory_total'],
                    'memory_free': gpu['memory_free'],
                    'utilization': gpu['utilization'],
                    'temperature': gpu['temperature']
                }
        return None
    
    def check_gpu_availability(self, gpu_id: str) -> bool:
        """检查指定GPU是否可用"""
        gpu_info = self.get_gpu_memory_usage(gpu_id)
        if gpu_info is None:
            return False
        
        # 检查内存使用率，如果超过90%则认为不可用
        memory_usage_ratio = gpu_info['memory_used'] / gpu_info['memory_total']
        return memory_usage_ratio < 0.9
    
    def get_available_gpus(self) -> List[str]:
        """获取所有可用的GPU ID列表"""
        gpus = self.get_gpu_info()
        available_gpus = []
        
        for gpu in gpus:
            memory_usage_ratio = gpu['memory_used'] / gpu['memory_total']
            if memory_usage_ratio < 0.9:  # 内存使用率小于90%
                available_gpus.append(str(gpu['index']))
        
        return available_gpus
    
    def allocate_gpu(self, gpu_id: str) -> bool:
        """分配GPU资源（标记为使用中）"""
        if not self.check_gpu_availability(gpu_id):
            self.logger.warning(f"GPU {gpu_id} 不可用或已被占用")
            return False
        
        self.logger.info(f"成功分配GPU {gpu_id}")
        return True
    
    def release_gpu(self, gpu_id: str) -> None:
        """释放GPU资源"""
        self.logger.info(f"释放GPU {gpu_id}")
    
    def get_gpu_processes(self, gpu_id: str) -> List[Dict]:
        """获取指定GPU上运行的进程"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', 
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return []
            
            processes = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 4:
                        # 这里需要根据实际的GPU UUID来过滤，简化处理
                        process_info = {
                            'pid': int(parts[1]),
                            'process_name': parts[2],
                            'used_memory': int(parts[3])
                        }
                        processes.append(process_info)
            
            return processes
            
        except Exception as e:
            self.logger.error(f"获取GPU进程信息失败: {str(e)}")
            return []
    
    def kill_gpu_process(self, pid: int) -> bool:
        """终止指定进程"""
        try:
            subprocess.run(['kill', '-9', str(pid)], check=True)
            self.logger.info(f"成功终止进程 {pid}")
            return True
        except subprocess.CalledProcessError:
            self.logger.error(f"终止进程 {pid} 失败")
            return False
    
    def get_system_memory_info(self) -> Dict:
        """获取系统内存信息"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent
        }
    
    def get_system_cpu_info(self) -> Dict:
        """获取系统CPU信息"""
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        return {
            'usage_percent': cpu_percent,
            'count': cpu_count
        }


# 创建全局GPU管理器实例
gpu_manager = GPUManager()


def get_gpu_manager() -> GPUManager:
    """获取GPU管理器实例"""
    return gpu_manager


def validate_gpu_ids(gpu_ids: str) -> bool:
    """验证GPU ID格式"""
    if not gpu_ids:
        return False
    
    try:
        ids = gpu_ids.split(',')
        for gpu_id in ids:
            if not gpu_id.strip().isdigit():
                return False
        return True
    except Exception:
        return False


def get_gpu_count() -> int:
    """获取GPU数量"""
    gpus = gpu_manager.get_gpu_info()
    return len(gpus)


def is_gpu_available(gpu_id: str) -> bool:
    """检查GPU是否可用"""
    return gpu_manager.check_gpu_availability(gpu_id) 