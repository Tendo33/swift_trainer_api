import os
import subprocess
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from application.config import settings
from application.models.training import (
    TrainingJob,
    TrainingJobCreateRequest,
    TrainingStatus,
)
from application.services.redis_service import get_redis_service
from application.utils.gpu_utils import get_gpu_manager
from application.utils.logger import get_system_logger, get_training_logger

logger = get_system_logger()


class TrainingService:
    """训练服务类，负责执行和管理Swift训练任务"""
    
    def __init__(self):
        self.redis_service = get_redis_service()
        self.gpu_manager = get_gpu_manager()
        self.logger = logger
        self.active_processes: Dict[str, subprocess.Popen] = {}
    
    def create_training_job(self, request: TrainingJobCreateRequest) -> TrainingJob:
        """创建训练任务"""
        try:
            # 验证GPU可用性
            gpu_ids = request.gpu_id.split(',')
            for gpu_id in gpu_ids:
                if not self.gpu_manager.check_gpu_availability(gpu_id.strip()):
                    raise ValueError(f"GPU {gpu_id} 不可用")
            
            # 创建训练任务 - 使用写死的默认参数
            job = TrainingJob(
                gpu_id=request.gpu_id,
                data_path=request.data_path or settings.DEFAULT_DATASET,
                model_path=request.model_path or settings.DEFAULT_MODEL,
                output_dir=request.output_dir or settings.OUTPUT_DIR,
                # 写死的训练参数
                num_epochs=1,
                batch_size=1,
                learning_rate=1e-4,
                vit_lr=1e-5,
                aligner_lr=1e-5,
                lora_rank=16,
                lora_alpha=32,
                gradient_accumulation_steps=4,
                eval_steps=100,
                save_steps=100,
                save_total_limit=2,
                logging_steps=5,
                max_length=8192,
                warmup_ratio=0.05,
                dataloader_num_workers=4,
                dataset_num_proc=4,
                deepspeed="zero2",
                save_only_model=True,
                train_type="custom",
                external_plugins="examples/train/multimodal/lora_llm_full_vit/custom_plugin.py",
                torch_dtype="bfloat16"
            )
            
            # 保存到Redis
            self.redis_service.save_training_job(job)
            
            # 添加创建事件
            self.redis_service.add_training_event(
                job.id, "job_created", f"训练任务 {job.id} 已创建"
            )
            
            self.logger.info(f"创建训练任务 {job.id}")
            return job
            
        except Exception as e:
            self.logger.error(f"创建训练任务失败: {str(e)}")
            raise
    
    def start_training(self, job_id: str) -> bool:
        """启动训练任务"""
        try:
            # 获取训练任务
            job = self.redis_service.get_training_job(job_id)
            if job is None:
                raise ValueError(f"训练任务 {job_id} 不存在")
            
            if job.status != TrainingStatus.PENDING:
                raise ValueError(f"训练任务 {job_id} 状态不正确: {job.status}")
            
            # 检查GPU可用性
            gpu_ids = job.gpu_id.split(',')
            for gpu_id in gpu_ids:
                if not self.gpu_manager.allocate_gpu(gpu_id.strip()):
                    raise ValueError(f"GPU {gpu_id} 分配失败")
            
            # 更新任务状态
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.now()
            job.log_file_path = os.path.join(settings.LOG_DIR, f"training_{job_id}.log")
            
            # 保存更新
            self.redis_service.save_training_job(job)
            
            # 创建训练日志记录器
            training_logger = get_training_logger(job_id)
            
            # 生成训练命令
            command = self._build_training_command(job)
            
            # 设置环境变量
            env = self._build_environment(job)
            
            # 启动训练进程
            process = subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 保存进程信息
            job.process_id = process.pid
            self.redis_service.save_training_job(job)
            self.active_processes[job_id] = process
            
            # 启动监控线程
            monitor_thread = threading.Thread(
                target=self._monitor_training_process,
                args=(job_id, process, training_logger)
            )
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # 添加启动事件
            self.redis_service.add_training_event(
                job_id, "training_started", f"训练任务 {job_id} 已启动"
            )
            
            training_logger.info(f"训练任务 {job_id} 已启动，进程ID: {process.pid}")
            self.logger.info(f"启动训练任务 {job_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"启动训练任务失败: {str(e)}")
            # 更新任务状态为失败
            if job:
                job.status = TrainingStatus.FAILED
                job.error_message = str(e)
                self.redis_service.save_training_job(job)
                self.redis_service.add_training_event(
                    job_id, "training_failed", f"训练任务启动失败: {str(e)}"
                )
            return False
    
    def stop_training(self, job_id: str) -> bool:
        """停止训练任务"""
        try:
            job = self.redis_service.get_training_job(job_id)
            if job is None:
                raise ValueError(f"训练任务 {job_id} 不存在")
            
            if job.status not in [TrainingStatus.RUNNING, TrainingStatus.PENDING]:
                raise ValueError(f"训练任务 {job_id} 状态不正确: {job.status}")
            
            # 终止进程
            if job_id in self.active_processes:
                process = self.active_processes[job_id]
                process.terminate()
                
                # 等待进程结束
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                
                del self.active_processes[job_id]
            
            # 更新任务状态
            job.status = TrainingStatus.CANCELLED
            job.completed_at = datetime.now()
            self.redis_service.save_training_job(job)
            
            # 释放GPU资源
            gpu_ids = job.gpu_id.split(',')
            for gpu_id in gpu_ids:
                self.gpu_manager.release_gpu(gpu_id.strip())
            
            # 添加停止事件
            self.redis_service.add_training_event(
                job_id, "training_cancelled", f"训练任务 {job_id} 已取消"
            )
            
            self.logger.info(f"停止训练任务 {job_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"停止训练任务失败: {str(e)}")
            return False
    
    def get_training_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """获取训练状态"""
        try:
            job = self.redis_service.get_training_job(job_id)
            if job is None:
                return None
            
            # 获取训练进度
            progress_data = self.redis_service.get_training_progress(job_id)
            
            # 获取GPU使用情况
            gpu_memory_usage = {}
            gpu_ids = job.gpu_id.split(',')
            for gpu_id in gpu_ids:
                gpu_info = self.gpu_manager.get_gpu_memory_usage(gpu_id.strip())
                if gpu_info:
                    gpu_memory_usage[gpu_id.strip()] = {
                        'memory_used': gpu_info['memory_used'],
                        'memory_total': gpu_info['memory_total'],
                        'utilization': gpu_info['utilization']
                    }
            
            status_data = {
                'job_id': job_id,
                'status': job.status,
                'progress': progress_data.get('progress', 0.0) if progress_data else 0.0,
                'current_epoch': progress_data.get('current_epoch'),
                'current_step': progress_data.get('current_step'),
                'loss': progress_data.get('loss'),
                'learning_rate': progress_data.get('learning_rate'),
                'gpu_memory_usage': gpu_memory_usage,
                'estimated_time_remaining': progress_data.get('estimated_time_remaining'),
                'created_at': job.created_at,
                'started_at': job.started_at,
                'completed_at': job.completed_at
            }
            
            return status_data
            
        except Exception as e:
            self.logger.error(f"获取训练状态失败: {str(e)}")
            return None
    
    def _build_training_command(self, job: TrainingJob) -> List[str]:
        """构建训练命令 - 使用写死的参数"""
        command = [
            "swift", "sft",
            "--model", job.model_path,
            "--dataset", job.data_path,
            "--train_type", "custom",
            "--external_plugins", "examples/train/multimodal/lora_llm_full_vit/custom_plugin.py",
            "--torch_dtype", "bfloat16",
            "--num_train_epochs", "1",
            "--per_device_train_batch_size", "1",
            "--per_device_eval_batch_size", "1",
            "--learning_rate", "1e-4",
            "--vit_lr", "1e-5",
            "--aligner_lr", "1e-5",
            "--lora_rank", "16",
            "--lora_alpha", "32",
            "--gradient_accumulation_steps", "4",
            "--eval_steps", "100",
            "--save_steps", "100",
            "--save_total_limit", "2",
            "--logging_steps", "5",
            "--max_length", "8192",
            "--output_dir", job.output_dir,
            "--warmup_ratio", "0.05",
            "--dataloader_num_workers", "4",
            "--dataset_num_proc", "4",
            "--deepspeed", "zero2",
            "--save_only_model"
        ]
        
        return command
    
    def _build_environment(self, job: TrainingJob) -> Dict[str, str]:
        """构建环境变量"""
        env = os.environ.copy()
        
        # 设置GPU环境变量
        env["CUDA_VISIBLE_DEVICES"] = job.gpu_id
        
        # 设置NCCL环境变量（用于多GPU训练）
        if len(job.gpu_id.split(',')) > 1:
            env["NCCL_P2P_DISABLE"] = "1"
            env["NCCL_IB_DISABLE"] = "1"
            env["NPROC_PER_NODE"] = str(len(job.gpu_id.split(',')))
        
        return env
    
    def _monitor_training_process(self, job_id: str, process: subprocess.Popen, training_logger):
        """监控训练进程"""
        try:
            start_time = time.time()
            
            # 读取进程输出
            for line in iter(process.stdout.readline, ''):
                if line:
                    line = line.strip()
                    training_logger.info(f"训练输出: {line}")
                    
                    # 解析训练进度
                    self._parse_training_progress(job_id, line, training_logger)
                    
                    # 检查进程是否还在运行
                    if process.poll() is not None:
                        break
            
            # 等待进程结束
            return_code = process.wait()
            
            # 计算训练时间
            training_time = time.time() - start_time
            
            # 更新任务状态
            job = self.redis_service.get_training_job(job_id)
            if job:
                if return_code == 0:
                    job.status = TrainingStatus.COMPLETED
                    job.training_time = training_time
                    training_logger.training_completed(0.0, training_time)  # 这里需要从日志中提取最终损失
                else:
                    job.status = TrainingStatus.FAILED
                    job.error_message = f"训练进程返回错误代码: {return_code}"
                    training_logger.training_failed(f"进程返回错误代码: {return_code}")
                
                job.completed_at = datetime.now()
                self.redis_service.save_training_job(job)
            
            # 释放GPU资源
            if job:
                gpu_ids = job.gpu_id.split(',')
                for gpu_id in gpu_ids:
                    self.gpu_manager.release_gpu(gpu_id.strip())
            
            # 清理进程记录
            if job_id in self.active_processes:
                del self.active_processes[job_id]
            
            # 添加完成事件
            event_type = "training_completed" if return_code == 0 else "training_failed"
            event_message = f"训练任务 {job_id} {'完成' if return_code == 0 else '失败'}"
            self.redis_service.add_training_event(job_id, event_type, event_message)
            
        except Exception as e:
            self.logger.error(f"监控训练进程失败: {str(e)}")
            # 更新任务状态为失败
            job = self.redis_service.get_training_job(job_id)
            if job:
                job.status = TrainingStatus.FAILED
                job.error_message = str(e)
                self.redis_service.save_training_job(job)
    
    def _parse_training_progress(self, job_id: str, line: str, training_logger):
        """解析训练进度"""
        try:
            # 这里可以添加更复杂的进度解析逻辑
            # 目前只是简单的示例
            
            # 检查是否包含损失信息
            if "loss:" in line.lower():
                # 提取损失值
                import re
                loss_match = re.search(r'loss:\s*([\d.]+)', line.lower())
                if loss_match:
                    loss = float(loss_match.group(1))
                    # 更新进度
                    self.redis_service.save_training_progress(
                        job_id, 
                        progress=50.0,  # 这里需要更精确的计算
                        loss=loss,
                        current_step=0,  # 需要从日志中提取
                        current_epoch=0   # 需要从日志中提取
                    )
            
            # 检查是否包含检查点保存信息
            if "saving checkpoint" in line.lower():
                # 提取检查点路径
                import re
                checkpoint_match = re.search(r'checkpoint.*?(\S+)', line)
                if checkpoint_match:
                    checkpoint_path = checkpoint_match.group(1)
                    training_logger.checkpoint_saved(checkpoint_path, 0)  # 需要从日志中提取步数
            
        except Exception as e:
            self.logger.error(f"解析训练进度失败: {str(e)}")


# 创建全局训练服务实例
training_service = TrainingService()


def get_training_service() -> TrainingService:
    """获取训练服务实例"""
    return training_service 