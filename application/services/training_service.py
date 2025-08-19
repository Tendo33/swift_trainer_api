import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from application.config import settings
from application.models.training_model import (
    TrainingJob,
    TrainingJobCreateRequest,
    TrainingStatus,
)
from application.services.redis_service import get_redis_service
from application.utils.gpu_utils import get_gpu_manager
from application.utils.logger import get_system_logger, get_training_logger

from .training_handler import TrainingHandler

logger = get_system_logger()


class TrainingService:
    """训练服务类，负责执行和管理Swift训练任务"""

    def __init__(self):
        self.redis_service = get_redis_service()
        self.gpu_manager = get_gpu_manager()
        self.logger = logger
        self.active_processes: Dict[str, subprocess.Popen] = {}
        self.queue_processor_running = False
        self.queue_processor_thread = None
        self.training_handler = TrainingHandler()

    def create_training_job(self, request: TrainingJobCreateRequest) -> TrainingJob:
        """创建训练任务，支持多任务类型"""
        try:
            # 根据配置自动分配GPU
            gpu_count = self.training_handler.get_gpu_count_for_task(request)
            gpu_id_list = self._auto_allocate_gpus(gpu_count)

            if request.task_type == "multimodal":
                job_kwargs = self.training_handler.handle_multimodal(
                    request, gpu_id_list
                )
                job = TrainingJob(**job_kwargs)
            elif request.task_type == "language_model":
                job_kwargs = self.training_handler.handle_language_model(
                    request, gpu_id_list
                )
                job = TrainingJob(**job_kwargs)
            elif request.task_type == "deploy":
                job_kwargs = self.training_handler.handle_deploy(
                    request, gpu_id_list
                )
                job = TrainingJob(**job_kwargs)
            else:
                raise ValueError(f"不支持的任务类型: {request.task_type}")

            # 检查GPU可用性
            gpu_available = True
            unavailable_gpus = []
            for gpu_id in gpu_id_list:
                if not self.gpu_manager.check_gpu_availability(gpu_id):
                    gpu_available = False
                    unavailable_gpus.append(gpu_id)

            # 如果GPU不可用，将任务添加到队列
            if not gpu_available:
                # 将任务添加到GPU队列
                success = self.redis_service.add_job_to_gpu_queue(
                    job.id, gpu_id_list, request.priority
                )

                if success:
                    # 保存到Redis
                    self.redis_service.save_training_job(job)

                    # 添加创建事件
                    self.redis_service.add_training_event(
                        job.id, "job_created", f"训练任务 {job.id} 已创建并加入GPU队列"
                    )

                    # 添加排队事件
                    self.redis_service.add_training_event(
                        job.id,
                        "job_queued",
                        f"训练任务 {job.id} 已加入GPU队列，等待GPU {','.join(unavailable_gpus)} 可用",
                    )

                    self.logger.info(f"创建训练任务 {job.id} 并加入GPU队列")
                    return job
                else:
                    raise ValueError("无法将任务添加到GPU队列")

            # 如果GPU可用，直接创建任务
            if gpu_available:
                # 保存到Redis
                self.redis_service.save_training_job(job)

                # 添加创建事件
                self.redis_service.add_training_event(
                    job.id, "job_created", f"训练任务 {job.id} 已创建"
                )

                self.logger.info(f"创建训练任务 {job.id}")
                return job

        except (ValueError, KeyError) as e:
            self.logger.error(f"创建训练任务参数错误: {str(e)}")
            raise
        except (ConnectionError, TimeoutError) as e:
            self.logger.error(f"创建训练任务时服务连接失败: {str(e)}")
            raise RuntimeError(f"服务连接失败: {str(e)}")
        except Exception as e:
            self.logger.error(f"创建训练任务失败: {str(e)}", exc_info=True)
            raise RuntimeError(f"创建训练任务失败: {str(e)}")

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
            gpu_ids = job.gpu_id.split(",")
            for gpu_id in gpu_ids:
                if not self.gpu_manager.check_gpu_availability(gpu_id.strip()):
                    raise ValueError(f"GPU {gpu_id} 不可用")

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
                bufsize=1,
            )

            # 保存进程信息
            job.process_id = process.pid
            self.redis_service.save_training_job(job)
            self.active_processes[job_id] = process

            # 启动监控线程
            monitor_thread = threading.Thread(
                target=self._monitor_training_process,
                args=(job_id, process, training_logger),
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

            # 如果任务在队列中，从队列中移除
            if job.status == TrainingStatus.PENDING:
                self.redis_service.remove_job_from_gpu_queue(job_id)

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
            gpu_ids = job.gpu_id.split(",")
            for gpu_id in gpu_ids:
                logger.info(f"释放GPU {gpu_id.strip()}")

            # 添加停止事件
            self.redis_service.add_training_event(
                job_id, "training_cancelled", f"训练任务 {job_id} 已取消"
            )

            self.logger.info(f"停止训练任务 {job_id}")
            return True

        except Exception as e:
            self.logger.error(f"停止训练任务失败: {str(e)}")
            return False

    def export_model(self, job_id: str) -> bool:
        """手动触发模型导出和合并"""
        try:
            job = self.redis_service.get_training_job(job_id)
            if job is None:
                raise ValueError(f"训练任务 {job_id} 不存在")

            if job.status != TrainingStatus.COMPLETED:
                raise ValueError(
                    f"只有已完成的训练任务才能导出模型，当前状态: {job.status}"
                )

            if job.export_completed:
                raise ValueError(f"训练任务 {job_id} 已经完成导出")

            # 创建训练日志记录器
            training_logger = get_training_logger(job_id)

            # 在后台线程中执行导出
            export_thread = threading.Thread(
                target=self._export_and_merge_model, args=(job_id, job, training_logger)
            )
            export_thread.daemon = True
            export_thread.start()

            self.logger.info(f"开始导出模型 {job_id}")
            return True

        except Exception as e:
            self.logger.error(f"导出模型失败: {str(e)}")
            return False

    def process_gpu_queue(self) -> Dict[str, Any]:
        """处理GPU队列，尝试启动队列中的任务"""
        try:
            result = {"processed": 0, "started": 0, "failed": 0, "details": []}

            # 获取队列中的任务
            queue_item = self.redis_service.get_job_from_gpu_queue()
            if not queue_item:
                return result

            job_id = queue_item["job_id"]
            queue_data = queue_item["queue_data"]
            original_gpu_ids = queue_data["gpu_ids"]

            # 检查原始GPU是否可用
            gpu_availability = self.redis_service.check_gpu_availability_for_queue(
                original_gpu_ids
            )

            if gpu_availability["can_start"]:
                # 原始GPU可用，尝试启动任务
                try:
                    success = self.start_training(job_id)
                    if success:
                        # 从队列中移除任务
                        self.redis_service.remove_job_from_gpu_queue(job_id)
                        result["started"] += 1
                        result["details"].append(
                            {
                                "job_id": job_id,
                                "action": "started",
                                "gpu_ids": original_gpu_ids,
                            }
                        )
                        self.logger.info(f"从队列启动训练任务 {job_id}")
                    else:
                        result["failed"] += 1
                        result["details"].append(
                            {
                                "job_id": job_id,
                                "action": "failed_to_start",
                                "error": "启动任务失败",
                            }
                        )
                except Exception as e:
                    result["failed"] += 1
                    result["details"].append(
                        {"job_id": job_id, "action": "failed_to_start", "error": str(e)}
                    )
                    self.logger.error(f"从队列启动任务失败 {job_id}: {str(e)}")
            else:
                # 原始GPU不可用，尝试重新分配GPU
                try:
                    # 获取任务信息以确定需要的GPU数量
                    job = self.redis_service.get_training_job(job_id)
                    if job:
                        gpu_count = len(original_gpu_ids)
                        new_gpu_ids = self._auto_allocate_gpus(gpu_count)

                        if new_gpu_ids:
                            # 更新任务的GPU分配
                            job.gpu_id = ",".join(new_gpu_ids)
                            self.redis_service.save_training_job(job)

                            # 更新队列中的GPU信息
                            queue_data["gpu_ids"] = new_gpu_ids
                            self.redis_service.redis_client.setex(
                                f"gpu_queue_detail:{job_id}",
                                86400 * 7,
                                json.dumps(queue_data),
                            )

                            # 尝试启动任务
                            success = self.start_training(job_id)
                            if success:
                                # 从队列中移除任务
                                self.redis_service.remove_job_from_gpu_queue(job_id)
                                result["started"] += 1
                                result["details"].append(
                                    {
                                        "job_id": job_id,
                                        "action": "started_with_reallocation",
                                        "original_gpu_ids": original_gpu_ids,
                                        "new_gpu_ids": new_gpu_ids,
                                    }
                                )
                                self.logger.info(
                                    f"从队列启动训练任务 {job_id}，重新分配GPU: {original_gpu_ids} -> {new_gpu_ids}"
                                )
                            else:
                                result["details"].append(
                                    {
                                        "job_id": job_id,
                                        "action": "reallocation_failed",
                                        "original_gpu_ids": original_gpu_ids,
                                        "new_gpu_ids": new_gpu_ids,
                                        "error": "重新分配GPU后启动失败",
                                    }
                                )
                        else:
                            # 无法重新分配GPU
                            result["details"].append(
                                {
                                    "job_id": job_id,
                                    "action": "skipped",
                                    "reason": f"无法重新分配GPU，原始GPU {original_gpu_ids} 不可用",
                                }
                            )
                    else:
                        result["details"].append(
                            {
                                "job_id": job_id,
                                "action": "skipped",
                                "reason": "任务不存在",
                            }
                        )

                except Exception as e:
                    result["details"].append(
                        {
                            "job_id": job_id,
                            "action": "reallocation_error",
                            "error": str(e),
                        }
                    )
                    self.logger.error(f"重新分配GPU失败 {job_id}: {str(e)}")

            result["processed"] += 1
            return result

        except Exception as e:
            self.logger.error(f"处理GPU队列失败: {str(e)}")
            return {
                "processed": 0,
                "started": 0,
                "failed": 1,
                "details": [{"error": str(e)}],
            }

    def get_queue_status(self) -> Dict[str, Any]:
        """获取GPU队列状态"""
        try:
            return self.redis_service.get_gpu_queue_status()
        except Exception as e:
            self.logger.error(f"获取队列状态失败: {str(e)}")
            return {"total_queued": 0, "queue_items": []}

    def remove_job_from_queue(self, job_id: str) -> bool:
        """从队列中移除任务"""
        try:
            # 检查任务是否在队列中
            queue_status = self.get_queue_status()
            job_in_queue = any(
                item["job_id"] == job_id for item in queue_status["queue_items"]
            )

            if not job_in_queue:
                raise ValueError(f"任务 {job_id} 不在队列中")

            # 从队列中移除
            success = self.redis_service.remove_job_from_gpu_queue(job_id)

            if success:
                # 添加事件
                self.redis_service.add_training_event(
                    job_id,
                    "job_removed_from_queue",
                    f"训练任务 {job_id} 已从GPU队列中移除",
                )
                self.logger.info(f"从队列中移除任务 {job_id}")

            return success

        except Exception as e:
            self.logger.error(f"从队列移除任务失败 {job_id}: {str(e)}")
            return False

    def start_queue_processor(self) -> bool:
        """启动后台队列处理器"""
        try:
            if self.queue_processor_running:
                self.logger.warning("队列处理器已经在运行")
                return True

            self.queue_processor_running = True
            self.queue_processor_thread = threading.Thread(
                target=self._queue_processor_loop, daemon=True
            )
            self.queue_processor_thread.start()

            self.logger.info("GPU队列处理器已启动")
            return True

        except Exception as e:
            self.logger.error(f"启动队列处理器失败: {str(e)}")
            self.queue_processor_running = False
            return False

    def stop_queue_processor(self) -> bool:
        """停止后台队列处理器"""
        try:
            if not self.queue_processor_running:
                self.logger.warning("队列处理器未在运行")
                return True

            self.queue_processor_running = False
            if self.queue_processor_thread:
                self.queue_processor_thread.join(timeout=10)

            self.logger.info("GPU队列处理器已停止")
            return True

        except Exception as e:
            self.logger.error(f"停止队列处理器失败: {str(e)}")
            return False

    def _queue_processor_loop(self):
        """队列处理器主循环"""
        self.logger.info("GPU队列处理器开始运行")

        while self.queue_processor_running:
            try:
                # 处理队列中的任务
                result = self.process_gpu_queue()

                if result["started"] > 0:
                    self.logger.info(f"队列处理器启动了 {result['started']} 个任务")

                # 等待一段时间再处理下一个任务
                time.sleep(settings.QUEUE_CHECK_INTERVAL)  # 使用配置的检查间隔

            except Exception as e:
                self.logger.error(f"队列处理器循环出错: {str(e)}")
                time.sleep(settings.QUEUE_CHECK_INTERVAL * 2)  # 出错时等待更长时间

        self.logger.info("GPU队列处理器已停止")

    def _auto_allocate_gpus(self, gpu_count: int) -> List[str]:
        """自动分配指定数量的GPU"""
        try:
            # 获取所有GPU信息
            gpu_info = self.gpu_manager.get_gpu_info()

            if not gpu_info:
                raise ValueError("未检测到可用的GPU设备")

            available_gpus = []

            # 按可用内存排序，优先选择内存大的GPU
            sorted_gpus = sorted(gpu_info, key=lambda x: x["memory_free"], reverse=True)

            for gpu in sorted_gpus:
                gpu_id = str(gpu["index"])
                if self.gpu_manager.check_gpu_availability(gpu_id):
                    available_gpus.append(gpu_id)
                    if len(available_gpus) >= gpu_count:
                        break

            if len(available_gpus) < gpu_count:
                # 如果没有足够的可用GPU，使用前几个GPU（即使不可用）
                # 这些GPU会在后续的队列处理中被重新分配
                total_gpus = len(gpu_info)
                needed_gpus = min(gpu_count, total_gpus)

                # 取前几个GPU的ID
                fallback_gpus = [str(gpu["index"]) for gpu in sorted_gpus[:needed_gpus]]

                self.logger.warning(
                    f"GPU资源不足，需要 {gpu_count} 个GPU，"
                    f"系统共有 {total_gpus} 个GPU，"
                    f"当前可用 {len(available_gpus)} 个GPU。"
                    f"使用备用GPU: {fallback_gpus}，任务将加入队列等待。"
                )

                return fallback_gpus

            self.logger.info(f"自动分配GPU: {available_gpus}")
            return available_gpus

        except Exception as e:
            self.logger.error(f"自动分配GPU失败: {str(e)}")
            raise

    def get_training_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """获取训练状态"""
        try:
            job = self.redis_service.get_training_job(job_id)
            if job is None:
                self.logger.warning(f"训练任务 {job_id} 不存在")
                return None

            # 获取训练进度
            progress_data = self.redis_service.get_training_progress(job_id)
            self.logger.debug(f"获取到训练进度数据: {progress_data}")

            # 获取GPU使用情况 - 修复数据结构以匹配模型定义
            gpu_memory_usage = {}
            gpu_ids = job.gpu_id.split(",")
            for gpu_id in gpu_ids:
                try:
                    gpu_info = self.gpu_manager.get_gpu_memory_usage(gpu_id.strip())
                    if gpu_info:
                        # 只返回内存使用率作为float值，匹配模型定义
                        memory_usage_ratio = (
                            gpu_info["memory_used"] / gpu_info["memory_total"]
                        )
                        gpu_memory_usage[gpu_id.strip()] = memory_usage_ratio
                    else:
                        self.logger.warning(
                            f"无法获取GPU {gpu_id.strip()} 的内存使用信息"
                        )
                except Exception as e:
                    self.logger.error(
                        f"获取GPU {gpu_id.strip()} 内存使用信息失败: {str(e)}"
                    )
                    # 继续处理其他GPU，不中断整个流程

            status_data = {
                "job_id": job_id,
                "status": job.status,
                "progress": progress_data.get("progress", 0.0)
                if progress_data
                else 0.0,
                "current_epoch": progress_data.get("current_epoch"),
                "current_step": progress_data.get("current_step"),
                "loss": progress_data.get("loss"),
                "learning_rate": progress_data.get("learning_rate"),
                "gpu_memory_usage": gpu_memory_usage,
                "estimated_time_remaining": progress_data.get(
                    "estimated_time_remaining"
                ),
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                # 导出信息
                "export_completed": job.export_completed,
                "export_time": job.export_time,
                "export_path": job.export_path,
                "export_error": job.export_error,
            }

            self.logger.debug(f"生成的训练状态数据: {status_data}")
            return status_data

        except Exception as e:
            self.logger.error(f"获取训练状态失败: {str(e)}", exc_info=True)
            return None

    def _build_training_command(self, job: TrainingJob) -> List[str]:
        """构建训练命令 - 支持多任务类型"""
        if job.task_type == "multimodal":
            command = [
                "swift",
                "sft",
                "--model",
                job.model_path,
                "--dataset",
                job.data_path,
                "--train_type",
                job.train_type,
                "--torch_dtype",
                job.torch_dtype,
                "--num_train_epochs",
                str(job.num_epochs),
                "--per_device_train_batch_size",
                str(job.batch_size),
                "--per_device_eval_batch_size",
                str(job.batch_size),
                "--learning_rate",
                str(job.learning_rate),
                "--vit_lr",
                str(job.vit_lr),
                "--aligner_lr",
                str(job.aligner_lr),
                "--lora_rank",
                str(job.lora_rank),
                "--lora_alpha",
                str(job.lora_alpha),
                "--gradient_accumulation_steps",
                str(job.gradient_accumulation_steps),
                "--eval_steps",
                str(job.eval_steps),
                "--save_steps",
                str(job.save_steps),
                "--save_total_limit",
                str(job.save_total_limit),
                "--logging_steps",
                str(job.logging_steps),
                "--max_length",
                str(job.max_length),
                "--output_dir",
                job.output_dir,
                "--warmup_ratio",
                str(job.warmup_ratio),
                "--dataloader_num_workers",
                str(job.dataloader_num_workers),
                "--dataset_num_proc",
                str(job.dataset_num_proc),
            ]
            if job.save_only_model:
                command.append("--save_only_model")
            return command
        elif job.task_type == "language_model":
            command = [
                "swift",
                "sft",
                "--model",
                job.model_path,
                "--dataset",
                job.data_path,
                "--train_type",
                job.train_type,
                "--torch_dtype",
                job.torch_dtype,
                "--num_train_epochs",
                str(job.num_epochs),
                "--per_device_train_batch_size",
                str(job.batch_size),
                "--per_device_eval_batch_size",
                str(job.batch_size),
                "--learning_rate",
                str(job.learning_rate),
                "--gradient_accumulation_steps",
                str(job.gradient_accumulation_steps),
                "--eval_steps",
                str(job.eval_steps),
                "--save_steps",
                str(job.save_steps),
                "--save_total_limit",
                str(job.save_total_limit),
                "--logging_steps",
                str(job.logging_steps),
                "--max_length",
                str(job.max_length),
                "--output_dir",
                job.output_dir,
                "--warmup_ratio",
                str(job.warmup_ratio),
                "--dataloader_num_workers",
                str(job.dataloader_num_workers),
                "--dataset_num_proc",
                str(job.dataset_num_proc),
            ]
            if job.save_only_model:
                command.append("--save_only_model")
            return command
        else:
            raise ValueError(f"不支持的任务类型: {job.task_type}")

    def _build_environment(self, job: TrainingJob) -> Dict[str, str]:
        """构建环境变量"""
        env = os.environ.copy()

        # 设置GPU环境变量
        env["CUDA_VISIBLE_DEVICES"] = job.gpu_id

        # 设置NCCL环境变量（用于多GPU训练）
        if len(job.gpu_id.split(",")) > 1:
            env["NCCL_P2P_DISABLE"] = "1"
            env["NCCL_IB_DISABLE"] = "1"
            env["NPROC_PER_NODE"] = str(len(job.gpu_id.split(",")))

        return env

    def _monitor_training_process(
        self, job_id: str, process: subprocess.Popen, training_logger
    ):
        """监控训练进程"""
        try:
            start_time = time.time()

            # 读取进程输出
            for line in iter(process.stdout.readline, ""):
                if line:
                    line = line.strip()
                    training_logger.info(f"{line}")

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
                    training_logger.training_completed(
                        0.0, training_time
                    )  # 这里需要从日志中提取最终损失

                    # 训练成功后，自动触发模型导出和合并
                    self.logger.info(f"训练任务 {job_id} 完成，开始自动导出和合并模型")
                    try:
                        # 在后台线程中执行导出，避免阻塞主流程
                        export_thread = threading.Thread(
                            target=self._export_and_merge_model,
                            args=(job_id, job, training_logger),
                        )
                        export_thread.daemon = True
                        export_thread.start()
                        self.logger.info(f"模型导出和合并任务已启动: {job_id}")
                    except Exception as export_error:
                        self.logger.error(
                            f"启动模型导出失败 {job_id}: {str(export_error)}"
                        )
                        self.redis_service.add_training_event(
                            job_id,
                            "export_failed",
                            f"启动导出失败: {str(export_error)}",
                        )
                else:
                    job.status = TrainingStatus.FAILED
                    job.error_message = f"训练进程返回错误代码: {return_code}"
                    training_logger.training_failed(f"进程返回错误代码: {return_code}")

                job.completed_at = datetime.now()
                self.redis_service.save_training_job(job)

            # 释放GPU资源
            if job:
                gpu_ids = job.gpu_id.split(",")
                for gpu_id in gpu_ids:
                    logger.info(f"释放GPU {gpu_id.strip()}")

            # 清理进程记录
            if job_id in self.active_processes:
                del self.active_processes[job_id]

            # 添加完成事件
            event_type = "training_completed" if return_code == 0 else "training_failed"
            event_message = (
                f"训练任务 {job_id} {'完成' if return_code == 0 else '失败'}"
            )
            self.redis_service.add_training_event(job_id, event_type, event_message)

        except Exception as e:
            self.logger.error(f"监控训练进程失败: {str(e)}")
            # 更新任务状态为失败
            job = self.redis_service.get_training_job(job_id)
            if job:
                job.status = TrainingStatus.FAILED
                job.error_message = str(e)
                self.redis_service.save_training_job(job)
                # 确保释放GPU资源
                gpu_ids = job.gpu_id.split(",")
                for gpu_id in gpu_ids:
                    logger.info(f"释放GPU {gpu_id.strip()}")
                # 清理进程记录
                if job_id in self.active_processes:
                    del self.active_processes[job_id]
                # 添加失败事件
                self.redis_service.add_training_event(
                    job_id, "training_failed", f"监控失败: {str(e)}"
                )

    def _parse_training_progress(self, job_id: str, line: str, training_logger):
        """解析训练进度 - 使用批量更新优化"""
        try:
            # re imported at top level

            # 初始化更新字典
            updates = {}

            # 1. 解析Swift训练进度条格式: Train: 1%| | 35/4950 [01:10<2:41:13, 1.97s/it]
            progress_match = re.search(
                r"Train:\s*(\d+)%\s*\|.*?(\d+)/(\d+)\s*\[([^<]+)<([^,]+),\s*([\d.]+)s/it\]",
                line,
            )
            if progress_match:
                progress_percent = int(progress_match.group(1))
                current_step = int(progress_match.group(2))
                total_steps = int(progress_match.group(3))
                elapsed_time = progress_match.group(4)
                remaining_time = progress_match.group(5)
                time_per_iter = float(progress_match.group(6))

                # 计算更精确的进度，保留两位小数
                if total_steps > 0:
                    calculated_progress = round((current_step / total_steps) * 100, 2)
                else:
                    calculated_progress = round(progress_percent, 2)

                # 收集进度更新
                updates.update(
                    {
                        "progress": calculated_progress,
                        "current_step": current_step,
                        "total_steps": total_steps,
                        "current_epoch": 1,  # Swift通常只有一个epoch
                        "elapsed_time": elapsed_time,
                        "remaining_time": remaining_time,
                        "time_per_iter": time_per_iter,
                        "estimated_time_remaining": remaining_time,
                    }
                )

                training_logger.info(
                    f"训练进度: {calculated_progress}% ({current_step}/{total_steps})"
                )

            # 2. 解析损失信息
            loss_match = re.search(r"loss:\s*([\d.]+)", line.lower())
            if loss_match:
                loss = float(loss_match.group(1))
                updates["loss"] = loss

            # 3. 解析学习率信息
            lr_match = re.search(r"lr:\s*([\d.e+-]+)", line.lower())
            if lr_match:
                learning_rate = float(lr_match.group(1))
                updates["learning_rate"] = learning_rate

            # 4. 检查是否包含检查点保存信息
            if "saving checkpoint" in line.lower():
                # 提取检查点路径
                checkpoint_match = re.search(r"checkpoint.*?(\S+)", line)
                if checkpoint_match:
                    checkpoint_path = checkpoint_match.group(1)
                    current_step = updates.get("current_step", 0)
                    training_logger.checkpoint_saved(checkpoint_path, current_step)

            # 5. 批量保存所有更新（如果有的话）
            if updates:
                self.redis_service.save_training_progress(job_id, **updates)

        except Exception as e:
            self.logger.error(f"解析训练进度失败: {str(e)}")
            # 不影响训练进程，只记录错误

    def _export_and_merge_model(self, job_id: str, job: TrainingJob, training_logger):
        """训练完成后自动导出和合并模型"""
        try:
            training_logger.info(f"开始执行模型导出和合并操作: {job_id}")
            self.redis_service.add_training_event(
                job_id, "export_started", "开始模型导出和合并"
            )

            # 查找最新的检查点目录
            checkpoint_dir = self._find_latest_checkpoint(job.output_dir)
            if not checkpoint_dir:
                raise ValueError(f"未找到检查点目录: {job.output_dir}")

            training_logger.info(f"找到检查点目录: {checkpoint_dir}")

            # 构建导出命令
            export_command = self._build_export_command(checkpoint_dir)

            # 设置环境变量
            env = self._build_environment(job)

            # 执行导出命令
            training_logger.info(f"执行导出命令: {' '.join(export_command)}")

            export_process = subprocess.Popen(
                export_command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            # 监控导出进程
            export_start_time = time.time()
            for line in iter(export_process.stdout.readline, ""):
                if line:
                    line = line.strip()
                    training_logger.info(f"导出输出: {line}")

            # 等待导出进程结束
            export_return_code = export_process.wait()
            export_time = time.time() - export_start_time

            if export_return_code == 0:
                training_logger.info(
                    f"模型导出和合并成功完成，耗时: {export_time:.2f}秒"
                )
                self.redis_service.add_training_event(
                    job_id,
                    "export_completed",
                    f"模型导出和合并成功完成，耗时: {export_time:.2f}秒",
                )

                # 更新任务状态，添加导出完成信息
                job.export_completed = True
                job.export_time = export_time

                job.export_path = f"{checkpoint_dir}-merged"
                self.redis_service.save_training_job(job)
            else:
                error_msg = f"模型导出失败，返回代码: {export_return_code}"
                training_logger.error(error_msg)
                self.redis_service.add_training_event(
                    job_id, "export_failed", error_msg
                )

                # 更新任务状态，记录导出失败信息
                job.export_error = error_msg
                self.redis_service.save_training_job(job)
                raise RuntimeError(error_msg)

        except Exception as e:
            error_msg = f"模型导出和合并失败: {str(e)}"
            training_logger.error(error_msg)
            self.redis_service.add_training_event(job_id, "export_failed", error_msg)

            # 更新任务状态，记录导出失败信息
            try:
                job = self.redis_service.get_training_job(job_id)
                if job:
                    job.export_error = error_msg
                    self.redis_service.save_training_job(job)
            except Exception as save_error:
                self.logger.error(f"保存导出错误状态失败: {str(save_error)}")

            # 不抛出异常，避免影响训练完成状态

    def _find_latest_checkpoint(self, output_dir: str) -> Optional[str]:
        """查找最新的检查点目录"""
        try:
            if not os.path.exists(output_dir):
                return None

            # 查找checkpoint-*目录，包括子文件夹中的
            checkpoint_dirs = []

            # 首先检查output_dir直接下的checkpoint目录
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isdir(item_path) and item.startswith("checkpoint-"):
                    checkpoint_dirs.append(item_path)

            # 如果没有找到，检查以"v"开头的子文件夹
            if not checkpoint_dirs:
                v_folders = []
                for item in os.listdir(output_dir):
                    item_path = os.path.join(output_dir, item)
                    if os.path.isdir(item_path) and item.startswith("v"):
                        v_folders.append(item_path)

                # 如果有多个v文件夹，按时间排序选择最新的
                if v_folders:
                    # 按文件夹名称中的时间戳排序（格式如：v0-20250703-155941）
                    v_folders.sort(
                        key=lambda x: x.split("-")[-2] + "-" + x.split("-")[-1],
                        reverse=True,
                    )
                    latest_v_folder = v_folders[0]

                    # 在最新的v文件夹中查找checkpoint目录
                    for sub_item in os.listdir(latest_v_folder):
                        sub_item_path = os.path.join(latest_v_folder, sub_item)
                        if os.path.isdir(sub_item_path) and sub_item.startswith(
                            "checkpoint-"
                        ):
                            checkpoint_dirs.append(sub_item_path)

            if not checkpoint_dirs:
                return None

            # 按目录名排序，获取最新的检查点
            checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
            return checkpoint_dirs[-1]

        except Exception as e:
            self.logger.error(f"查找检查点目录失败: {str(e)}")
            return None

    def _build_export_command(self, checkpoint_dir: str) -> List[str]:
        """构建模型导出命令"""

        command = [
            "swift",
            "export",
            "--adapters",
            checkpoint_dir,
            "--merge_lora",
            "true",
        ]

        return command


# 创建全局训练服务实例
training_service = TrainingService()


def get_training_service() -> TrainingService:
    """获取训练服务实例"""
    return training_service
