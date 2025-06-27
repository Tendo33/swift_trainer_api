import os
import signal
import subprocess
import threading
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional

from application.exceptions import ProcessError
from application.utils.logger import LogContext, get_system_logger

logger = get_system_logger()


class ProcessManager:
    """进程管理器，负责管理训练进程的生命周期"""
    
    def __init__(self):
        self.active_processes: Dict[str, subprocess.Popen] = {}
        self.process_metadata: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self.logger = logger
    
    def start_process(
        self,
        job_id: str,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        timeout: int = 300,  # 5分钟超时
        on_output: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None
    ) -> bool:
        """启动进程"""
        with LogContext(self.logger, "启动进程", job_id=job_id, command=command):
            try:
                # 检查进程是否已存在
                if job_id in self.active_processes:
                    self.logger.warning(f"任务 {job_id} 的进程已存在")
                    return False
                
                # 设置环境变量
                process_env = os.environ.copy()
                if env:
                    process_env.update(env)
                
                # 启动进程
                process = subprocess.Popen(
                    command,
                    env=process_env,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                    preexec_fn=os.setsid  # 创建新的进程组
                )
                
                # 记录进程信息
                with self._lock:
                    self.active_processes[job_id] = process
                    self.process_metadata[job_id] = {
                        "started_at": datetime.now(),
                        "command": command,
                        "cwd": cwd,
                        "timeout": timeout
                    }
                
                self.logger.info(f"进程启动成功，PID: {process.pid}")
                
                # 启动输出监控线程
                if on_output or on_error:
                    monitor_thread = threading.Thread(
                        target=self._monitor_process_output,
                        args=(job_id, process, on_output, on_error),
                        daemon=True
                    )
                    monitor_thread.start()
                
                # 启动超时监控线程
                if timeout > 0:
                    timeout_thread = threading.Thread(
                        target=self._monitor_timeout,
                        args=(job_id, timeout),
                        daemon=True
                    )
                    timeout_thread.start()
                
                return True
                
            except Exception as e:
                self.logger.error(f"启动进程失败: {str(e)}")
                raise ProcessError(f"启动进程失败: {str(e)}", operation="start")
    
    def stop_process(self, job_id: str, force: bool = False, timeout: int = 30) -> bool:
        """停止进程"""
        with LogContext(self.logger, "停止进程", job_id=job_id, force=force):
            try:
                if job_id not in self.active_processes:
                    self.logger.warning(f"任务 {job_id} 的进程不存在")
                    return False
                
                process = self.active_processes[job_id]
                
                if force:
                    # 强制终止进程组
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        self.logger.info(f"强制终止进程组 {process.pid}")
                    except ProcessLookupError:
                        self.logger.warning(f"进程 {process.pid} 已不存在")
                else:
                    # 优雅关闭
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                        self.logger.info(f"发送SIGTERM到进程组 {process.pid}")
                        
                        # 等待进程结束
                        try:
                            process.wait(timeout=timeout)
                        except subprocess.TimeoutExpired:
                            # 超时后强制终止
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                            self.logger.warning(f"进程 {process.pid} 超时，强制终止")
                    except ProcessLookupError:
                        self.logger.warning(f"进程 {process.pid} 已不存在")
                
                # 清理进程记录
                with self._lock:
                    self.active_processes.pop(job_id, None)
                    self.process_metadata.pop(job_id, None)
                
                return True
                
            except Exception as e:
                self.logger.error(f"停止进程失败: {str(e)}")
                raise ProcessError(f"停止进程失败: {str(e)}", process_id=process.pid if 'process' in locals() else None)
    
    def get_process_status(self, job_id: str) -> Optional[Dict]:
        """获取进程状态"""
        if job_id not in self.active_processes:
            return None
        
        process = self.active_processes[job_id]
        metadata = self.process_metadata.get(job_id, {})
        
        try:
            returncode = process.poll()
            return {
                "job_id": job_id,
                "pid": process.pid,
                "status": "running" if returncode is None else "terminated",
                "return_code": returncode,
                "started_at": metadata.get("started_at"),
                "runtime": (datetime.now() - metadata.get("started_at", datetime.now())).total_seconds() if metadata.get("started_at") else None
            }
        except Exception as e:
            self.logger.error(f"获取进程状态失败: {str(e)}")
            return None
    
    def get_all_processes(self) -> List[Dict]:
        """获取所有进程状态"""
        processes = []
        with self._lock:
            for job_id in list(self.active_processes.keys()):
                status = self.get_process_status(job_id)
                if status:
                    processes.append(status)
        return processes
    
    def cleanup_terminated_processes(self) -> int:
        """清理已终止的进程"""
        cleaned_count = 0
        with self._lock:
            for job_id in list(self.active_processes.keys()):
                process = self.active_processes[job_id]
                if process.poll() is not None:  # 进程已终止
                    self.active_processes.pop(job_id)
                    self.process_metadata.pop(job_id, None)
                    cleaned_count += 1
                    self.logger.info(f"清理已终止的进程: {job_id}")
        
        return cleaned_count
    
    def _monitor_process_output(
        self,
        job_id: str,
        process: subprocess.Popen,
        on_output: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None
    ):
        """监控进程输出"""
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    line = line.rstrip()
                    if on_output:
                        on_output(line)
                    
                    # 检查是否包含错误信息
                    if on_error and any(error_keyword in line.lower() for error_keyword in ['error', 'exception', 'failed', 'failure']):
                        on_error(line)
        except Exception as e:
            self.logger.error(f"监控进程输出失败: {str(e)}")
    
    def _monitor_timeout(self, job_id: str, timeout: int):
        """监控进程超时"""
        try:
            time.sleep(timeout)
            
            # 检查进程是否仍在运行
            if job_id in self.active_processes:
                process = self.active_processes[job_id]
                if process.poll() is None:  # 进程仍在运行
                    self.logger.warning(f"进程 {job_id} 超时，准备终止")
                    self.stop_process(job_id, force=True)
        except Exception as e:
            self.logger.error(f"监控进程超时失败: {str(e)}")
    
    def kill_all_processes(self):
        """终止所有进程"""
        with LogContext(self.logger, "终止所有进程"):
            job_ids = list(self.active_processes.keys())
            for job_id in job_ids:
                try:
                    self.stop_process(job_id, force=True)
                except Exception as e:
                    self.logger.error(f"终止进程 {job_id} 失败: {str(e)}")


# 全局进程管理器实例
process_manager = ProcessManager()


def get_process_manager() -> ProcessManager:
    """获取进程管理器实例"""
    return process_manager 