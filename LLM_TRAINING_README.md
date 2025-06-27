# Swift LLM训练功能使用指南

## 概述

本系统现在支持两种类型的训练任务：
- **VLM训练**：多模态视觉语言模型训练
- **LLM训练**：大语言模型训练

本文档主要介绍LLM训练功能的使用方法。

## LLM训练命令示例

LLM训练使用以下Swift命令格式：

```bash
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
```

## API接口

### 1. 创建LLM训练任务

**接口**: `POST /training/llm/jobs`

**请求体**:
```json
{
    "gpu_id": "0",
    "datasets": [
        "AI-ModelScope/alpaca-gpt4-data-zh#500",
        "AI-ModelScope/alpaca-gpt4-data-en#500",
        "swift/self-cognition#500"
    ],
    "model_path": "Qwen/Qwen2.5-7B-Instruct",
    "output_dir": "output/llm_test",
    "num_epochs": 1,
    "batch_size": 1,
    "learning_rate": 1e-4,
    "lora_rank": 8,
    "lora_alpha": 32,
    "target_modules": "all-linear",
    "gradient_accumulation_steps": 16,
    "eval_steps": 50,
    "save_steps": 50,
    "save_total_limit": 2,
    "logging_steps": 5,
    "max_length": 2048,
    "warmup_ratio": 0.05,
    "dataloader_num_workers": 4,
    "torch_dtype": "bfloat16",
    "system": "You are a helpful assistant.",
    "model_author": "swift",
    "model_name": "swift-robot"
}
```

**响应**:
```json
{
    "job_id": "uuid-string",
    "status": "pending",
    "message": "LLM训练任务创建成功"
}
```

### 2. 启动LLM训练任务

**接口**: `POST /training/jobs/{job_id}/start`

**响应**:
```json
{
    "job_id": "uuid-string",
    "status": "running",
    "message": "训练任务启动成功"
}
```

### 3. 获取LLM训练任务列表

**接口**: `GET /training/llm/jobs`

**查询参数**:
- `page`: 页码（默认1）
- `size`: 每页大小（默认10）
- `status`: 状态过滤（可选）

**响应**:
```json
{
    "jobs": [
        {
            "id": "uuid-string",
            "status": "completed",
            "created_at": "2024-01-01T00:00:00",
            "gpu_id": "0",
            "model_path": "Qwen/Qwen2.5-7B-Instruct",
            "datasets": ["dataset1", "dataset2"],
            "output_dir": "output/llm_test"
        }
    ],
    "total": 1,
    "page": 1,
    "size": 10
}
```

### 4. 获取所有训练任务列表

**接口**: `GET /training/all/jobs`

**响应**:
```json
{
    "jobs": [
        {
            "id": "uuid-string",
            "training_type": "llm",
            "status": "completed",
            "created_at": "2024-01-01T00:00:00"
        }
    ],
    "total": 1,
    "page": 1,
    "size": 10,
    "total_vlm": 0,
    "total_llm": 1
}
```

### 5. 获取训练任务状态

**接口**: `GET /training/jobs/{job_id}/status`

**响应**:
```json
{
    "job_id": "uuid-string",
    "training_type": "llm",
    "status": "running",
    "progress": 45.5,
    "current_epoch": 1,
    "current_step": 100,
    "loss": 2.34,
    "learning_rate": 1e-4,
    "gpu_memory_usage": {
        "0": {
            "memory_used": 15000,
            "memory_total": 24000,
            "utilization": 85
        }
    },
    "created_at": "2024-01-01T00:00:00",
    "started_at": "2024-01-01T00:05:00"
}
```

### 6. 停止训练任务

**接口**: `POST /training/jobs/{job_id}/stop`

**响应**:
```json
{
    "job_id": "uuid-string",
    "status": "cancelled",
    "message": "训练任务停止成功"
}
```

### 7. 手动触发模型导出

**接口**: `POST /training/jobs/{job_id}/export`

**响应**:
```json
{
    "job_id": "uuid-string",
    "status": "exporting",
    "message": "模型导出已开始"
}
```

## 使用示例

### Python客户端示例

```python
import requests

# 创建LLM训练任务
def create_llm_job():
    url = "http://localhost:8000/training/llm/jobs"
    data = {
        "gpu_id": "0",
        "datasets": [
            "AI-ModelScope/alpaca-gpt4-data-zh#500",
            "AI-ModelScope/alpaca-gpt4-data-en#500"
        ],
        "model_path": "Qwen/Qwen2.5-7B-Instruct",
        "output_dir": "output/llm_test",
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-4,
        "lora_rank": 8,
        "lora_alpha": 32,
        "target_modules": "all-linear",
        "gradient_accumulation_steps": 16,
        "eval_steps": 50,
        "save_steps": 50,
        "save_total_limit": 2,
        "logging_steps": 5,
        "max_length": 2048,
        "warmup_ratio": 0.05,
        "dataloader_num_workers": 4,
        "torch_dtype": "bfloat16",
        "system": "You are a helpful assistant.",
        "model_author": "swift",
        "model_name": "swift-robot"
    }
    
    response = requests.post(url, json=data)
    return response.json()

# 启动训练
def start_training(job_id):
    url = f"http://localhost:8000/training/jobs/{job_id}/start"
    response = requests.post(url)
    return response.json()

# 获取状态
def get_status(job_id):
    url = f"http://localhost:8000/training/jobs/{job_id}/status"
    response = requests.get(url)
    return response.json()

# 使用示例
if __name__ == "__main__":
    # 创建任务
    result = create_llm_job()
    job_id = result["job_id"]
    print(f"创建任务: {job_id}")
    
    # 启动训练
    start_result = start_training(job_id)
    print(f"启动训练: {start_result}")
    
    # 监控状态
    status = get_status(job_id)
    print(f"当前状态: {status}")
```

### cURL示例

```bash
# 创建LLM训练任务
curl -X POST "http://localhost:8000/training/llm/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "gpu_id": "0",
    "datasets": ["AI-ModelScope/alpaca-gpt4-data-zh#500"],
    "model_path": "Qwen/Qwen2.5-7B-Instruct",
    "output_dir": "output/llm_test",
    "num_epochs": 1,
    "batch_size": 1,
    "learning_rate": 1e-4,
    "lora_rank": 8,
    "lora_alpha": 32,
    "target_modules": "all-linear",
    "gradient_accumulation_steps": 16,
    "eval_steps": 50,
    "save_steps": 50,
    "save_total_limit": 2,
    "logging_steps": 5,
    "max_length": 2048,
    "warmup_ratio": 0.05,
    "dataloader_num_workers": 4,
    "torch_dtype": "bfloat16",
    "system": "You are a helpful assistant.",
    "model_author": "swift",
    "model_name": "swift-robot"
  }'

# 启动训练
curl -X POST "http://localhost:8000/training/jobs/{job_id}/start"

# 获取状态
curl -X GET "http://localhost:8000/training/jobs/{job_id}/status"

# 获取LLM任务列表
curl -X GET "http://localhost:8000/training/llm/jobs"

# 获取所有任务列表
curl -X GET "http://localhost:8000/training/all/jobs"
```

## 测试

运行测试脚本：

```bash
cd swift_trainer_api
python test_llm_training.py
```

## 注意事项

1. **GPU资源管理**: 确保指定的GPU ID可用且未被其他任务占用
2. **数据集格式**: 数据集需要符合Swift框架的要求格式
3. **模型路径**: 确保模型路径正确且可访问
4. **输出目录**: 确保输出目录有写入权限
5. **训练参数**: 根据硬件配置调整batch_size、gradient_accumulation_steps等参数

## 故障排除

### 常见问题

1. **GPU不可用**: 检查GPU ID是否正确，GPU是否被其他进程占用
2. **数据集加载失败**: 检查数据集路径和格式是否正确
3. **内存不足**: 减小batch_size或gradient_accumulation_steps
4. **训练中断**: 检查日志文件，查看具体错误信息

### 日志查看

训练日志保存在 `logs/` 目录下：
- VLM训练: `logs/training_{job_id}.log`
- LLM训练: `logs/llm_training_{job_id}.log`

### 状态监控

可以通过API接口实时监控训练状态：
- 训练进度
- GPU使用情况
- 损失值变化
- 预计完成时间 