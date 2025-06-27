# Swift LLM训练功能使用指南

## 概述

本系统现在支持两种类型的训练任务：
- **VLM训练**：多模态视觉语言模型训练
- **LLM训练**：大语言模型训练

**重要说明**：LLM训练使用固定的参数配置，用户只需要指定GPU ID和输出目录，其他参数都是预设的固定值。

## LLM训练固定配置

LLM训练使用以下固定配置：

### 固定模型和数据集
- **模型**: `Qwen/Qwen2.5-7B-Instruct`
- **数据集**: 
  - `AI-ModelScope/alpaca-gpt4-data-zh#500`
  - `AI-ModelScope/alpaca-gpt4-data-en#500`
  - `swift/self-cognition#500`

### 固定训练参数
- **训练轮数**: 1 epoch
- **批次大小**: 1
- **学习率**: 1e-4
- **LoRA rank**: 8
- **LoRA alpha**: 32
- **目标模块**: all-linear
- **梯度累积步数**: 16
- **评估步数**: 50
- **保存步数**: 50
- **保存总数限制**: 2
- **日志步数**: 5
- **最大长度**: 2048
- **预热比例**: 0.05
- **数据加载器工作进程数**: 4
- **PyTorch数据类型**: bfloat16
- **系统提示**: "You are a helpful assistant."
- **模型作者**: swift
- **模型名称**: swift-robot

## LLM训练命令示例

LLM训练使用以下Swift命令格式（固定参数）：

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
    --output_dir {output_dir} \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
```

## API接口

### 1. 创建LLM训练任务

**接口**: `POST /training/llm/jobs`

**请求体** (只需要提供必要参数):
```json
{
    "gpu_id": "0",
    "output_dir": "output/llm_test"
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

**注意**: 所有其他参数都使用固定值，用户无需指定。

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
            "datasets": [
                "AI-ModelScope/alpaca-gpt4-data-zh#500",
                "AI-ModelScope/alpaca-gpt4-data-en#500",
                "swift/self-cognition#500"
            ],
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

# 创建LLM训练任务 - 只需要提供GPU ID和输出目录
def create_llm_job():
    url = "http://localhost:8000/training/llm/jobs"
    data = {
        "gpu_id": "0",
        "output_dir": "output/llm_test"
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
    # 创建任务 - 非常简单，只需要两个参数
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
# 创建LLM训练任务 - 非常简单
curl -X POST "http://localhost:8000/training/llm/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "gpu_id": "0",
    "output_dir": "output/llm_test"
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

1. **简化使用**: LLM训练现在只需要指定GPU ID和输出目录，其他参数都是固定的
2. **固定配置**: 所有训练参数都是经过优化的固定值，确保训练效果
3. **GPU资源管理**: 确保指定的GPU ID可用且未被其他任务占用
4. **输出目录**: 确保输出目录有写入权限
5. **模型和数据集**: 使用固定的模型和数据集，确保兼容性

## 故障排除

### 常见问题

1. **GPU不可用**: 检查GPU ID是否正确，GPU是否被其他进程占用
2. **输出目录权限**: 确保输出目录有写入权限
3. **内存不足**: 如果GPU内存不足，可能需要使用更小的GPU或等待其他任务完成
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

## 优势

1. **简化操作**: 用户只需要提供最基本的参数
2. **标准化**: 所有LLM训练使用相同的配置，确保一致性
3. **减少错误**: 避免用户配置错误导致的训练失败
4. **快速启动**: 无需复杂的参数配置，快速开始训练 