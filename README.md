# Swift Trainer API 🚀

基于 FastAPI 的 Swift 训练任务管理 API 系统，支持多 GPU 训练、Redis 状态管理、GPU自动排队功能和详细日志记录。

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![Redis](https://img.shields.io/badge/Redis-7+-red.svg)](https://redis.io/)
[![Swift](https://img.shields.io/badge/Swift-3.5.0-orange.svg)](https://github.com/modelscope/swift)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## ✨ 主要特性

- 🚀 **Swift训练任务管理**: 完整的训练任务生命周期管理
- 🎯 **GPU自动排队**: 智能GPU资源分配和排队机制
- 🔄 **优先级管理**: 支持任务优先级设置（0-10）
- 📊 **实时监控**: 训练进度、GPU状态、系统资源监控
- 💾 **Redis存储**: 持久化任务状态和训练数据
- 📝 **详细日志**: 完整的训练日志和事件记录
- 🐳 **Docker支持**: 一键部署和容器化运行
- 🔧 **RESTful API**: 标准化的API接口设计

## 🏗️ 项目结构

```
swift-api/
├── application/              # 主应用目录
│   ├── __init__.py
│   ├── main.py              # FastAPI应用入口
│   ├── config.py            # 配置管理
│   ├── models/              # 数据模型
│   │   ├── __init__.py
│   │   └── training_model.py # 训练任务模型
│   ├── services/            # 业务逻辑
│   │   ├── __init__.py
│   │   ├── training_service.py  # 训练服务
│   │   └── redis_service.py     # Redis服务
│   ├── api/                 # API路由
│   │   ├── __init__.py
│   │   └── training.py      # 训练相关API
│   └── utils/               # 工具函数
│       ├── __init__.py
│       ├── logger.py        # 日志工具
│       └── gpu_utils.py     # GPU工具
├── env/                     # 环境配置文件
├── install_all.sh          # Swift环境安装脚本
├── start.py                # 启动脚本
├── docker-compose.yml      # Docker编排配置
├── Dockerfile              # Docker镜像配置
```

## 🎯 GPU排队功能

### 核心特性

1. **自动GPU分配**: 无需手动指定GPU ID，系统自动选择最优GPU
2. **智能排队**: 当GPU不可用时，任务自动加入队列等待
3. **优先级管理**: 支持0-10级优先级，数字越大优先级越高
4. **动态重分配**: 队列中的任务可以重新分配GPU
5. **后台处理**: 自动队列处理器定期检查并启动任务

### 工作流程

```
用户创建任务 → 系统自动分配GPU → 检查GPU可用性
                                    ↓
                              GPU可用？ → 是 → 直接启动任务
                                    ↓ 否
                              加入队列 → 等待GPU可用 → 自动启动
```

## 🧩 多任务类型训练支持

自 v2.0 起，系统支持多种训练任务类型（如多模态模型、语言模型等），通过 `task_type` 字段区分。

- `task_type`: 任务类型，当前支持 `multimodal`（多模态）和 `language_model`（语言模型），后续可扩展。
- `train_params`: 训练参数，结构随任务类型变化，详见下方示例。

### 任务类型与参数模型

| 任务类型         | 说明           | 参数模型（train_params）示例 |
|------------------|----------------|-----------------------------|
| multimodal       | 多模态模型训练 | MultiModalTrainParams       |
| language_model   | 语言模型训练   | LanguageModelTrainParams    |

> 若不指定 `task_type`，默认为 `multimodal`，兼容老接口。

---

## 📚 API文档（多任务类型示例）

### 创建多模态训练任务

```bash
curl -X POST "http://localhost:8000/api/v1/training/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "multimodal",
    "data_path": "AI-ModelScope/coco#20000",
    "model_path": "Qwen/Qwen2.5-VL-7B-Instruct",
    "output_dir": "output/multimodal_001",
    "train_params": {
      "num_epochs": 2,
      "batch_size": 8,
      "vit_lr": 1e-5
    }
  }'
```

### 创建语言模型训练任务

```bash
curl -X POST "http://localhost:8000/api/v1/training/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "language_model",
    "data_path": "AI-ModelScope/text#10000",
    "model_path": "Qwen/Qwen2.5-7B",
    "output_dir": "output/lm_001",
    "train_params": {
      "num_epochs": 3,
      "batch_size": 4,
      "learning_rate": 0.0001
    }
  }'
```

---

## 📚 API文档

### 核心端点

| 方法 | 端点 | 描述 | 状态码 |
|------|------|------|--------|
| `POST` | `/api/v1/training/jobs` | 创建训练任务 | 201 |
| `POST` | `/api/v1/training/jobs/{job_id}/start` | 启动训练任务 | 200 |
| `POST` | `/api/v1/training/jobs/{job_id}/stop` | 停止训练任务 | 200 |
| `POST` | `/api/v1/training/jobs/{job_id}/export` | 手动触发模型导出 | 200 |
| `GET` | `/api/v1/training/jobs/{job_id}/status` | 获取训练状态 | 200 |
| `GET` | `/api/v1/training/jobs/{job_id}` | 获取任务详情 | 200 |
| `GET` | `/api/v1/training/jobs` | 获取任务列表 | 200 |
| `DELETE` | `/api/v1/training/jobs/{job_id}` | 删除训练任务 | 204 |
| `GET` | `/api/v1/training/jobs/{job_id}/logs` | 获取训练日志 | 200 |
| `GET` | `/api/v1/training/jobs/{job_id}/events` | 获取训练事件 | 200 |
| `GET` | `/api/v1/training/gpus` | 获取GPU信息 | 200 |
| `GET` | `/api/v1/training/system/status` | 获取系统状态 | 200 |
| `GET` | `/api/v1/training/health` | 健康检查 | 200 |

### GPU队列管理端点

| 方法 | 端点 | 描述 | 状态码 |
|------|------|------|--------|
| `GET` | `/api/v1/training/queue` | 获取GPU队列状态 | 200 |
| `POST` | `/api/v1/training/queue/process` | 手动处理队列 | 200 |
| `DELETE` | `/api/v1/training/queue/{job_id}` | 从队列移除任务 | 200 |
| `GET` | `/api/v1/training/queue/{job_id}/status` | 获取任务队列状态 | 200 |
| `POST` | `/api/v1/training/queue/processor/start` | 启动队列处理器 | 200 |
| `POST` | `/api/v1/training/queue/processor/stop` | 停止队列处理器 | 200 |
| `GET` | `/api/v1/training/queue/processor/status` | 获取处理器状态 | 200 |

## ⚙️ 训练参数说明（新版）

- 训练参数通过 `train_params` 字段传递，结构随 `task_type` 变化。
- 典型参数如下：

### MultiModalTrainParams
```json
{
  "num_epochs": 1,
  "batch_size": 1,
  "learning_rate": 0.0001,
  "vit_lr": 0.00001,
  "aligner_lr": 0.00001,
  "lora_rank": 16,
  "lora_alpha": 32,
  "gradient_accumulation_steps": 4,
  "eval_steps": 100,
  "save_steps": 100,
  "save_total_limit": 2,
  "logging_steps": 5,
  "max_length": 8192,
  "warmup_ratio": 0.05,
  "dataloader_num_workers": 4,
  "dataset_num_proc": 4,
  "save_only_model": true,
  "train_type": "lora",
  "torch_dtype": "bfloat16"
}
```

### LanguageModelTrainParams
```json
{
  "num_epochs": 1,
  "batch_size": 1,
  "learning_rate": 0.0001,
  "gradient_accumulation_steps": 4,
  "eval_steps": 100,
  "save_steps": 100,
  "save_total_limit": 2,
  "logging_steps": 5,
  "max_length": 2048,
  "warmup_ratio": 0.05,
  "dataloader_num_workers": 4,
  "dataset_num_proc": 4,
  "save_only_model": true,
  "train_type": "standard",
  "torch_dtype": "bfloat16"
}
```

> 你可以根据实际需求，仅传递需要覆盖的参数，未传递的参数将使用默认值。

## ⚙️ 配置说明

### 环境变量配置

创建 `env/.env.dev` 文件：

```env
# 环境配置
ENVIRONMENT=dev

# Redis配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# 日志配置
LOG_LEVEL=INFO
LOG_DIR=logs

# 应用配置
API_PREFIX=/api/v1
APP_HOST=0.0.0.0
APP_PORT=8000

```

### 主要配置项说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `REDIS_HOST` | Redis服务器地址 | localhost |
| `REDIS_PORT` | Redis端口 | 6379 |
| `LOG_LEVEL` | 日志级别 | INFO |

## ⚡ 快速开始

### Docker部署 (推荐)

```bash
# 1. 克隆项目
git clone <repository-url>
cd swift-api

# 2. 构建并启动服务
docker-compose up -d

# 3. 查看服务状态
docker-compose ps

# 4. 访问API文档
# 打开浏览器访问：http://localhost:8000/docs
```

## 💡 使用示例

### 创建训练任务

#### 基本用法（自动GPU分配）

```bash
# 创建训练任务 - 系统自动分配GPU
curl -X POST "http://localhost:8000/api/v1/training/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "data_path": "AI-ModelScope/coco#20000",
    "model_path": "Qwen/Qwen2.5-VL-7B-Instruct",
    "output_dir": "output/training_001"
  }'
```

**响应示例:**

```json
// GPU可用时 - 直接创建
{
    "job_id": "training_001",
    "status": "pending",
    "message": "训练任务创建成功"
}

// GPU不可用时 - 加入队列
{
    "job_id": "training_001",
    "status": "queued",
    "message": "训练任务已创建并加入GPU队列",
    "queue_position": 2,
    "estimated_wait_time": "根据队列位置和GPU使用情况估算"
}
```

#### 高级用法（设置优先级）

```bash
# 高优先级任务
curl -X POST "http://localhost:8000/api/v1/training/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "data_path": "AI-ModelScope/coco#20000",
    "model_path": "Qwen/Qwen2.5-VL-7B-Instruct",
    "output_dir": "output/urgent_training",
    "priority": 9
  }'

# 低优先级任务
curl -X POST "http://localhost:8000/api/v1/training/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "data_path": "AI-ModelScope/coco#20000",
    "model_path": "Qwen/Qwen2.5-VL-7B-Instruct",
    "output_dir": "output/low_priority_training",
    "priority": 1
  }'
```

### 多任务类型创建示例

```bash
# 创建多模态训练任务
curl -X POST "http://localhost:8000/api/v1/training/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "multimodal",
    "data_path": "AI-ModelScope/coco#20000",
    "model_path": "Qwen/Qwen2.5-VL-7B-Instruct",
    "output_dir": "output/multimodal_001",
    "train_params": {
      "num_epochs": 2,
      "batch_size": 8,
      "vit_lr": 1e-5
    }
  }'

# 创建语言模型训练任务
curl -X POST "http://localhost:8000/api/v1/training/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "language_model",
    "data_path": "AI-ModelScope/text#10000",
    "model_path": "Qwen/Qwen2.5-7B",
    "output_dir": "output/lm_001",
    "train_params": {
      "num_epochs": 3,
      "batch_size": 4,
      "learning_rate": 0.0001
    }
  }'
```

### GPU队列管理

```bash
# 查看队列状态
curl -X GET "http://localhost:8000/api/v1/training/queue"

# 手动处理队列
curl -X POST "http://localhost:8000/api/v1/training/queue/process"

# 查看特定任务在队列中的状态
curl -X GET "http://localhost:8000/api/v1/training/queue/training_001/status"

# 从队列中移除任务
curl -X DELETE "http://localhost:8000/api/v1/training/queue/training_001"

# 启动队列处理器
curl -X POST "http://localhost:8000/api/v1/training/queue/processor/start"

# 查看处理器状态
curl -X GET "http://localhost:8000/api/v1/training/queue/processor/status"
```

### 完整的训练工作流

```bash
# 1. 创建训练任务
curl -X POST "http://localhost:8000/api/v1/training/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "data_path": "AI-ModelScope/coco#20000",
    "model_path": "Qwen/Qwen2.5-VL-7B-Instruct",
    "output_dir": "output/training_001"
  }'

# 2. 启动训练任务（如果任务在队列中，会自动启动）
curl -X POST "http://localhost:8000/api/v1/training/jobs/training_001/start"

# 3. 监控训练进度
curl -X GET "http://localhost:8000/api/v1/training/jobs/training_001/progress"

# 4. 查看训练日志
curl -X GET "http://localhost:8000/api/v1/training/jobs/training_001/logs?limit=50"

# 5. 训练完成后，手动导出模型（如果需要）
curl -X POST "http://localhost:8000/api/v1/training/jobs/training_001/export"

# 6. 停止训练任务（如果需要）
curl -X POST "http://localhost:8000/api/v1/training/jobs/training_001/stop"
```

### 系统监控

```bash
# 获取GPU信息
curl -X GET "http://localhost:8000/api/v1/training/gpus"

# 获取系统状态
curl -X GET "http://localhost:8000/api/v1/training/system/status"

# 健康检查
curl -X GET "http://localhost:8000/api/v1/training/health"
```

### 任务管理

```bash
# 获取所有任务列表
curl -X GET "http://localhost:8000/api/v1/training/jobs"

# 获取特定任务详情
curl -X GET "http://localhost:8000/api/v1/training/jobs/training_001"

# 获取训练事件历史
curl -X GET "http://localhost:8000/api/v1/training/jobs/training_001/events"

# 删除训练任务
curl -X DELETE "http://localhost:8000/api/v1/training/jobs/training_001"
```

## 🎯 优先级使用指南

### 优先级说明

| 优先级 | 适用场景 | 示例 |
|--------|----------|------|
| 8-10 | 紧急任务、生产环境 | 线上模型更新、紧急修复 |
| 5-7 | 重要任务 | 重要实验、关键验证 |
| 2-4 | 普通任务 | 日常训练、测试 |
| 0-1 | 低优先级任务 | 实验性训练、调试 |

### 优先级策略

- **数字越大，优先级越高**（0-10，10最高）
- 高优先级的任务会排在队列前面
- 同优先级按创建时间排序（FIFO）
- 当GPU可用时，优先启动高优先级的任务

### 动态调整优先级

```bash
# 1. 查看当前队列状态
curl -X GET "http://localhost:8000/api/v1/training/queue"

# 2. 从队列中移除任务
curl -X DELETE "http://localhost:8000/api/v1/training/queue/training_001"

# 3. 重新创建任务（使用新优先级）
curl -X POST "http://localhost:8000/api/v1/training/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "data_path": "AI-ModelScope/coco#20000",
    "model_path": "Qwen/Qwen2.5-VL-7B-Instruct",
    "output_dir": "output/training_001",
    "priority": 9
  }'
```

## 🔧 开发指南

### 项目依赖

主要依赖包：

```toml
[project]
dependencies = [
    "fastapi[all]>=0.115.14",
    "httpx>=0.28.1",
    "loguru>=0.7.3",
    "psutil>=7.0.0",
    "pydantic>=2.11.7",
    "pydantic-settings>=2.10.1",
    "redis>=6.2.0",
    "requests>=2.32.4",
    "uvicorn[standard]>=0.34.3",
]
```