# Swift Trainer API

基于FastAPI的Swift训练任务管理API系统，支持多GPU训练、Redis状态管理和详细日志记录。

## 功能特性

- 🚀 **FastAPI接口**：RESTful API管理训练任务
- 🔄 **Redis状态管理**：实时记录训练状态和进度
- 📝 **详细日志系统**：多级别日志记录和查询
- ⚡ **异步任务处理**：支持并发训练任务
- 🎯 **GPU资源管理**：智能分配GPU资源
- 📊 **任务监控**：实时监控训练进度和状态

## 项目结构

```
swift_trainer_api/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI应用入口
│   ├── config.py            # 配置管理
│   ├── models/              # 数据模型
│   │   ├── __init__.py
│   │   └── training.py      # 训练任务模型
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
├── logs/                    # 日志文件目录
├── output/                  # 训练输出目录
├── requirements.txt         # 依赖包
├── docker-compose.yml       # Docker配置
└── README.md               # 项目文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动Redis服务

```bash
docker-compose up -d redis
```

### 3. 启动API服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 访问API文档

打开浏览器访问：http://localhost:8000/docs

## API使用示例

### 创建训练任务

```bash
curl -X POST "http://localhost:8000/api/v1/training/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "gpu_id": "0",
    "data_path": "/path/to/dataset",
    "model_path": "Qwen/Qwen2.5-VL-7B-Instruct",
    "output_dir": "output",
    "num_epochs": 1,
    "batch_size": 1,
    "learning_rate": 1e-4
  }'
```

### 查询任务状态

```bash
curl -X GET "http://localhost:8000/api/v1/training/jobs/{job_id}/status"
```

### 获取训练日志

```bash
curl -X GET "http://localhost:8000/api/v1/training/jobs/{job_id}/logs"
```

## 配置说明

主要配置项在 `app/config.py` 中：

- `REDIS_HOST`: Redis服务器地址
- `REDIS_PORT`: Redis端口
- `REDIS_DB`: Redis数据库编号
- `LOG_LEVEL`: 日志级别
- `LOG_DIR`: 日志目录
- `OUTPUT_DIR`: 训练输出目录

## 开发说明

### 添加新的训练参数

1. 在 `app/models/training.py` 中添加新的模型字段
2. 在 `app/services/training_service.py` 中更新训练命令生成逻辑
3. 更新API文档和测试用例
