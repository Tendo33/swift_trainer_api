# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Swift Trainer API is a comprehensive machine learning training management system built with FastAPI. It manages Swift model training workflows with intelligent GPU resource allocation, queue management, and real-time monitoring.

## Key Architecture Components

### Core Services Architecture
- **TrainingService**: Main business logic for training job management, GPU allocation, and queue processing
- **RedisService**: State management, job persistence, and event logging
- **GPU Manager**: Hardware resource monitoring and allocation
- **TrainingHandler**: Multi-task type processing (multimodal, language_model, deploy)

### Multi-Task Type Support
The system supports three main task types through the `TrainingTaskType` enum:
- `multimodal`: Multi-modal model training with `MultiModalTrainParams`
- `language_model`: Language model training with `LanguageModelTrainParams`  
- `deploy`: Model deployment with `DeployParams`

### GPU Queue Management
Intelligent automatic GPU allocation with priority-based queuing (0-10 priority levels). Tasks automatically join queue when GPUs are unavailable, with background processor handling queue execution.

## Development Commands

### Environment Setup
```bash
# Install Swift dependencies (requires Python 3.10+, CUDA 12.*)
sh install_all.sh

# Install Python dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Development mode
python start.py

# Docker deployment
docker-compose up -d

# Access API documentation at: http://localhost:8000/docs
```

### Configuration
Create `env/.env.dev` for environment-specific settings:
```env
ENVIRONMENT=dev
REDIS_HOST=localhost
REDIS_PORT=6379
LOG_LEVEL=INFO
API_PREFIX=/api/v1
APP_HOST=0.0.0.0
APP_PORT=8000
```

## Critical Implementation Details

### Training Job Lifecycle
1. **Creation**: Auto-allocates GPUs, checks availability, queues if needed
2. **Queue Processing**: Background thread processes priority-based queue
3. **Execution**: Swift training processes spawned via subprocess
4. **Monitoring**: Real-time progress tracking and event logging
5. **Completion**: Model export and cleanup

### GPU Resource Management
- Uses `gpu_utils.py` for NVIDIA GPU monitoring
- Automatic allocation based on availability
- Priority-based queuing with FIFO for same priority
- Dynamic reassignment when GPUs become available

### Redis Data Structure
- **Training Jobs**: Hash with job metadata
- **GPU Status**: Hash for each GPU's current state
- **Queue**: Sorted set with priority scores
- **Events**: Lists for job event history
- **Logs**: Lists for training process logs

### Error Handling Patterns
- All API endpoints include comprehensive exception handling
- Redis operations have fallback strategies
- GPU allocation failures trigger queue management
- Subprocess monitoring for training job failures

## API Endpoint Patterns

### Training Job Management
- `POST /api/v1/training/jobs` - Create with automatic GPU allocation
- `GET /api/v1/training/jobs/{job_id}/progress` - Real-time progress monitoring
- `POST /api/v1/training/jobs/{job_id}/export` - Manual model export trigger

### Queue Management
- `GET /api/v1/training/queue` - Queue status with position estimates
- `POST /api/v1/training/queue/process` - Manual queue processing
- `DELETE /api/v1/training/queue/{job_id}` - Remove from queue

### System Monitoring
- `GET /api/v1/training/gpus` - GPU status and availability
- `GET /api/v1/training/system/status` - System resource metrics
- `GET /api/v1/training/health` - Service health check

## Key Dependencies

### Core Framework
- **FastAPI**: Web framework with automatic API documentation
- **Pydantic**: Data validation and serialization
- **Redis**: State management and job persistence
- **Uvicorn**: ASGI server for production deployment

### ML Infrastructure
- **ms-swift**: Primary training framework (v3.5.0)
- **vLLM**: Inference optimization
- **DeepSpeed**: Distributed training
- **Flash Attention**: Performance optimization

## Testing and Validation

The system includes comprehensive health checks:
- Redis connectivity validation
- GPU manager availability
- Training process monitoring
- Queue processor status

All endpoints return structured error responses with appropriate HTTP status codes and detailed error information.

## Deployment Notes

### Docker Configuration
- Uses host network mode for GPU access
- Volume mounts for data persistence
- NVIDIA runtime for GPU support
- Health checks for Redis dependency

### Environment Requirements
- Python 3.10+
- CUDA 12.*
- NVIDIA GPU with drivers
- Redis server for state management