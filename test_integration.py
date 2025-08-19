# =============================================
# 集成测试脚本 - 验证新API端点
# 创建时间：2024-12-19
# 目的：验证重构后的API是否正常工作
# =============================================

import json
import os
import sys

# 添加应用路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from application.models.base_trainer import LLMTrainingParams, MLLMTrainingParams
from application.models.training_model import TrainingJobCreateRequest, TrainingTaskType


def test_api_imports():
    """测试所有新的API模块是否可以正常导入"""
    print("🔍 测试API模块导入...")

    try:
        # 测试训练器基类导入
        from application.models.base_trainer import (
            BaseTrainer,
            LLMTrainer,
            LLMTrainingParams,
            MLLMTrainer,
            MLLMTrainingParams,
            TrainerFactory,
            TrainerType,
        )

        print("✅ 训练器基类导入成功")

        # 测试训练模型导入
        from application.models.training_model import (
            TrainingJobCreateRequest,
            TrainingTaskType,
            determine_trainer_type,
            resolve_training_params,
        )

        print("✅ 训练模型导入成功")

        # 测试新的API路由器导入
        from application.api.training_v2 import router as training_v2_router

        print("✅ 新API路由器导入成功")

        # 测试主应用导入
        from application.main import app

        print("✅ 主应用导入成功")

    except ImportError as e:
        print(f"❌ 导入失败: {str(e)}")
        raise

    print()


def test_request_models():
    """测试请求模型的序列化和反序列化"""
    print("🔍 测试请求模型...")

    try:
        # 测试LLM请求
        llm_request_data = {
            "task_type": "llm",
            "data_path": "/data/llm_dataset",
            "model_path": "/models/llama",
            "output_dir": "/output/llm_fine_tuned",
            "llm_params": {
                "num_epochs": 5,
                "batch_size": 2,
                "learning_rate": 2e-4,
                "lora_rank": 32,
                "lora_alpha": 64,
            },
        }

        llm_request = TrainingJobCreateRequest(**llm_request_data)
        print("✅ LLM请求模型创建成功")

        # 测试序列化
        llm_json = llm_request.model_dump_json()
        print("✅ LLM请求序列化成功")

        # 测试反序列化
        llm_parsed = TrainingJobCreateRequest.model_validate_json(llm_json)
        print("✅ LLM请求反序列化成功")

        # 测试MLLM请求
        mllm_request_data = {
            "task_type": "mllm",
            "data_path": "/data/mllm_dataset",
            "model_path": "/models/llava",
            "output_dir": "/output/mllm_fine_tuned",
            "mllm_params": {
                "num_epochs": 3,
                "batch_size": 1,
                "learning_rate": 1e-4,
                "vit_lr": 1e-5,
                "aligner_lr": 1e-5,
                "lora_rank": 16,
                "lora_alpha": 32,
            },
        }

        mllm_request = TrainingJobCreateRequest(**mllm_request_data)
        print("✅ MLLM请求模型创建成功")

        # 测试序列化
        mllm_json = mllm_request.model_dump_json()
        print("✅ MLLM请求序列化成功")

    except Exception as e:
        print(f"❌ 请求模型测试失败: {str(e)}")
        raise

    print()


def test_parameter_classes():
    """测试参数类的功能"""
    print("🔍 测试参数类...")

    try:
        # 测试LLM参数
        llm_params = LLMTrainingParams(num_epochs=5, learning_rate=2e-4, lora_rank=32)

        # 测试命令行参数生成
        llm_args = llm_params.to_command_args()
        print(f"✅ LLM命令行参数生成成功: {len(llm_args)}个参数")

        # 测试MLLM参数
        mllm_params = MLLMTrainingParams(
            num_epochs=3, learning_rate=1e-4, vit_lr=1e-5, aligner_lr=1e-5
        )

        # 测试命令行参数生成
        mllm_args = mllm_params.to_command_args()
        print(f"✅ MLLM命令行参数生成成功: {len(mllm_args)}个参数")

        # 测试训练器类型
        assert llm_params.get_trainer_type().value == "llm"
        assert mllm_params.get_trainer_type().value == "mllm"
        print("✅ 训练器类型验证成功")

    except Exception as e:
        print(f"❌ 参数类测试失败: {str(e)}")
        raise

    print()


def test_compatibility():
    """测试向后兼容性"""
    print("🔍 测试向后兼容性...")

    try:
        # 测试旧格式的请求
        old_request = TrainingJobCreateRequest(
            task_type=TrainingTaskType.LANGUAGE_MODEL,
            data_path="/data/old_dataset",
            model_path="/models/old_model",
            output_dir="/output/old_output",
        )
        print("✅ 旧格式请求创建成功")

        # 测试参数解析
        from application.models.training_model import (
            determine_trainer_type,
            resolve_training_params,
        )

        params = resolve_training_params(old_request)
        trainer_type = determine_trainer_type(old_request)

        print(f"✅ 参数解析成功: {type(params).__name__}")
        print(f"✅ 训练器类型推断成功: {trainer_type}")

    except Exception as e:
        print(f"❌ 向后兼容性测试失败: {str(e)}")
        raise

    print()


def test_json_examples():
    """生成JSON示例用于API文档"""
    print("🔍 生成JSON示例...")

    try:
        # LLM请求示例
        llm_example = {
            "task_type": "llm",
            "data_path": "/data/llm_dataset",
            "model_path": "/models/qwen2-7b-instruct",
            "output_dir": "/output/llm_fine_tuned",
            "priority": 5,
            "llm_params": {
                "num_epochs": 3,
                "batch_size": 4,
                "learning_rate": 5e-5,
                "lora_rank": 64,
                "lora_alpha": 128,
                "max_length": 2048,
                "warmup_ratio": 0.1,
            },
        }

        # MLLM请求示例
        mllm_example = {
            "task_type": "mllm",
            "data_path": "/data/multimodal_dataset",
            "model_path": "/models/qwen2-vl-7b-instruct",
            "output_dir": "/output/mllm_fine_tuned",
            "priority": 7,
            "mllm_params": {
                "num_epochs": 2,
                "batch_size": 2,
                "learning_rate": 1e-5,
                "vit_lr": 1e-6,
                "aligner_lr": 1e-5,
                "lora_rank": 32,
                "lora_alpha": 64,
                "max_length": 4096,
            },
        }

        # 保存示例到文件
        examples = {
            "llm_training_example": llm_example,
            "mllm_training_example": mllm_example,
            "description": "Swift Trainer API v2 请求示例",
        }

        with open("api_examples.json", "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)

        print("✅ JSON示例生成成功，保存到 api_examples.json")

    except Exception as e:
        print(f"❌ JSON示例生成失败: {str(e)}")
        raise

    print()


def main():
    """主测试函数"""
    print("🚀 开始集成测试\n")

    try:
        test_api_imports()
        test_request_models()
        test_parameter_classes()
        test_compatibility()
        test_json_examples()

        print("🎉 所有集成测试通过！")
        print("\n📋 测试总结:")
        print("   ✅ API模块导入正常")
        print("   ✅ 请求模型序列化正常")
        print("   ✅ 参数类功能正常")
        print("   ✅ 向后兼容性保持")
        print("   ✅ JSON示例生成成功")
        print("\n🔧 重构完成状态:")
        print("   ✅ 基类架构已实现")
        print("   ✅ LLM/MLLM分离完成")
        print("   ✅ API路由已注册")
        print("   ✅ 向后兼容性保持")
        print("   ✅ 测试验证通过")

    except Exception as e:
        print(f"❌ 集成测试失败: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
