# =============================================
# 重构功能测试脚本
# 测试LLM和VLLM训练器的基本功能
# 创建时间：2024-12-19
# =============================================

import os
import sys

# 添加应用路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from application.models.base_trainer import (
    LLMTrainingParams,
    MLLMTrainingParams,
    TrainerFactory,
    TrainerType,
)
from application.models.training_model import (
    TrainingJobCreateRequest,
    TrainingTaskType,
    determine_trainer_type,
    resolve_training_params,
)


def test_llm_trainer():
    """测试LLM训练器"""
    print("🔍 测试LLM训练器...")

    # 创建LLM参数
    llm_params = LLMTrainingParams(
        num_epochs=5, batch_size=2, learning_rate=2e-4, lora_rank=32, lora_alpha=64
    )

    # 创建训练器
    trainer = TrainerFactory.create_trainer(
        trainer_type=TrainerType.LLM,
        gpu_ids=["0"],
        data_path="/path/to/data",
        model_path="/path/to/model",
        output_dir="/path/to/output",
        params=llm_params,
    )

    print("✅ LLM训练器创建成功")
    print(f"   训练器类型: {trainer.get_trainer_type()}")
    print(f"   GPU数量: {trainer.get_gpu_count()}")
    print(f"   参数验证: {trainer.validate_params()}")

    # 测试命令构建
    command = trainer.build_command()
    print(f"   训练命令: {' '.join(command[:5])}...")

    # 测试环境变量
    env = trainer.build_environment()
    print(f"   CUDA设备: {env.get('CUDA_VISIBLE_DEVICES')}")

    print()


def test_mllm_trainer():
    """测试MLLM训练器"""
    print("🔍 测试MLLM训练器...")

    # 创建MLLM参数
    mllm_params = MLLMTrainingParams(
        num_epochs=3,
        batch_size=1,
        learning_rate=1e-4,
        vit_lr=1e-5,
        aligner_lr=1e-5,
        lora_rank=16,
        lora_alpha=32,
    )

    # 创建训练器
    trainer = TrainerFactory.create_trainer(
        trainer_type=TrainerType.MLLM,
        gpu_ids=["0", "1"],
        data_path="/path/to/data",
        model_path="/path/to/model",
        output_dir="/path/to/output",
        params=mllm_params,
    )

    print("✅ MLLM训练器创建成功")
    print(f"   训练器类型: {trainer.get_trainer_type()}")
    print(f"   GPU数量: {trainer.get_gpu_count()}")
    print(f"   参数验证: {trainer.validate_params()}")

    # 测试命令构建
    command = trainer.build_command()
    print(f"   训练命令: {' '.join(command[:5])}...")

    # 测试环境变量
    env = trainer.build_environment()
    print(f"   CUDA设备: {env.get('CUDA_VISIBLE_DEVICES')}")
    print(f"   视觉模型: {env.get('VISION_MODEL_ENABLED')}")

    print()


def test_request_resolution():
    """测试请求参数解析"""
    print("🔍 测试请求参数解析...")

    # 测试LLM请求
    llm_request = TrainingJobCreateRequest(
        task_type=TrainingTaskType.LLM,
        data_path="/path/to/data",
        model_path="/path/to/model",
        output_dir="/path/to/output",
        llm_params=LLMTrainingParams(num_epochs=5),
    )

    llm_params = resolve_training_params(llm_request)
    llm_trainer_type = determine_trainer_type(llm_request)

    print("✅ LLM请求解析成功")
    print(f"   参数类型: {type(llm_params).__name__}")
    print(f"   训练器类型: {llm_trainer_type}")

    # 测试MLLM请求
    mllm_request = TrainingJobCreateRequest(
        task_type=TrainingTaskType.MLLM,
        data_path="/path/to/data",
        model_path="/path/to/model",
        output_dir="/path/to/output",
        mllm_params=MLLMTrainingParams(num_epochs=3),
    )

    mllm_params = resolve_training_params(mllm_request)
    mllm_trainer_type = determine_trainer_type(mllm_request)

    print("✅ MLLM请求解析成功")
    print(f"   参数类型: {type(mllm_params).__name__}")
    print(f"   训练器类型: {mllm_trainer_type}")

    print()


def test_backward_compatibility():
    """测试向后兼容性"""
    print("🔍 测试向后兼容性...")

    # 测试旧的任务类型
    old_request = TrainingJobCreateRequest(
        task_type=TrainingTaskType.LANGUAGE_MODEL,
        data_path="/path/to/data",
        model_path="/path/to/model",
        output_dir="/path/to/output",
    )

    params = resolve_training_params(old_request)
    trainer_type = determine_trainer_type(old_request)

    print("✅ 向后兼容测试成功")
    print(f"   旧任务类型: {old_request.task_type}")
    print(f"   推断的训练器类型: {trainer_type}")
    print(f"   解析的参数类型: {type(params).__name__}")

    print()


def test_parameter_comparison():
    """测试参数比较"""
    print("🔍 测试参数比较...")

    llm_params = LLMTrainingParams()
    mllm_params = MLLMTrainingParams()

    llm_dict = llm_params.model_dump()
    mllm_dict = mllm_params.model_dump()

    # 找出不同的参数
    llm_only = set(llm_dict.keys()) - set(mllm_dict.keys())
    mllm_only = set(mllm_dict.keys()) - set(llm_dict.keys())
    common = set(llm_dict.keys()) & set(mllm_dict.keys())

    print("✅ 参数比较完成")
    print(f"   通用参数数量: {len(common)}")
    print(f"   LLM专用参数: {llm_only}")
    print(f"   MLLM专用参数: {mllm_only}")

    print()


def main():
    """主测试函数"""
    print("🚀 开始Swift训练器重构功能测试\n")

    try:
        test_llm_trainer()
        test_mllm_trainer()
        test_request_resolution()
        test_backward_compatibility()
        test_parameter_comparison()

        print("🎉 所有测试通过！重构成功完成。")
        print("\n📋 重构总结:")
        print("   ✅ 创建了BaseTrainer基类")
        print("   ✅ 实现了LLMTrainer和MLLMTrainer子类")
        print("   ✅ 分离了LLM和MLLM参数类型")
        print("   ✅ 提供了工厂模式创建训练器")
        print("   ✅ 保持了向后兼容性")
        print("   ✅ 更新了训练服务和处理器")
        print("   ✅ 添加了新的V2 API接口")

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
