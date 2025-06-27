#!/usr/bin/env python3
"""
LLM训练功能测试脚本
"""

import requests

# API基础URL
BASE_URL = "http://localhost:8000"

def test_create_llm_job() -> str:
    """测试创建LLM训练任务"""
    print("=== 测试创建LLM训练任务 ===")
    
    # LLM训练任务请求数据
    llm_request = {
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
    
    try:
        response = requests.post(
            f"{BASE_URL}/training/llm/jobs",
            json=llm_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            job_id = result["job_id"]
            print(f"✅ LLM训练任务创建成功: {job_id}")
            print(f"   状态: {result['status']}")
            print(f"   消息: {result['message']}")
            return job_id
        else:
            print(f"❌ LLM训练任务创建失败: {response.status_code}")
            print(f"   错误: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ 请求失败: {str(e)}")
        return None

def test_get_llm_jobs():
    """测试获取LLM训练任务列表"""
    print("\n=== 测试获取LLM训练任务列表 ===")
    
    try:
        response = requests.get(f"{BASE_URL}/training/llm/jobs")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 获取LLM训练任务列表成功")
            print(f"   总数: {result['total']}")
            print(f"   当前页: {result['page']}")
            print(f"   每页大小: {result['size']}")
            
            if result['jobs']:
                print("   任务列表:")
                for job in result['jobs'][:3]:  # 只显示前3个
                    print(f"     - {job['id']}: {job['status']} (创建时间: {job['created_at']})")
            else:
                print("   暂无LLM训练任务")
        else:
            print(f"❌ 获取LLM训练任务列表失败: {response.status_code}")
            print(f"   错误: {response.text}")
            
    except Exception as e:
        print(f"❌ 请求失败: {str(e)}")

def test_get_all_jobs():
    """测试获取所有训练任务列表"""
    print("\n=== 测试获取所有训练任务列表 ===")
    
    try:
        response = requests.get(f"{BASE_URL}/training/all/jobs")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 获取所有训练任务列表成功")
            print(f"   总数: {result['total']}")
            print(f"   VLM任务数: {result['total_vlm']}")
            print(f"   LLM任务数: {result['total_llm']}")
            print(f"   当前页: {result['page']}")
            print(f"   每页大小: {result['size']}")
            
            if result['jobs']:
                print("   任务列表:")
                for job in result['jobs'][:5]:  # 只显示前5个
                    training_type = job.get('training_type', 'unknown')
                    print(f"     - {job['id']}: {training_type} - {job['status']} (创建时间: {job['created_at']})")
            else:
                print("   暂无训练任务")
        else:
            print(f"❌ 获取所有训练任务列表失败: {response.status_code}")
            print(f"   错误: {response.text}")
            
    except Exception as e:
        print(f"❌ 请求失败: {str(e)}")

def test_get_job_detail(job_id: str):
    """测试获取训练任务详情"""
    print(f"\n=== 测试获取训练任务详情: {job_id} ===")
    
    try:
        response = requests.get(f"{BASE_URL}/training/jobs/{job_id}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 获取训练任务详情成功")
            print(f"   任务ID: {result['id']}")
            print(f"   训练类型: {result.get('training_type', 'unknown')}")
            print(f"   状态: {result['status']}")
            print(f"   创建时间: {result['created_at']}")
            print(f"   GPU ID: {result['gpu_id']}")
            
            if result.get('training_type') == 'llm':
                print(f"   模型路径: {result['model_path']}")
                print(f"   数据集: {result['datasets']}")
                print(f"   输出目录: {result['output_dir']}")
            else:
                print(f"   模型路径: {result['model_path']}")
                print(f"   数据路径: {result['data_path']}")
                print(f"   输出目录: {result['output_dir']}")
        else:
            print(f"❌ 获取训练任务详情失败: {response.status_code}")
            print(f"   错误: {response.text}")
            
    except Exception as e:
        print(f"❌ 请求失败: {str(e)}")

def test_get_job_status(job_id: str):
    """测试获取训练任务状态"""
    print(f"\n=== 测试获取训练任务状态: {job_id} ===")
    
    try:
        response = requests.get(f"{BASE_URL}/training/jobs/{job_id}/status")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 获取训练任务状态成功")
            print(f"   任务ID: {result['job_id']}")
            print(f"   训练类型: {result.get('training_type', 'unknown')}")
            print(f"   状态: {result['status']}")
            print(f"   进度: {result['progress']}%")
            print(f"   创建时间: {result['created_at']}")
            
            if result.get('started_at'):
                print(f"   开始时间: {result['started_at']}")
            if result.get('completed_at'):
                print(f"   完成时间: {result['completed_at']}")
            if result.get('loss'):
                print(f"   当前损失: {result['loss']}")
            if result.get('gpu_memory_usage'):
                print(f"   GPU内存使用: {result['gpu_memory_usage']}")
        else:
            print(f"❌ 获取训练任务状态失败: {response.status_code}")
            print(f"   错误: {response.text}")
            
    except Exception as e:
        print(f"❌ 请求失败: {str(e)}")

def test_health_check():
    """测试健康检查"""
    print("\n=== 测试健康检查 ===")
    
    try:
        response = requests.get(f"{BASE_URL}/training/health")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 健康检查成功")
            print(f"   状态: {result.get('status', 'unknown')}")
            print(f"   时间: {result.get('timestamp', 'unknown')}")
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            print(f"   错误: {response.text}")
            
    except Exception as e:
        print(f"❌ 请求失败: {str(e)}")

def main():
    """主测试函数"""
    print("🚀 开始LLM训练功能测试")
    print("=" * 50)
    
    # 测试健康检查
    test_health_check()
    
    # 测试获取现有任务列表
    test_get_llm_jobs()
    test_get_all_jobs()
    
    # 测试创建LLM训练任务
    job_id = test_create_llm_job()
    
    if job_id:
        # 测试获取任务详情
        test_get_job_detail(job_id)
        
        # 测试获取任务状态
        test_get_job_status(job_id)
        
        # 再次获取任务列表，确认新任务已创建
        print("\n" + "=" * 50)
        print("🔄 重新获取任务列表，确认新任务已创建")
        test_get_llm_jobs()
        test_get_all_jobs()
    
    print("\n" + "=" * 50)
    print("✅ LLM训练功能测试完成")

if __name__ == "__main__":
    main() 