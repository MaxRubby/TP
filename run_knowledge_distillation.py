#!/usr/bin/env python3
"""
知识蒸馏训练脚本
教师模型：STAEformer
学生模型：STMLP

使用方法：
1. 只训练教师模型：python run_knowledge_distillation.py -t
2. 只训练学生模型（需要预训练的教师模型）：python run_knowledge_distillation.py
3. 完整知识蒸馏流程：python run_knowledge_distillation.py -t

作者：AI Assistant
日期：2025-01-24
"""

import os
import sys
import subprocess

def main():
    print("=" * 60)
    print("知识蒸馏训练脚本")
    print("教师模型：STAEformer")
    print("学生模型：STMLP")
    print("=" * 60)
    
    # 基础参数
    base_cmd = [
        "python", "Mains.py",
        "--dataset", "PEMS04",
        "--device", "cuda:0",
        "--mode", "train",
        "--epochs", "100",
        "--batch_size", "32",
        "--lr_init", "0.001",
        "--early_stop", "True",
        "--early_stop_patience", "15",
        "--use_staeformer_teacher",  # 使用STAEformer作为教师模型
        "--tod", "True",  # 启用time of day特征
    ]
    
    # 检查是否需要训练教师模型
    if "-t" in sys.argv or "--teacher" in sys.argv:
        print("\n🎓 第一阶段：训练教师模型 (STAEformer)")
        teacher_cmd = base_cmd + ["-t"]
        
        print("执行命令:", " ".join(teacher_cmd))
        result = subprocess.run(teacher_cmd, cwd=".")
        
        if result.returncode != 0:
            print("❌ 教师模型训练失败!")
            return 1
        else:
            print("✅ 教师模型训练完成!")
    
    print("\n👨‍🎓 第二阶段：知识蒸馏训练学生模型 (STMLP)")
    student_cmd = base_cmd  # 不包含-t参数
    
    print("执行命令:", " ".join(student_cmd))
    result = subprocess.run(student_cmd, cwd=".")
    
    if result.returncode != 0:
        print("❌ 学生模型训练失败!")
        return 1
    else:
        print("✅ 学生模型训练完成!")
    
    print("\n🎉 知识蒸馏训练流程完成!")
    print("\n模型保存位置:")
    print("- 教师模型: SAVE/train/MTGNN/PEMS04/best_model.pth")
    print("- 学生模型: SAVE/train/MTGNN/PEMS04/best_modelstudent.pth")
    
    return 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='知识蒸馏训练脚本')
    parser.add_argument('-t', '--teacher', action='store_true', 
                       help='训练教师模型 (如果不指定，只训练学生模型)')
    args = parser.parse_args()
    
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 训练过程中出现错误: {e}")
        sys.exit(1)
