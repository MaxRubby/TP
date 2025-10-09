#!/usr/bin/env python3
"""
比较不同损失函数的数值范围
"""

import torch
import torch.nn as nn
import numpy as np

def compare_loss_functions():
    """比较不同损失函数在相同预测误差下的数值"""
    
    # 模拟PEMS04数据范围的预测和真实值
    # 真实值范围：0-836，均值189
    true_values = torch.tensor([100.0, 200.0, 300.0, 150.0, 250.0])
    
    # 模拟不同程度的预测误差
    pred_values_good = true_values + torch.tensor([5.0, -3.0, 2.0, -1.0, 4.0])  # 好预测
    pred_values_bad = true_values + torch.tensor([50.0, -30.0, 20.0, -10.0, 40.0])  # 差预测
    
    # 定义损失函数
    mae_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    huber_loss = nn.HuberLoss(delta=1.0)
    
    print("=== 损失函数数值对比 ===")
    print(f"真实值: {true_values.numpy()}")
    print()
    
    # 好预测的情况
    print("📈 好预测（误差±1-5）:")
    print(f"预测值: {pred_values_good.numpy()}")
    print(f"MAE Loss: {mae_loss(pred_values_good, true_values).item():.6f}")
    print(f"MSE Loss: {mse_loss(pred_values_good, true_values).item():.6f}")
    print(f"Huber Loss: {huber_loss(pred_values_good, true_values).item():.6f}")
    print()
    
    # 差预测的情况
    print("📉 差预测（误差±10-50）:")
    print(f"预测值: {pred_values_bad.numpy()}")
    print(f"MAE Loss: {mae_loss(pred_values_bad, true_values).item():.6f}")
    print(f"MSE Loss: {mse_loss(pred_values_bad, true_values).item():.6f}")
    print(f"Huber Loss: {huber_loss(pred_values_bad, true_values).item():.6f}")
    print()
    
    # 分析Huber Loss的特性
    print("🔍 Huber Loss特性分析:")
    errors = torch.abs(pred_values_good - true_values)
    print(f"绝对误差: {errors.numpy()}")
    print("对于误差 < δ(1.0)的情况，Huber使用平方损失")
    print("对于误差 ≥ δ(1.0)的情况，Huber使用线性损失")
    
    # 手动计算Huber Loss验证
    delta = 1.0
    manual_huber = 0
    for error in errors:
        if error <= delta:
            manual_huber += 0.5 * error**2
        else:
            manual_huber += delta * (error - 0.5 * delta)
    manual_huber /= len(errors)
    print(f"手动计算Huber Loss: {manual_huber:.6f}")

def analyze_validation_loss():
    """分析验证损失的合理性"""
    print("\n=== 验证损失分析 ===")
    print("Huber Loss验证损失 ≈ 0.64")
    print("这意味着:")
    print("1. 大部分预测误差 < 1.0（使用平方损失）")
    print("2. 或者误差虽然大，但Huber的线性增长使数值较小")
    print()
    print("要判断模型性能，应该看:")
    print("1. MAE指标（平均绝对误差）")
    print("2. RMSE指标（均方根误差）")
    print("3. MAPE指标（平均绝对百分比误差）")
    print()
    print("💡 建议：使用相同的评估指标比较不同损失函数训练的模型")

if __name__ == "__main__":
    compare_loss_functions()
    analyze_validation_loss()



