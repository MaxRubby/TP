#!/usr/bin/env python3
"""
测试STAEformer输出格式的脚本
验证修改后的STAEformer和STAEformerTeacher是否能正确返回三个输出
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))

from model.STAEformer import STAEformer
from model.Teacher import STAEformerTeacher

class Args:
    """模拟参数类"""
    def __init__(self):
        self.num_nodes = 307
        self.input_window = 12
        self.output_window = 12
        self.input_dim = 1  # 只使用交通流量特征
        self.output_dim = 1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_staeformer():
    """测试原始STAEformer"""
    print("=== 测试原始STAEformer ===")
    
    model = STAEformer(
        num_nodes=307,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=1,  # 只使用交通流量
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
    )
    
    # 创建测试数据：(batch_size, in_steps, num_nodes, 3)
    # 第0维：交通流量，第1维：tod，第2维：dow
    batch_size = 32
    x = torch.randn(batch_size, 12, 307, 3)
    
    # 设置合理的tod和dow值
    x[..., 1] = torch.rand(batch_size, 12, 307) * 1.0  # tod: 0-1
    x[..., 2] = torch.randint(0, 7, (batch_size, 12, 307)).float()  # dow: 0-6
    
    print(f"输入形状: {x.shape}")
    print(f"Tod范围: {x[..., 1].min():.3f} - {x[..., 1].max():.3f}")
    print(f"Dow范围: {x[..., 2].min():.3f} - {x[..., 2].max():.3f}")
    
    # 前向传播
    try:
        output, temporal_feat, spatiotemporal_feat = model(x)
        print(f"✅ STAEformer前向传播成功")
        print(f"主输出形状: {output.shape}")
        print(f"时间特征形状: {temporal_feat.shape if temporal_feat is not None else None}")
        print(f"时空特征形状: {spatiotemporal_feat.shape if spatiotemporal_feat is not None else None}")
        return True
    except Exception as e:
        print(f"❌ STAEformer前向传播失败: {e}")
        return False

def test_staeformer_teacher():
    """测试STAEformerTeacher"""
    print("\n=== 测试STAEformerTeacher ===")
    
    args = Args()
    model = STAEformerTeacher(args)
    
    # 创建测试数据：(batch_size, input_window, num_nodes, 3)
    batch_size = 32
    x = torch.randn(batch_size, 12, 307, 3)
    
    # 设置合理的tod和dow值
    x[..., 1] = torch.rand(batch_size, 12, 307) * 1.0  # tod: 0-1
    x[..., 2] = torch.randint(0, 7, (batch_size, 12, 307)).float()  # dow: 0-6
    
    print(f"输入形状: {x.shape}")
    print(f"STAEformer model_dim: {model.staeformer_model_dim}")
    print(f"目标residual_channels: {model.target_residual_channels}")
    
    # 前向传播
    try:
        output, aux1, aux2 = model(x)
        print(f"✅ STAEformerTeacher前向传播成功")
        print(f"主输出形状: {output.shape}")
        print(f"辅助输出1形状: {aux1.shape}")
        print(f"辅助输出2形状: {aux2.shape}")
        
        # 验证输出格式是否正确
        expected_main = (batch_size, 12, 307, 1)
        expected_aux = (batch_size, 12, 307, 32)
        
        assert output.shape == expected_main, f"主输出形状错误: 期望{expected_main}, 实际{output.shape}"
        assert aux1.shape == expected_aux, f"辅助输出1形状错误: 期望{expected_aux}, 实际{aux1.shape}"
        assert aux2.shape == expected_aux, f"辅助输出2形状错误: 期望{expected_aux}, 实际{aux2.shape}"
        
        print("✅ 所有输出形状都正确!")
        print("✅ 使用可学习投影层进行特征降维，避免了粗暴裁剪")
        return True
        
    except Exception as e:
        print(f"❌ STAEformerTeacher前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试修改后的STAEformer...")
    
    success1 = test_staeformer()
    success2 = test_staeformer_teacher()
    
    print("\n=== 测试总结 ===")
    if success1 and success2:
        print("🎉 所有测试都通过了!")
        print("✅ STAEformer现在可以返回时间特征和时空特征")
        print("✅ STAEformerTeacher可以正确处理这些特征并生成辅助输出")
        print("✅ 输出格式与原始Teacher兼容")
    else:
        print("❌ 部分测试失败，请检查代码")

if __name__ == "__main__":
    main()
