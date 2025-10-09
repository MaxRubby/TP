#!/usr/bin/env python3
"""
数据检查脚本
用于检查可视化数据的具体结构和内容
"""

import os
import pickle
import numpy as np
from pathlib import Path
import sys

def safe_tensor_info(obj):
    """安全地获取tensor信息，不依赖torch"""
    if hasattr(obj, 'shape'):
        return f"Shape: {obj.shape}, Type: {type(obj).__name__}"
    elif isinstance(obj, (list, tuple)):
        return f"Length: {len(obj)}, Type: {type(obj).__name__}, First item type: {type(obj[0]).__name__ if obj else 'Empty'}"
    elif isinstance(obj, dict):
        return f"Dict with keys: {list(obj.keys())}"
    elif isinstance(obj, np.ndarray):
        return f"NumPy array - Shape: {obj.shape}, dtype: {obj.dtype}"
    else:
        return f"Type: {type(obj).__name__}, Value: {str(obj)[:100]}..."

def inspect_data_structure(data, prefix="", max_depth=3, current_depth=0):
    """递归检查数据结构"""
    if current_depth >= max_depth:
        return [f"{prefix}... (max depth reached)"]
    
    result = []
    
    if isinstance(data, dict):
        result.append(f"{prefix}Dict with {len(data)} keys:")
        for key, value in data.items():
            result.append(f"{prefix}  {key}: {safe_tensor_info(value)}")
            if isinstance(value, (dict, list)) and current_depth < max_depth - 1:
                result.extend(inspect_data_structure(value, prefix + "    ", max_depth, current_depth + 1))
    
    elif isinstance(data, (list, tuple)):
        result.append(f"{prefix}{type(data).__name__} with {len(data)} items:")
        if data:
            for i, item in enumerate(data[:3]):  # 只显示前3个项目
                result.append(f"{prefix}  [{i}]: {safe_tensor_info(item)}")
                if isinstance(item, (dict, list)) and current_depth < max_depth - 1:
                    result.extend(inspect_data_structure(item, prefix + "    ", max_depth, current_depth + 1))
            if len(data) > 3:
                result.append(f"{prefix}  ... and {len(data) - 3} more items")
    
    else:
        result.append(f"{prefix}{safe_tensor_info(data)}")
    
    return result

def inspect_pkl_file(file_path, detailed=True):
    """检查单个pkl文件"""
    print(f"\n{'='*60}")
    print(f"检查文件: {file_path}")
    print(f"文件大小: {os.path.getsize(file_path) / 1024 / 1024:.1f} MB")
    print(f"{'='*60}")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"数据类型: {type(data)}")
        
        if detailed:
            structure_info = inspect_data_structure(data, "", max_depth=4)
            for line in structure_info:
                print(line)
        else:
            print(f"基本信息: {safe_tensor_info(data)}")
            
        return data
    
    except Exception as e:
        print(f"加载失败: {e}")
        return None

def main():
    """主函数"""
    data_dir = Path("TP/model/SAVE/train/MTGNN/PEMS04/visualization_data")
    
    if not data_dir.exists():
        print(f"数据目录不存在: {data_dir}")
        return
    
    pkl_files = sorted(list(data_dir.glob('*.pkl')))
    print(f"找到 {len(pkl_files)} 个pkl文件")
    
    if not pkl_files:
        print("没有找到pkl文件")
        return
    
    # 检查前3个文件的详细结构
    print("\n" + "="*80)
    print("详细检查前3个文件:")
    print("="*80)
    
    sample_files = pkl_files[:3]
    all_data = {}
    
    for file_path in sample_files:
        data = inspect_pkl_file(file_path, detailed=True)
        if data is not None:
            all_data[file_path.stem] = data
    
    # 快速检查所有文件的基本信息
    print(f"\n" + "="*80)
    print("所有文件的基本信息:")
    print("="*80)
    
    for file_path in pkl_files:
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            epoch_num = int(file_path.stem.split('_')[1])
            print(f"Epoch {epoch_num:2d}: {safe_tensor_info(data)}")
            
            # 统计关键字段
            if isinstance(data, dict):
                key_summary = []
                for key in data.keys():
                    if data[key] is not None:
                        key_summary.append(key)
                print(f"          可用字段: {key_summary}")
                
        except Exception as e:
            print(f"{file_path.name}: 加载失败 - {e}")
    
    # 分析数据一致性
    print(f"\n" + "="*80)
    print("数据一致性分析:")
    print("="*80)
    
    if all_data:
        # 获取所有可能的键
        all_keys = set()
        for data in all_data.values():
            if isinstance(data, dict):
                all_keys.update(data.keys())
        
        print(f"所有可能的数据字段: {sorted(all_keys)}")
        
        # 检查每个字段在不同文件中的存在情况
        for key in sorted(all_keys):
            availability = []
            for file_name, data in all_data.items():
                if isinstance(data, dict) and key in data and data[key] is not None:
                    availability.append("✓")
                else:
                    availability.append("✗")
            print(f"字段 '{key}': {' '.join(availability)} ({' '.join(all_data.keys())})")

if __name__ == "__main__":
    main() 