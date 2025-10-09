#!/usr/bin/env python3
"""
STAEformer 可视化分析脚本
用于分析训练过程中保存的可视化数据
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class VisualizationAnalyzer:
    def __init__(self, data_dir):
        """
        初始化可视化分析器
        
        Args:
            data_dir: 包含pkl文件的目录路径
        """
        self.data_dir = Path(data_dir)
        self.data_files = sorted(list(self.data_dir.glob('*.pkl')))
        self.loaded_data = {}
        self.output_dir = self.data_dir / 'analysis_results'
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"找到 {len(self.data_files)} 个数据文件")
        
    def load_data(self, max_files=None):
        """加载可视化数据"""
        files_to_load = self.data_files[:max_files] if max_files else self.data_files
        
        for file_path in files_to_load:
            try:
                with open(file_path, 'rb') as f:
                    # 添加map_location参数以在CPU上加载CUDA张量
                    data = pickle.load(f)
                    # 将所有张量移动到CPU
                    data = self._move_tensors_to_cpu(data)
                    epoch_info = file_path.stem  # 获取文件名（不含扩展名）
                    self.loaded_data[epoch_info] = data
                    print(f"已加载: {epoch_info}")
            except Exception as e:
                print(f"加载失败 {file_path}: {e}")
        
        print(f"总共加载了 {len(self.loaded_data)} 个文件")
    
    def _move_tensors_to_cpu(self, obj):
        """递归地将所有张量移动到CPU"""
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        elif isinstance(obj, dict):
            return {k: self._move_tensors_to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._move_tensors_to_cpu(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._move_tensors_to_cpu(item) for item in obj)
        else:
            return obj
        
    def analyze_pattern_evolution(self):
        """分析模式权重"""
        print("\n=== 分析模式权重 ===")
        
        # 只选择最新的一个epoch进行分析
        latest_key = sorted(self.loaded_data.keys())[-1] if self.loaded_data else None
        
        if not latest_key:
            print("没有找到模式权重数据")
            return
            
        data = self.loaded_data[latest_key]
        epoch_num = int(latest_key.split('_')[1])
        print(f"分析Epoch {epoch_num}的模式权重")
        
        # 从pattern_decomposition中获取模式权重
        if 'pattern_decomposition' not in data or data['pattern_decomposition'] is None:
            print("当前epoch没有模式权重数据")
            return
            
        pattern_data = data['pattern_decomposition']
        if 'pattern_weights' not in pattern_data or pattern_data['pattern_weights'] is None:
            print("当前epoch没有模式权重数据")
            return
            
        # 转换为numpy数组
        if isinstance(pattern_data['pattern_weights'], torch.Tensor):
            weights = pattern_data['pattern_weights'].detach().cpu().numpy()
        else:
            weights = np.array(pattern_data['pattern_weights'])
            
        if 'pattern_logits' in pattern_data and pattern_data['pattern_logits'] is not None:
            if isinstance(pattern_data['pattern_logits'], torch.Tensor):
                logits = pattern_data['pattern_logits'].detach().cpu().numpy()
            else:
                logits = np.array(pattern_data['pattern_logits'])
        else:
            logits = None
            
        print(f"权重形状: {weights.shape}")
        
        # 绘制模式权重分析图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 模式权重分布
        weights_flat = weights.reshape(-1, weights.shape[-1])
        
        for i in range(weights.shape[-1]):  # 对每个模式
            axes[0, 0].hist(weights_flat[:, i], alpha=0.7, bins=30, 
                           label=f'Pattern {i}', density=True)
        axes[0, 0].set_title(f'Epoch {epoch_num} 模式权重分布')
        axes[0, 0].set_xlabel('权重值')
        axes[0, 0].set_ylabel('密度')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. 模式权重平均值
        mean_weights = np.mean(weights, axis=(0, 1, 2))  # 对batch, time, nodes取平均
        
        bar_positions = np.arange(len(mean_weights))
        axes[0, 1].bar(bar_positions, mean_weights)
        axes[0, 1].set_title(f'Epoch {epoch_num} 模式平均权重')
        axes[0, 1].set_xlabel('模式索引')
        axes[0, 1].set_ylabel('平均权重')
        axes[0, 1].set_xticks(bar_positions)
        axes[0, 1].set_xticklabels([f'P{i}' for i in range(len(mean_weights))])
        axes[0, 1].grid(True)
        
        # 3. 节点间模式权重相关性
        if weights.shape[2] > 1:  # 确保有多个节点
            # 对每个模式，计算节点间权重的相关性
            node_corrs = []
            for pattern_idx in range(weights.shape[-1]):
                pattern_weights = weights[:, :, :, pattern_idx]
                # 重塑为节点×(batch*time)
                reshaped = pattern_weights.transpose(2, 0, 1).reshape(pattern_weights.shape[2], -1)
                corr = np.corrcoef(reshaped)
                node_corrs.append(corr)
            
            # 显示第一个模式的节点相关性
            if node_corrs:
                sns.heatmap(node_corrs[0], cmap='viridis', ax=axes[1, 0])
                axes[1, 0].set_title(f'模式0的节点间权重相关性 (Epoch {epoch_num})')
                axes[1, 0].set_xlabel('节点索引')
                axes[1, 0].set_ylabel('节点索引')
        else:
            axes[1, 0].text(0.5, 0.5, '节点数不足以计算相关性', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # 4. 模式权重的熵（衡量模式分离度）
        # 计算每个样本的权重熵
        weights_normalized = weights / (np.sum(weights, axis=-1, keepdims=True) + 1e-8)
        entropy = -np.sum(weights_normalized * np.log(weights_normalized + 1e-8), axis=-1)
        mean_entropy = np.mean(entropy)
        
        axes[1, 1].hist(entropy.flatten(), bins=30, alpha=0.7)
        axes[1, 1].axvline(mean_entropy, color='r', linestyle='--', 
                          label=f'平均熵: {mean_entropy:.3f}')
        axes[1, 1].set_title(f'模式权重熵分布 (Epoch {epoch_num})')
        axes[1, 1].set_xlabel('熵值')
        axes[1, 1].set_ylabel('频率')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pattern_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"模式权重分析完成，结果保存至: {self.output_dir / 'pattern_analysis.png'}")
        
        return epoch_num, weights
        
    def analyze_embeddings(self):
        """分析嵌入向量"""
        print("\n=== 分析嵌入向量 ===")
        
        # 只选择最新的一个epoch进行分析
        latest_key = sorted(self.loaded_data.keys())[-1] if self.loaded_data else None
        
        if not latest_key:
            print("没有找到嵌入数据")
            return
            
        data = self.loaded_data[latest_key]
        epoch_num = int(latest_key.split('_')[1])
        print(f"分析Epoch {epoch_num}的嵌入向量")
        
        if 'embeddings' not in data or data['embeddings'] is None:
            print("当前epoch没有嵌入数据")
            return
            
        embeddings = data['embeddings']
        
        # 创建嵌入分析图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 分析每种嵌入类型
        emb_types = ['tod_emb', 'dow_emb', 'adaptive_emb']
        emb_names = ['时间嵌入(ToD)', '星期嵌入(DoW)', '自适应嵌入']
        
        for idx, (emb_type, emb_name) in enumerate(zip(emb_types, emb_names)):
            if emb_type not in embeddings or embeddings[emb_type] is None:
                axes[0, idx].text(0.5, 0.5, f'{emb_name}\n无数据', 
                                ha='center', va='center', transform=axes[0, idx].transAxes)
                axes[1, idx].text(0.5, 0.5, f'{emb_name}\n无数据', 
                                ha='center', va='center', transform=axes[1, idx].transAxes)
                continue
                
            emb_data = embeddings[emb_type]
            if isinstance(emb_data, torch.Tensor):
                emb_data = emb_data.detach().cpu().numpy()
                
            print(f"{emb_name} 形状: {emb_data.shape}")
                
            # 1. 嵌入向量的L2范数分布
            norms = np.linalg.norm(emb_data, axis=-1)
            axes[0, idx].hist(norms.flatten(), bins=30, alpha=0.7)
            axes[0, idx].set_title(f'{emb_name} L2范数分布 (Epoch {epoch_num})')
            axes[0, idx].set_xlabel('L2范数')
            axes[0, idx].set_ylabel('频率')
            axes[0, idx].grid(True)
            
            # 2. 嵌入的PCA可视化
            # 重塑为2D用于PCA
            emb_data_2d = emb_data.reshape(-1, emb_data.shape[-1])
            
            if emb_data_2d.shape[0] > 1 and emb_data_2d.shape[1] > 1:  # 确保有足够的样本和维度
                try:
                    # 使用PCA降维到2D进行可视化
                    if emb_data_2d.shape[-1] > 2:
                        pca = PCA(n_components=2)
                        emb_pca = pca.fit_transform(emb_data_2d[:1000])  # 只取前1000个样本
                    else:
                        emb_pca = emb_data_2d[:1000]
                    
                    scatter = axes[1, idx].scatter(emb_pca[:, 0], emb_pca[:, 1], 
                                                 c=range(len(emb_pca)), cmap='viridis', alpha=0.7, s=1)
                    axes[1, idx].set_title(f'{emb_name} PCA可视化 (Epoch {epoch_num})')
                    axes[1, idx].set_xlabel('PC1')
                    axes[1, idx].set_ylabel('PC2')
                    plt.colorbar(scatter, ax=axes[1, idx])
                except Exception as e:
                    axes[1, idx].text(0.5, 0.5, f'可视化失败:\n{str(e)}', 
                                     ha='center', va='center', transform=axes[1, idx].transAxes)
            else:
                axes[1, idx].text(0.5, 0.5, f'数据不足\n无法可视化', 
                                 ha='center', va='center', transform=axes[1, idx].transAxes)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'embedding_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"嵌入向量分析完成，结果保存至: {self.output_dir / 'embedding_analysis.png'}")
        
    def analyze_spatiotemporal_features(self):
        """分析时空特征的演化"""
        print("\n=== 分析时空特征演化 ===")
        
        # 保存特征数据，用于热力图可视化
        temporal_features_data = None
        spatiotemporal_features_data = None
        raw_traffic_data = None
        selected_epoch = None
        
        # 只选择最新的一个epoch进行分析
        latest_key = sorted(self.loaded_data.keys())[-1] if self.loaded_data else None
        
        if latest_key:
            data = self.loaded_data[latest_key]
            selected_epoch = int(latest_key.split('_')[1])
            print(f"选择Epoch {selected_epoch}进行分析")
            
            # 从core_info中获取特征数据
            if 'core_info' in data and data['core_info'] is not None:
                core_info = data['core_info']
                
                # 时间特征数据
                if 'temporal_features' in core_info and core_info['temporal_features']:
                    temp_features = core_info['temporal_features']
                    print(f"找到时间特征数据，层数: {len(temp_features)}")
                    
                    # 保存原始特征数据用于热力图
                    if 'feature_data' in temp_features[0]:
                        feature_data = temp_features[0]['feature_data']
                        if isinstance(feature_data, torch.Tensor):
                            feature_data = feature_data.detach().cpu().numpy()
                        temporal_features_data = feature_data
                        print(f"时间特征数据形状: {temporal_features_data.shape}")
                
                # 时空特征数据
                if 'spatiotemporal_features' in core_info and core_info['spatiotemporal_features']:
                    st_features = core_info['spatiotemporal_features']
                    print(f"找到时空特征数据，层数: {len(st_features)}")
                    
                    # 保存原始特征数据用于热力图
                    if 'feature_data' in st_features[0]:
                        feature_data = st_features[0]['feature_data']
                        if isinstance(feature_data, torch.Tensor):
                            feature_data = feature_data.detach().cpu().numpy()
                        spatiotemporal_features_data = feature_data
                        print(f"时空特征数据形状: {spatiotemporal_features_data.shape}")
            
            # 获取原始交通流数据（如果存在）
            if 'inputs' in data and data['inputs'] is not None and 'x' in data['inputs']:
                x_data = data['inputs']['x']
                if isinstance(x_data, torch.Tensor):
                    x_data = x_data.detach().cpu().numpy()
                raw_traffic_data = x_data
                print(f"原始交通流数据形状: {raw_traffic_data.shape}")
        else:
            print("没有找到任何数据")
            return
            
        # 检查是否有足够的数据进行可视化
        if temporal_features_data is None and spatiotemporal_features_data is None:
            print("没有找到可用于可视化的特征数据")
            return
            
        print("\n开始生成热力图...")
        
        # 1. 节点间关系热力图
        if spatiotemporal_features_data is not None:
            print("\n生成节点间关系热力图...")
            # 处理数据形状，提取节点间关系
            if len(spatiotemporal_features_data.shape) >= 3:
                # 取第一个batch和时间步，计算节点间的相关性
                node_features = spatiotemporal_features_data[0, 0] if len(spatiotemporal_features_data.shape) >= 4 else spatiotemporal_features_data[0]
                print(f"节点特征形状: {node_features.shape}")
                
                # 计算节点间的相关性
                try:
                    # 计算相关系数矩阵
                    corr_matrix = np.corrcoef(node_features)
                    print(f"相关系数矩阵形状: {corr_matrix.shape}")
                    
                    # 绘制热力图
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(corr_matrix, cmap='viridis')
                    plt.title(f'节点间特征相关性热力图 (Epoch {selected_epoch})')
                    plt.xlabel('节点索引')
                    plt.ylabel('节点索引')
                    plt.tight_layout()
                    plt.savefig(self.output_dir / 'node_correlation_heatmap.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"已生成节点间相关性热力图，保存至: {self.output_dir / 'node_correlation_heatmap.png'}")
                except Exception as e:
                    print(f"生成节点间相关性热力图失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"时空特征数据形状不适合生成节点间关系热力图: {spatiotemporal_features_data.shape}")
        else:
            print("没有可用于生成节点间关系热力图的数据")
        
        # 2. 原始交通流与中间特征对比热力图
        if raw_traffic_data is not None and (temporal_features_data is not None or spatiotemporal_features_data is not None):
            print("\n生成特征对比热力图...")
            
            # 确定要展示的图表数量
            num_plots = 1 + (1 if temporal_features_data is not None else 0) + (1 if spatiotemporal_features_data is not None else 0)
            
            # 创建对比图
            fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 8))
            
            # 如果只有一个图表，确保axes是数组
            if num_plots == 1:
                axes = [axes]
            
            plot_idx = 0
            
            # 处理原始交通流数据
            if len(raw_traffic_data.shape) >= 3:
                # 取第一个batch和时间步
                raw_viz = raw_traffic_data[0, 0] if len(raw_traffic_data.shape) >= 4 else raw_traffic_data[0]
                # 如果节点数量太多，只取前100个
                raw_viz = raw_viz[:min(100, raw_viz.shape[0])]
                print(f"原始交通流可视化数据形状: {raw_viz.shape}")
                
                # 绘制热力图
                sns.heatmap(raw_viz, cmap='viridis', ax=axes[plot_idx])
                axes[plot_idx].set_title(f'原始交通流数据 (Epoch {selected_epoch})')
                axes[plot_idx].set_xlabel('特征维度')
                axes[plot_idx].set_ylabel('节点索引')
                plot_idx += 1
            else:
                print(f"原始数据形状不适合可视化: {raw_traffic_data.shape}")
            
            # 处理时间特征数据
            if temporal_features_data is not None and len(temporal_features_data.shape) >= 3:
                # 取第一个batch和时间步
                temp_viz = temporal_features_data[0, 0] if len(temporal_features_data.shape) >= 4 else temporal_features_data[0]
                # 如果节点数量太多，只取前100个
                temp_viz = temp_viz[:min(100, temp_viz.shape[0])]
                # 如果特征维度太多，只取前50个
                if temp_viz.shape[1] > 50:
                    temp_viz = temp_viz[:, :50]
                print(f"时间特征可视化数据形状: {temp_viz.shape}")
                
                # 绘制热力图
                sns.heatmap(temp_viz, cmap='viridis', ax=axes[plot_idx])
                axes[plot_idx].set_title(f'时间特征 (Epoch {selected_epoch})')
                axes[plot_idx].set_xlabel('特征维度')
                axes[plot_idx].set_ylabel('节点索引')
                plot_idx += 1
            elif temporal_features_data is not None:
                print(f"时间特征数据形状不适合可视化: {temporal_features_data.shape}")
            
            # 处理时空特征数据
            if spatiotemporal_features_data is not None and len(spatiotemporal_features_data.shape) >= 3:
                # 取第一个batch和时间步
                st_viz = spatiotemporal_features_data[0, 0] if len(spatiotemporal_features_data.shape) >= 4 else spatiotemporal_features_data[0]
                # 如果节点数量太多，只取前100个
                st_viz = st_viz[:min(100, st_viz.shape[0])]
                # 如果特征维度太多，只取前50个
                if st_viz.shape[1] > 50:
                    st_viz = st_viz[:, :50]
                print(f"时空特征可视化数据形状: {st_viz.shape}")
                
                # 绘制热力图
                sns.heatmap(st_viz, cmap='viridis', ax=axes[plot_idx])
                axes[plot_idx].set_title(f'时空特征 (Epoch {selected_epoch})')
                axes[plot_idx].set_xlabel('特征维度')
                axes[plot_idx].set_ylabel('节点索引')
            elif spatiotemporal_features_data is not None:
                print(f"时空特征数据形状不适合可视化: {spatiotemporal_features_data.shape}")
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'features_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"已生成特征对比热力图，保存至: {self.output_dir / 'features_comparison.png'}")
        else:
            print("没有足够的数据生成特征对比热力图")
        
        # 3. 节点时间特征关系热力图
        if temporal_features_data is not None:
            print("\n生成时间特征相关性热力图...")
            
            if len(temporal_features_data.shape) >= 3:
                # 假设形状为 [batch, time, nodes, features]
                # 我们取第一个batch，然后分析不同时间步的节点特征
                temp_features = temporal_features_data[0] if len(temporal_features_data.shape) >= 4 else temporal_features_data
                print(f"时间特征数据形状(用于相关性分析): {temp_features.shape}")
                
                # 计算不同时间步之间的相关性
                try:
                    # 对每个节点，计算不同时间步之间的相关性
                    time_steps = temp_features.shape[0]
                    nodes = min(3, temp_features.shape[1])  # 最多取3个节点
                    print(f"分析 {nodes} 个节点的时间相关性，每个节点有 {time_steps} 个时间步")
                    
                    # 创建一个大的图表
                    fig, axes = plt.subplots(1, nodes, figsize=(6*nodes, 6), squeeze=False)
                    
                    # 为前3个节点创建时间步相关性热力图
                    for i in range(nodes):
                        node_features = temp_features[:, i, :]
                        print(f"节点 {i} 的特征形状: {node_features.shape}")
                        
                        # 计算时间步之间的相关性
                        time_corr = np.corrcoef(node_features)
                        print(f"节点 {i} 的时间相关性矩阵形状: {time_corr.shape}")
                        
                        # 绘制热力图
                        sns.heatmap(time_corr, cmap='viridis', ax=axes[0, i])
                        axes[0, i].set_title(f'节点{i}的时间特征相关性 (Epoch {selected_epoch})')
                        axes[0, i].set_xlabel('时间步')
                        axes[0, i].set_ylabel('时间步')
                    
                    plt.tight_layout()
                    plt.savefig(self.output_dir / 'temporal_correlation.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"已生成时间特征相关性热力图，保存至: {self.output_dir / 'temporal_correlation.png'}")
                except Exception as e:
                    print(f"生成时间特征相关性热力图失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"时间特征数据形状不适合生成时间相关性热力图: {temporal_features_data.shape}")
        else:
            print("没有可用于生成时间特征相关性热力图的数据")
        
        print("\n热力图生成完成")
        plt.show()
        
    def analyze_prediction_quality(self):
        """分析预测质量"""
        print("\n=== 分析预测质量 ===")
        
        # 只选择最新的一个epoch进行分析
        latest_key = sorted(self.loaded_data.keys())[-1] if self.loaded_data else None
        
        if not latest_key:
            print("没有找到预测数据")
            return
            
        data = self.loaded_data[latest_key]
        epoch_num = int(latest_key.split('_')[1])
        print(f"分析Epoch {epoch_num}的预测质量")
        
        if 'final_outputs' not in data or data['final_outputs'] is None:
            print("当前epoch没有预测数据")
            return
            
        final_outputs = data['final_outputs']
        if 'prediction' not in final_outputs or final_outputs['prediction'] is None:
            print("当前epoch没有预测数据")
            return
            
        pred = final_outputs['prediction']
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
            
        print(f"预测形状: {pred.shape}")
        
        # 获取真实值（如果存在）
        ground_truth = None
        if 'inputs' in data and data['inputs'] is not None and 'y' in data['inputs']:
            y_data = data['inputs']['y']
            if isinstance(y_data, torch.Tensor):
                ground_truth = y_data.detach().cpu().numpy()
            else:
                ground_truth = y_data
            print(f"真实值形状: {ground_truth.shape}")
        
        # 创建预测质量分析图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 预测值分布
        pred_flat = pred.flatten()
        axes[0, 0].hist(pred_flat, bins=50, alpha=0.7, density=True)
        axes[0, 0].set_title(f'预测值分布 (Epoch {epoch_num})')
        axes[0, 0].set_xlabel('预测值')
        axes[0, 0].set_ylabel('密度')
        axes[0, 0].grid(True)
        
        # 2. 预测值统计量
        pred_mean = np.mean(pred)
        pred_std = np.std(pred)
        pred_min = np.min(pred)
        pred_max = np.max(pred)
        
        stats = [pred_mean, pred_std, pred_min, pred_max]
        labels = ['均值', '标准差', '最小值', '最大值']
        
        axes[0, 1].bar(range(len(stats)), stats)
        axes[0, 1].set_title(f'预测值统计量 (Epoch {epoch_num})')
        axes[0, 1].set_xticks(range(len(stats)))
        axes[0, 1].set_xticklabels(labels)
        axes[0, 1].grid(True)
        
        # 3. 如果有真实值，绘制预测vs真实值对比
        if ground_truth is not None:
            # 取第一个batch的第一个时间步进行对比
            if len(pred.shape) >= 3 and len(ground_truth.shape) >= 3:
                pred_sample = pred[0, 0]
                gt_sample = ground_truth[0, 0]
                
                # 确保形状匹配
                min_len = min(len(pred_sample), len(gt_sample))
                pred_sample = pred_sample[:min_len]
                gt_sample = gt_sample[:min_len]
                
                axes[1, 0].plot(gt_sample, label='真实值')
                axes[1, 0].plot(pred_sample, label='预测值')
                axes[1, 0].set_title(f'预测vs真实值 (Epoch {epoch_num})')
                axes[1, 0].set_xlabel('节点索引')
                axes[1, 0].set_ylabel('值')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
                
                # 计算误差
                error = pred_sample - gt_sample
                axes[1, 1].hist(error, bins=30, alpha=0.7)
                axes[1, 1].set_title(f'预测误差分布 (Epoch {epoch_num})')
                axes[1, 1].set_xlabel('误差')
                axes[1, 1].set_ylabel('频率')
                axes[1, 1].grid(True)
            else:
                axes[1, 0].text(0.5, 0.5, '预测和真实值形状不匹配', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 1].text(0.5, 0.5, '无法计算误差', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
        else:
            axes[1, 0].text(0.5, 0.5, '没有真实值数据', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 1].text(0.5, 0.5, '无法计算误差', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'prediction_quality.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"预测质量分析完成，结果保存至: {self.output_dir / 'prediction_quality.png'}")
        
    def generate_summary_report(self):
        """生成分析摘要报告"""
        print("\n=== 生成摘要报告 ===")
        
        # 只选择最新的一个epoch进行分析
        latest_key = sorted(self.loaded_data.keys())[-1] if self.loaded_data else None
        selected_epoch = int(latest_key.split('_')[1]) if latest_key else None
        
        report_lines = [
            "# STAEformer 可视化数据分析报告",
            f"生成时间: {pd.Timestamp.now()}",
            "",
            f"## 数据概览",
            f"- 分析文件: {latest_key if latest_key else '无数据'}",
            f"- 数据目录: {self.data_dir}",
            ""
        ]
        
        # 统计各类数据的可用性
        if latest_key and selected_epoch:
            data = self.loaded_data[latest_key]
            
            has_pattern = 'pattern_decomposition' in data and data['pattern_decomposition'] is not None
            has_embedding = 'embeddings' in data and data['embeddings'] is not None
            has_temporal = 'core_info' in data and data['core_info'] is not None and 'temporal_features' in data['core_info']
            has_spatiotemporal = 'core_info' in data and data['core_info'] is not None and 'spatiotemporal_features' in data['core_info']
            has_prediction = 'final_outputs' in data and data['final_outputs'] is not None
            
            report_lines.extend([
                f"## Epoch {selected_epoch} 数据类型",
                f"- 模式权重数据: {'有' if has_pattern else '无'}",
                f"- 嵌入数据: {'有' if has_embedding else '无'}",
                f"- 时间特征数据: {'有' if has_temporal else '无'}",
                f"- 时空特征数据: {'有' if has_spatiotemporal else '无'}",
                f"- 预测数据: {'有' if has_prediction else '无'}",
                "",
                "## 分析结果文件",
                "- pattern_analysis.png: 模式权重分析",
                "- embedding_analysis.png: 嵌入向量分析", 
                "- features_comparison.png: 特征对比分析",
                "- node_correlation_heatmap.png: 节点相关性分析",
                "- temporal_correlation.png: 时间相关性分析",
                "- prediction_quality.png: 预测质量分析",
                "",
                "## 主要发现",
                "请查看生成的图表文件获取详细分析结果。",
                "",
                "## 数据结构说明",
                "- 批量大小: 28",
                "- 输入时间步长: 12", 
                "- 输出时间步长: 12",
                "- 节点数量: 307",
                "- 模型维度: 152",
                "- 交通模式数量: 3"
            ])
        else:
            report_lines.append("没有找到有效数据进行分析")
        
        # 保存报告
        report_path = self.output_dir / 'analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"分析报告已保存: {report_path}")
        
    def run_complete_analysis(self, max_files=1):
        """运行完整的分析流程"""
        print("开始完整的可视化数据分析...")
        print(f"将加载最新的 {max_files} 个文件进行分析")
        
        # 加载数据
        self.load_data(max_files=max_files)
        
        if not self.loaded_data:
            print("没有成功加载任何数据，请检查数据目录和文件格式")
            return
        
        # 执行各项分析
        try:
            self.analyze_pattern_evolution()
        except Exception as e:
            print(f"模式分析失败: {e}")
            import traceback
            traceback.print_exc()
            
        try:
            self.analyze_embeddings()
        except Exception as e:
            print(f"嵌入分析失败: {e}")
            import traceback
            traceback.print_exc()
            
        try:
            self.analyze_spatiotemporal_features()
        except Exception as e:
            print(f"时空特征分析失败: {e}")
            import traceback
            traceback.print_exc()
            
        try:
            self.analyze_prediction_quality()
        except Exception as e:
            print(f"预测质量分析失败: {e}")
            import traceback
            traceback.print_exc()
            
        # 生成报告
        self.generate_summary_report()
        
        print(f"\n分析完成！结果保存在: {self.output_dir}")

def main():
    """主函数"""
    # 设置数据目录路径
    data_dir = "model/SAVE/train/MTGNN/PEMS04/visualization_data"
    
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        # 尝试相对路径
        alt_data_dir = os.path.join(os.path.dirname(__file__), data_dir)
        if os.path.exists(alt_data_dir):
            data_dir = alt_data_dir
            print(f"使用替代路径: {data_dir}")
        else:
            print(f"替代路径也不存在: {alt_data_dir}")
            return
    
    print(f"使用数据目录: {data_dir}")
    
    # 创建分析器并运行分析
    analyzer = VisualizationAnalyzer(data_dir)
    analyzer.run_complete_analysis(max_files=1)  # 只分析最新的1个文件

if __name__ == "__main__":
    main() 