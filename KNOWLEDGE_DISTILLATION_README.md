# 知识蒸馏训练系统

本系统实现了基于STAEformer（教师）→ STMLP（学生）的知识蒸馏训练框架。

## 🎯 系统架构

### 教师模型：STAEformer
- **特点**: 复杂的时空注意力机制，性能强大但计算开销大
- **用途**: 提供高质量的预测结果和中间特征表示
- **保存位置**: `SAVE/train/MTGNN/PEMS04/best_model.pth`

### 学生模型：STMLP  
- **特点**: 轻量级MLP架构，计算效率高
- **用途**: 通过知识蒸馏学习教师模型的知识，在保持效率的同时提升性能
- **保存位置**: `SAVE/train/MTGNN/PEMS04/best_modelstudent.pth`

## 🚀 快速开始

### 方法1：使用便捷脚本（推荐）

```bash
# 完整知识蒸馏流程（先训练教师，再训练学生）
python run_knowledge_distillation.py -t

# 只训练学生模型（需要预训练的教师模型）
python run_knowledge_distillation.py
```

### 方法2：手动执行

```bash
# 第一阶段：训练教师模型
python Mains.py --dataset PEMS04 --device cuda:0 --mode train --epochs 100 -t --use_staeformer_teacher --tod True

# 第二阶段：知识蒸馏训练学生模型
python Mains.py --dataset PEMS04 --device cuda:0 --mode train --epochs 100 --use_staeformer_teacher --tod True
```

## 📊 数据特征支持

系统严格按照STAEformer的数据处理逻辑：

### 特征类型
- **交通流量数据**: 基础特征（必需），**需要标准化**
- **Time of Day (tod)**: 时间占位符（0-1范围），**不需要归一化**
- **Day of Week (dow)**: 星期占位符（0-6范围），**不需要归一化**

### 数据处理逻辑
```python
# 1. 数据加载时：只对交通流量进行标准化
scaler_data = StandardScaler(data_train[..., 0:1].mean(), data_train[..., 0:1].std())
# tod和dow保持原始值，用作嵌入索引

# 2. 模型内部：STAEformer会将tod/dow转换为嵌入
if self.tod_embedding_dim > 0:
    tod_emb = self.tod_embedding((tod * self.steps_per_day).long())
if self.dow_embedding_dim > 0:
    dow_emb = self.dow_embedding(dow.long())
```

### 配置示例
```python
# 在 Mains.py 中配置
train_loader, val_loader, test_loader, scaler_data, scaler_day, scaler_week = get_dataloader(
    args,
    normalizer=args.normalizer,
    tod=True,   # 启用时间占位符
    dow=False,  # 禁用星期占位符
    weather=False, 
    single=False
)
```

## 🔧 核心组件

### 1. 修复后的训练逻辑 (`Mains.py`)
- ✅ 明确区分教师模型和学生模型
- ✅ 正确的知识蒸馏训练流程
- ✅ 独立的优化器和学习率调度器

### 2. 增强的数据加载器 (`lib/dataloader.py`)
- ✅ 支持tod/dow特征选择
- ✅ 智能特征维度处理
- ✅ 数据缓存机制

### 3. 完善的训练器 (`model/BasicTrainer.py`)
- ✅ 专门的教师模型训练方法 `train_teacher()`
- ✅ 知识蒸馏训练方法 `trainS()`
- ✅ 多种损失函数组合

### 4. 修复的教师模型 (`model/Teacher.py`)
- ✅ 移除调试断点
- ✅ STAEformerTeacher包装器
- ✅ 兼容的输出格式

## 📈 知识蒸馏损失函数

学生模型的总损失包含：

```python
loss = loss1 + 10*tkloss + 1*scl
```

其中：
- `loss1`: 基础预测损失（MSE/MAE）
- `tkloss`: KL散度损失（知识蒸馏核心）
- `scl`: 对比学习损失（特征对齐）

## 📁 项目结构

```
TP/
├── model/
│   ├── Mains.py                    # 主训练脚本
│   ├── BasicTrainer.py             # 训练器类
│   ├── Teacher.py                  # 模型定义
│   └── STAEformer.py              # STAEformer实现
├── lib/
│   ├── dataloader.py              # 数据加载器
│   ├── metrics.py                 # 评估指标
│   └── utils.py                   # 工具函数
├── run_knowledge_distillation.py  # 便捷运行脚本
└── KNOWLEDGE_DISTILLATION_README.md
```

## 🎛️ 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | PEMS04 | 数据集名称 |
| `--device` | cuda:0 | 训练设备 |
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 32 | 批次大小 |
| `--lr_init` | 0.001 | 初始学习率 |
| `--early_stop_patience` | 15 | 早停耐心值 |
| `-t` | False | 是否训练教师模型 |
| `--use_staeformer_teacher` | False | 使用STAEformer作为教师 |
| `--tod` | False | 启用时间特征 |

## 🔍 监控训练过程

训练日志保存在：`SAVE/train/MTGNN/PEMS04/STAEformer-PEMS04-{timestamp}.log`

关键指标：
- **Train Loss**: 训练损失
- **Val Loss**: 验证损失  
- **RMSE/MAE/MAPE**: 回归评估指标
- **推理时间**: 模型效率指标

## 🚨 常见问题

### Q: 教师模型训练失败？
A: 检查GPU内存，STAEformer需要较大显存。可以减小batch_size。

### Q: 知识蒸馏效果不好？
A: 调整损失函数权重，特别是KL散度损失的权重（当前为10）。

### Q: 数据维度不匹配？
A: 检查tod/dow设置，确保数据特征维度与模型期望一致。

### Q: 缓存数据有问题？
A: 删除 `../PEMS_data/PEMS04/data_cache/` 目录重新生成缓存。

## 📝 更新日志

- **2025-01-24**: 
  - ✅ 修复教师模型训练逻辑
  - ✅ 添加tod/dow特征支持
  - ✅ 移除调试断点
  - ✅ 完善知识蒸馏流程

## 🤝 贡献

如有问题或改进建议，请提交Issue或Pull Request。

---
**注意**: 确保在运行前已正确配置数据路径和CUDA环境。
