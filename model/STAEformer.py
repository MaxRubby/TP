import torch.nn as nn
import torch
from torchinfo import summary


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class STAEformer(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
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
        # 新增参数：交通模式解耦
        num_traffic_patterns=3,  # 交通模式数量P
        use_pattern_decomposition=True,  # 是否启用模式解耦
        pattern_mlp_hidden_dim=64,  # MLP隐藏层维度
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        
        # 新增：模式解耦参数
        self.num_traffic_patterns = num_traffic_patterns
        self.use_pattern_decomposition = use_pattern_decomposition
        
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

        # 新增：交通模式解耦MLP
        if self.use_pattern_decomposition:
            # 计算时空嵌入特征的总维度
            embedding_dim = tod_embedding_dim + dow_embedding_dim + spatial_embedding_dim
            
            # 模式比例学习的MLP
            self.pattern_mlp = nn.Sequential(
                nn.Linear(embedding_dim, pattern_mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(pattern_mlp_hidden_dim, pattern_mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(pattern_mlp_hidden_dim, num_traffic_patterns)
            )
            
            # 模式解耦后的特征投影层
            self.pattern_proj = nn.Linear(input_dim * num_traffic_patterns, input_embedding_dim)

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, save_visualization_data=False):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        # 初始化可视化数据字典
        if save_visualization_data:
            
            self.visualization_data = {
                'input_data': {},
                'embeddings': {},
                'pattern_decomposition': {},
                'temporal_features': [],
                'spatiotemporal_features': [],
                'attention_weights': [],
                'final_outputs': {}
            }
            
            # 保存输入数据
            self.visualization_data['input_data']['raw_input'] = x.clone().detach()

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
            if save_visualization_data:
                self.visualization_data['input_data']['tod'] = tod.clone().detach()
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
            if save_visualization_data:
                self.visualization_data['input_data']['dow'] = dow.clone().detach()
        
        # 保存原始交通流数据用于模式解耦
        x_raw = x[..., : self.input_dim]  # (batch_size, in_steps, num_nodes, input_dim)
        if save_visualization_data:
            self.visualization_data['input_data']['x_raw'] = x_raw.clone().detach()

        x = self.input_proj(x_raw)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]
        
        # 收集时空嵌入特征用于模式解耦
        embedding_features = []
        
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding((tod * self.steps_per_day).long())  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
            embedding_features.append(tod_emb)
            if save_visualization_data:
                self.visualization_data['embeddings']['tod_emb'] = tod_emb.clone().detach()
                
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
            embedding_features.append(dow_emb)
            if save_visualization_data:
                self.visualization_data['embeddings']['dow_emb'] = dow_emb.clone().detach()
                
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
            embedding_features.append(spatial_emb)
            if save_visualization_data:
                self.visualization_data['embeddings']['spatial_emb'] = spatial_emb.clone().detach()
                self.visualization_data['embeddings']['node_emb_params'] = self.node_emb.clone().detach()
                
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
            if save_visualization_data:
                self.visualization_data['embeddings']['adaptive_emb'] = adp_emb.clone().detach()
                self.visualization_data['embeddings']['adaptive_emb_params'] = self.adaptive_embedding.clone().detach()
        
        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)
        if save_visualization_data:
            self.visualization_data['embeddings']['combined_features'] = x.clone().detach()
        
        # 交通模式解耦
        if self.use_pattern_decomposition and len(embedding_features) > 0:
            # 1. 特征嵌入拼接：F_emb = Concat(T_d, T_w, E_nd)
            F_emb = torch.cat(embedding_features, dim=-1)  # (batch_size, in_steps, num_nodes, embedding_dim)
            if save_visualization_data:
                self.visualization_data['pattern_decomposition']['F_emb'] = F_emb.clone().detach()
            
            # 2. 模式比例学习：通过MLP和Softmax学习每个时空位置属于不同交通模式的比例
            # Omega'_p(t,i) = MLP(F_emb(t,i))
            pattern_logits = self.pattern_mlp(F_emb)  # (batch_size, in_steps, num_nodes, num_traffic_patterns)
            if save_visualization_data:
                self.visualization_data['pattern_decomposition']['pattern_logits'] = pattern_logits.clone().detach()
            
            # Omega_p(t,i) = Softmax(Omega'_p(t,i))_p
            pattern_weights = torch.softmax(pattern_logits, dim=-1)  # (batch_size, in_steps, num_nodes, num_traffic_patterns)
            if save_visualization_data:
                self.visualization_data['pattern_decomposition']['pattern_weights'] = pattern_weights.clone().detach()
            
            # 3. 交通流解耦：X_p(t,i) = X_raw(t,i) ⊙ Omega_p(t,i)
            # 扩展原始交通流维度以匹配模式数量
            x_raw_expanded = x_raw.unsqueeze(-1)  # (batch_size, in_steps, num_nodes, input_dim, 1)
            pattern_weights_expanded = pattern_weights.unsqueeze(-2)  # (batch_size, in_steps, num_nodes, 1, num_traffic_patterns)
            
            # 计算每个模式的交通流
            x_patterns = x_raw_expanded * pattern_weights_expanded  # (batch_size, in_steps, num_nodes, input_dim, num_traffic_patterns)
            if save_visualization_data:
                self.visualization_data['pattern_decomposition']['x_patterns'] = x_patterns.clone().detach()
            
            # 将多模式交通流重新组织为特征
            # 这里我们将不同模式的流量作为额外的特征维度
            x_patterns_reshaped = x_patterns.view(batch_size, self.in_steps, self.num_nodes, -1)  # (batch_size, in_steps, num_nodes, input_dim * num_traffic_patterns)
            
            # 将模式解耦后的特征投影到相同的嵌入维度
            x_pattern_emb = self.pattern_proj(x_patterns_reshaped)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
            if save_visualization_data:
                self.visualization_data['pattern_decomposition']['x_pattern_emb'] = x_pattern_emb.clone().detach()
            
            # 更新特征列表，用模式解耦后的特征替换原始输入特征
            features[0] = x_pattern_emb
            x = torch.cat(features, dim=-1)  # 重新拼接所有特征
            
            # 存储模式权重用于分析（可选）
            self.last_pattern_weights = pattern_weights

        # 保存中间特征用于知识蒸馏
        temporal_features = []  # 模拟tout：时间注意力后的特征
        spatiotemporal_features = []  # 模拟sout：时空注意力后的特征
        
        # 时间注意力层
        for i, attn in enumerate(self.attn_layers_t):
            x = attn(x, dim=1)
            temporal_features.append(x.clone())  # 保存时间注意力后的特征
            if save_visualization_data:
                self.visualization_data['temporal_features'].append({
                    'layer_idx': i,
                    'feature': x.clone().detach()
                })
            
        # 空间注意力层
        for i, attn in enumerate(self.attn_layers_s):
            x = attn(x, dim=2)
            spatiotemporal_features.append(x.clone())  # 保存时空注意力后的特征
            if save_visualization_data:
                self.visualization_data['spatiotemporal_features'].append({
                    'layer_idx': i,
                    'feature': x.clone().detach()
                })
        
        # (batch_size, in_steps, num_nodes, model_dim)
        if save_visualization_data:
            self.visualization_data['final_outputs']['pre_projection_features'] = x.clone().detach()

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        if save_visualization_data:
            self.visualization_data['final_outputs']['prediction'] = out.clone().detach()
            # 添加一些统计信息
            self.visualization_data['meta_info'] = {
                'batch_size': batch_size,
                'in_steps': self.in_steps,
                'out_steps': self.out_steps,
                'num_nodes': self.num_nodes,
                'model_dim': self.model_dim,
                'num_traffic_patterns': self.num_traffic_patterns if self.use_pattern_decomposition else 0,
                'use_pattern_decomposition': self.use_pattern_decomposition
            }

        # 返回主输出和中间特征
        # temporal_features[-1]: 最后一层时间注意力特征 (模拟tout[-1])
        # spatiotemporal_features[-1]: 最后一层时空注意力特征 (模拟sout[-1])
        return out, temporal_features[-1] if temporal_features else None, spatiotemporal_features[-1] if spatiotemporal_features else None


if __name__ == "__main__":
    model = STAEformer(207, 12, 12)
    summary(model, [64, 12, 207, 3])
