# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GAT_layers import GraphAttentionLayer

class GraphDistanceModel(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=2, dropout=0.1):
        super(GraphDistanceModel, self).__init__()
        self.hidden_dim = hidden_dim
        # 输入编码：将2维的(src标记, tgt标记)特征映射到隐藏维度
        self.input_encoder = nn.Linear(2, hidden_dim)
        # 图神经网络层：使用注意力机制的 GNN 层，堆叠 num_layers 层
        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)
        ])
        # 输出解码层：将源和目标的隐藏表示转换为距离预测
        # 这里使用两层全连接：先将[src_feat, tgt_feat, |src_feat - tgt_feat|]映射到隐藏维，再映射到单一标量输出
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, adj_batch, src_idx_batch, tgt_idx_batch):
        """
        adj_batch: [B, N, N] 批量邻接矩阵
        src_idx_batch: [B] 每个图的源节点索引
        tgt_idx_batch: [B] 每个图的目标节点索引
        返回: [B] 每个图预测的距离（标量）
        """
        B, N, _ = adj_batch.shape  # B:批大小, N:图中节点数(已padding为批中最大值)
        device = adj_batch.device

        # 构造初始节点特征矩阵，[B, N, 2]，第3维的两个值分别表示源/目标节点标记
        x = torch.zeros(B, N, 2, device=device)
        # 对每个图，将对应源节点位置的第一个特征置1，目标节点位置的第二个特征置1
        x[torch.arange(B), src_idx_batch, 0] = 1.0
        x[torch.arange(B), tgt_idx_batch, 1] = 1.0

        # 输入编码：线性映射到隐藏维度大小 [B, N, hidden_dim]
        h = self.input_encoder(x)

        # 图神经网络传播：依次通过每一层 GraphAttentionLayer
        for layer in self.gnn_layers:
            h = layer(h, adj_batch)
        # 此时 h 为最终每个节点的表示，[B, N, hidden_dim]

        # 提取每个图中源节点和目标节点的最终表示
        src_feat = h[torch.arange(B), src_idx_batch]  # [B, hidden_dim]
        tgt_feat = h[torch.arange(B), tgt_idx_batch]  # [B, hidden_dim]

        # 将源和目标的表示进行组合：直接拼接以及差的绝对值
        combined = torch.cat([src_feat, tgt_feat, torch.abs(src_feat - tgt_feat)], dim=-1)  # [B, 3*hidden_dim]

        # 输出层解码出距离并压缩维度 [B, 1] -> [B]
        output = self.output_layer(combined).squeeze(-1)  # [B]

        return output


class GraphStructModel(nn.Module):
    """
    Graph structuring model that predicts a graph-level structural metric.
    Inputs:
      - adj_batch: [B, N, N] adjacency matrices
      - mask_batch: [B, N] boolean mask for valid nodes
    Output:
      - predictions: [B] scalar metric per graph
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, out_dim: int, dropout: float):
        super(GraphStructModel, self).__init__()
        # Node feature encoder from scalar to hidden_dim
        self.input_fc = nn.Linear(in_dim, hidden_dim)
        # Stacked Graph Attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        # MLP head: hidden -> hidden/2 -> 1
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, adj_batch: torch.Tensor, mask_batch: torch.Tensor) -> torch.Tensor:
        B, N, _ = adj_batch.size()
        # Compute initial node features (degree)
        deg = adj_batch.sum(dim=-1, keepdim=True)       # [B, N, 1]
        h = F.relu(self.input_fc(deg))                   # [B, N, hidden_dim]
        # Propagate through GAT layers
        for gat in self.gat_layers:
            h = gat(h, adj_batch)                        # each returns [B, N, hidden_dim]
        # Masked mean pooling to get graph representation
        mask = mask_batch.unsqueeze(-1)                  # [B, N, 1]
        h = h * mask                                      # zero out padding nodes
        sum_h = h.sum(dim=1)                             # [B, hidden_dim]
        denom = mask.sum(dim=1).clamp(min=1)             # [B, 1]
        graph_repr = sum_h / denom                       # [B, hidden_dim]
        # MLP head to scalar
        out = self.mlp(graph_repr).squeeze(-1)            # [B]
        return out