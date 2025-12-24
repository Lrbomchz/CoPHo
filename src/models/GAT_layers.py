# layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 可学习的线性变换，将每个节点的输入特征映射到输出特征空间
        self.W = nn.Linear(in_features, out_features, bias=False)
        # 注意力打分的参数，将两个节点特征拼接后映射为一个标量打分
        self.attn_fc = nn.Linear(2 * out_features, 1, bias=False)
        self.dropout = dropout

    def forward(self, h, adj):
        """
        h: 节点特征张量，形状 [B, N, in_features]
        adj: 邻接矩阵张量，形状 [B, N, N] （元素为1表示有边，0表示无边）
        返回更新后的节点特征，形状 [B, N, out_features]
        """
        # 1. 线性变换节点特征
        Wh = self.W(h)  # [B, N, out_features]
        B, N, _ = Wh.shape

        # 2. 计算所有节点对的注意力打分 e_{ij}
        # 通过广播机制构造每对邻居的组合特征 [Wh_i || Wh_j]
        Wh_i = Wh.unsqueeze(2).expand(B, N, N, self.out_features)  # [B, N, N, out_features]
        Wh_j = Wh.unsqueeze(1).expand(B, N, N, self.out_features)  # [B, N, N, out_features]
        # 拼接后通过前馈网络得到注意力打分
        e = self.attn_fc(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1)  # [B, N, N]

        # 3. 对不存在连接的对赋极低值以屏蔽它们的影响
        e = e + (1 - adj) * (-1e9)

        # 4. 对每个节点i的邻居j的得分做softmax归一化
        alpha = F.softmax(e, dim=-1)  # [B, N, N]，按行softmax（每个节点的邻边权重）
        if self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 5. 加权汇聚邻居特征
        h_out = torch.matmul(alpha, Wh)  # [B, N, out_features]

        return h_out