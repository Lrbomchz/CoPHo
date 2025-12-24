# data.py
import torch
from torch.utils.data import Dataset

class GraphDataset(Dataset):
    def __init__(self, graphs, src_list, tgt_list, dist_list):
        """
        graphs: 图的邻接矩阵列表，每个元素为形状 [N, N] 的 numpy 数组或二维列表
        src_list, tgt_list: 源节点和目标节点的索引列表（与 graphs 对应）
        dist_list: 最短路径距离标签列表
        """
        self.graphs = graphs
        self.src_list = src_list
        self.tgt_list = tgt_list
        self.dist_list = dist_list

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        # 将邻接矩阵和距离转换为张量，索引为整数
        adj_matrix = torch.tensor(self.graphs[idx], dtype=torch.float32)
        src = int(self.src_list[idx])
        tgt = int(self.tgt_list[idx])
        dist = torch.tensor(self.dist_list[idx], dtype=torch.float32)
        return adj_matrix, src, tgt, dist

def collate_fn(batch):
    """
    将一批GraphDataset样本打包成批次张量。
    每个样本: (adj_matrix, src, tgt, dist)
    输出:
      batch_adj: [B, N_max, N_max]的张量，N_max为该批次中最大图节点数
      batch_src: [B] 源节点索引张量
      batch_tgt: [B] 目标节点索引张量
      batch_dist: [B] 实际距离标签张量
    """
    # 批次大小
    batch_size = len(batch)
    # 找到该批次中最大节点数量，用于padding
    max_nodes = max(item[0].shape[0] for item in batch)
    # 准备填充邻接矩阵的张量，初始化为全0
    batch_adj = torch.zeros(batch_size, max_nodes, max_nodes, dtype=torch.float32)
    batch_src = []
    batch_tgt = []
    batch_dist = []
    for i, (adj, src, tgt, dist) in enumerate(batch):
        N = adj.shape[0]
        # 将原邻接矩阵填入张量的前 N 行 N 列
        batch_adj[i, :N, :N] = adj
        batch_src.append(src)
        batch_tgt.append(tgt)
        batch_dist.append(dist.item())  # dist 是张量，取出标量值
    batch_src = torch.tensor(batch_src, dtype=torch.long)
    batch_tgt = torch.tensor(batch_tgt, dtype=torch.long)
    batch_dist = torch.tensor(batch_dist, dtype=torch.float32)
    return batch_adj, batch_src, batch_tgt, batch_dist