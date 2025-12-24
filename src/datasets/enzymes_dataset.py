import os
import pathlib

import torch
from torch.utils.data import random_split
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

import networkx as nx
import pickle

class MyData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        # 在拼接时不在任何维度拼接 cond，按照列表形式存储
        if key == 'cond':
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        # 在反切片重建时，不对 cond 做任何“偏移”
        if key == 'cond':
            return 0
        return super().__inc__(key, value, *args, **kwargs)

def extract_adj_matrices(graphs, as_numpy=True):
    adj_list = []
    for G in graphs:
        # 使用 nx.to_numpy_array 得到 (n, n) 的密集矩阵
        A = nx.to_numpy_array(
            G)  # 默认 dtype=float，nonedge=0.0
        if as_numpy:
            adj_list.append(torch.tensor(A))
        else:
            # 如果需要稀疏矩阵，可用 adjacency_matrix
            # 注意返回的是 SciPy CSR 矩阵
            A_sparse = nx.adjacency_matrix(
                G)
            adj_list.append(A_sparse)
    return adj_list

def pad_and_stack_adjs(
    adj_list,
    pad_value=-1.0
) -> torch.Tensor:
    # 找到最大的维度
    max_n = max(mat.shape[0] for mat in adj_list)
    bs = len(adj_list)

    # 创建一个填充后的张量，初始都填 pad_value
    stacked = torch.full((bs, max_n, max_n),
                         pad_value,
                         dtype=adj_list[0].dtype,
                         device=adj_list[0].device)

    # 依次将每个矩阵 copy 到左上角
    for i, mat in enumerate(adj_list):
        n = mat.shape[0]
        stacked[i, :n, :n] = mat

    return stacked

class SpectreGraphDataset(InMemoryDataset):
    def __init__(self, dataset_name, split, root, transform=None, pre_transform=None, pre_filter=None, cond_type=None):
        self.enzymes_file = 'ENZYMES.pkl'
        self.dataset_name = dataset_name
        self.split = split
        self.num_graphs = 500
        self.cond_type = ["transitivity"]
        if cond_type != None:
            self.cond_type = cond_type
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
            return [self.split + '.pt']

    def download(self):
        # 定义本地数据集路径
        local_dataset_path = '/data/ssd1/xigongli/zzy'
        file_path = os.path.join(local_dataset_path, 'ENZYMES.pkl')

        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file {file_path} not found.")

        #adjs, eigvals, eigvecs, n_nodes, max_eigval, min_eigval, same_sample, n_max = torch.load(file_path)

        with open(file_path, 'rb') as f:
            result = pickle.load(f)

        adjs = extract_adj_matrices(result)
        #adjs = pad_and_stack_adjs(data)

        graph_num = len(adjs)
        print(len(adjs))

        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)

        test_len = int(round(self.num_graphs * 0.2))
        train_len = int(round((self.num_graphs) * 0.8))
        val_len = graph_num - train_len - test_len
        indices = torch.randperm(graph_num, generator=g_cpu)
        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len + val_len]
        test_indices = indices[train_len + val_len:]

        train_data = []
        val_data = []
        test_data = []

        for i, adj in enumerate(adjs):
            if i in train_indices:
                train_data.append(adj)
            elif i in val_indices:
                val_data.append(adj)
            elif i in test_indices:
                test_data.append(adj)
            else:
                raise ValueError(f'Index {i} not in any split')

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])

    def load_struct_cond(self,file_path='/home/xigongli/zzy/gnn_struct/graph_metrics_enzymes_test.pkl'):
            """
            读取保存的结果文件，并返回包含 graph_list, srcs_list, tgts_list 和 dist_list 的字典。
            """
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            print("condition type: ", self.cond_type)
            return [data[f'{type}_list'] for type in self.cond_type]  #


    def process(self):
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])

        conds = torch.tensor(self.load_struct_cond()) * 10

        data_list = []
        for idx, adj in enumerate(raw_dataset):
            n = adj.shape[-1]
            X = torch.ones(n, 1, dtype=torch.float)
            y = torch.zeros([1, 0]).float()

            if self.split == "test":
                cond = torch.tensor([cond[idx] for cond in conds])
            else:
                cond = torch.zeros([1, 0]).float()

            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            num_nodes = n * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                             y=y, n_nodes=num_nodes, cond=cond)
            # print(X.shape)
            # data_list.append(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])



class SpectreGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=200, cond_type=None):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)


        datasets = {'train': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                                 split='train', root=root_path,),
                    'val': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                        split='val', root=root_path),
                    'test': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                        split='test', root=root_path, cond_type=cond_type)}
        # print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class SpectreDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = torch.tensor([1])               # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)

