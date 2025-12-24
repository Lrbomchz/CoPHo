import os
import pathlib

import torch
from torch.utils.data import random_split
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url
import pandas as pd
import pickle

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from torch_geometric.data import Data

from hydra.utils import to_absolute_path, get_original_cwd

class MyData(Data):
    def __cat_dim__(self, key, value, store=None):
        if key == 'cond':
            # 返回 None 表示 cond 属性在 collate 时不自动拼接
            return None
        return super().__cat_dim__(key, value, store)
    

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

class SpectreGraphDataset(InMemoryDataset):
    def __init__(self, dataset_name, split, root, transform=None, pre_transform=None, pre_filter=None, cond_type=None):
        self.sbm_file = 'sbm_200.pt'
        self.planar_file = 'planar_64_200.pt'
        self.comm20_file = 'community_12_21_100.pt'
        self.dataset_name = dataset_name
        self.split = split
        self.num_graphs = 200
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
        return 
        # 定义本地数据集路径
        local_dataset_path = to_absolute_path("input_data/conditions/")

        if self.dataset_name == 'sbm':
            file_path = os.path.join(local_dataset_path, 'sbm_200.pt')
        elif self.dataset_name == 'planar':
            file_path = os.path.join(local_dataset_path, 'planar_64_200.pt')
        elif self.dataset_name == 'comm20':
            file_path = os.path.join(local_dataset_path, 'community_12_21_100.pt')
        else:
            raise ValueError(f'Unknown dataset {self.dataset_name}')

        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file {file_path} not found.")

        adjs, eigvals, eigvecs, n_nodes, max_eigval, min_eigval, same_sample, n_max = torch.load(file_path)

        print("adjs类型是：" + str(type(adjs)))
        print(adjs)

        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)

        test_len = int(round(self.num_graphs * 0.2))
        train_len = int(round((self.num_graphs - test_len) * 0.8))
        val_len = self.num_graphs - train_len - test_len
        indices = torch.randperm(self.num_graphs, generator=g_cpu)
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

    def csv_to_tensor(self, file_path: str) -> torch.Tensor:
        df = pd.read_csv(file_path)
        data = df.values.astype(float)
        return torch.tensor(data, dtype=torch.float)

    def load_shortest_path(self, file_name='shortest_path_comm20_testset.pkl'):
        """
        读取保存的结果文件，并返回包含 graph_list, srcs_list, tgts_list 和 dist_list 的字典。
        """
        local_dataset_path = to_absolute_path("input_data/conditions/")
        file_path = os.path.join(local_dataset_path, file_name)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data['sps']

    def load_struct_cond(self, file_name='graph_struct_metrics_comm20_testset.pkl'):
        """
        读取保存的结果文件，并返回包含 graph_list, srcs_list, tgts_list 和 dist_list 的字典。
        """
        local_dataset_path = to_absolute_path("input_data/conditions/")
        file_path = os.path.join(local_dataset_path, file_name)

        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print("condition type: ", self.cond_type)
        return [data[f'{type}_list'] for type in self.cond_type]  #

    def process(self):
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])

        if self.dataset_name == 'sbm':
            pass
        elif self.dataset_name == 'planar':
            conds = self.load_shortest_path()
            conds = torch.stack(conds, dim=0)
        elif self.dataset_name == 'comm20':
            conds = torch.tensor(self.load_struct_cond())
            #conds = torch.stack(conds, dim=0)
        else:
            raise ValueError(f'Unknown dataset {self.dataset_name}')
        print("[DEBUG]",conds.shape)


        data_list = []
        for idx, adj in enumerate(raw_dataset):
            n = adj.shape[-1]
            X = torch.ones(n, 1, dtype=torch.float)
            # y = torch.zeros([1, 0]).float()
            # y = data_y[idx].unsqueeze(0)
            y = torch.zeros([1, 0]).float()
            if self.split == "test":
                cond = torch.tensor([cond[idx] for cond in conds])
            else:
                cond = torch.zeros([1, 0]).float()
            print(cond)
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            num_nodes = n * torch.ones(1, dtype=torch.long)
            data = MyData(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                             y=y, n_nodes=num_nodes, cond=cond)
            #data_list.append(data)

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
                                                 split='train', root=root_path),
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

