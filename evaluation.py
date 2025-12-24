import numpy as np
import os
import torch
import pickle
import re
import pandas as pd
import networkx as nx
from collections import defaultdict

def parse_experiment_name(exp_name):
    """
    Parse experiment name like:
      |conditions-transitivity|phase0.6-1.0|modetop+-1

    Returns:
      condition (str), copho_start (float), copho_end (float)
    """
    # split by '|' and drop empty parts
    parts = [p for p in exp_name.split("|") if p]

    condition = None
    copho_start = None
    copho_end = None

    for p in parts:
        if p.startswith("conditions-"):
            # e.g. "conditions-transitivity" -> "transitivity"
            condition = p[len("conditions-"):]
        elif p.startswith("phase"):
            # e.g. "phase0.6-1.0" -> "0.6-1.0"
            phase_str = p[len("phase"):]
            try:
                s_str, e_str = phase_str.split("-")
                copho_start = float(s_str)
                copho_end = float(e_str)
            except Exception as e:
                raise ValueError(f"Failed to parse phase in experiment name: {exp_name}") from e

    if condition is None or copho_start is None or copho_end is None:
        raise ValueError(f"Cannot parse experiment name: {exp_name}")

    return condition, copho_start, copho_end


def read_graph_list(file_path):
    """
    从一个txt文件中读取多个图，文件中每个图用空行分隔，
    每个图的格式如下：

    N=64
    X:
    0 0 0 0 ...  (X的数值可能跨一行或多行)
    E:
    0 0 0 0 ...   # 第一行（应有N个数字）
    0 0 0 0 ...   # 第二行
    ...
    共N行

    返回一个graph_list，每个graph是一个字典，包含键 "N", "X", "E"。
    """
    graph_list = []
    with open(file_path, 'r') as f:
        # 先将文件内容按空行分块
        content = f.read().strip()
    # 以连续的换行符分割每个图块
    blocks = content.split("\n\n")
    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        graph = {}
        idx = 0
        # 解析节点数
        if lines[idx].startswith("N="):
            try:
                graph["N"] = int(lines[idx].split('=')[1].strip())
            except Exception as e:
                raise ValueError(f"解析节点数失败: {lines[idx]}") from e
            idx += 1
        else:
            raise ValueError("图块缺少节点数信息 (N=...)")
        # 解析X部分：假设标识行为"X:"，之后一段行直到遇到"E:"为止
        if idx < len(lines) and lines[idx].startswith("X:"):
            idx += 1
            X_vals = []
            while idx < len(lines) and not lines[idx].startswith("E:"):
                X_vals.extend(lines[idx].split())
                idx += 1
            # 这里将所有数字转为 float；根据实际情况可改为 int
            graph["X"] = np.array([float(x) for x in X_vals])
        else:
            raise ValueError("图块缺少X部分")
        # 解析E部分：假设标识行为"E:"，之后紧跟N行，每行N个数字
        if idx < len(lines) and lines[idx].startswith("E:"):
            idx += 1
            E_list = []
            N = graph["N"]
            for i in range(N):
                if idx + i >= len(lines):
                    raise ValueError("E部分的行数不足")
                row = [int(x) for x in lines[idx + i].split()]
                if len(row) != N:
                    raise ValueError(f"第{i + 1}行E数据长度不等于N")
                E_list.append(row)
            graph["E"] = np.array(E_list)
            idx += N
        else:
            raise ValueError("图块缺少E部分")
        graph_list.append(graph)
    return graph_list

def load_generated_samples(directory):
    """
    Reads all 'generated_samples*.txt' files in `directory`,
    uses read_graph_list to load graphs from each file,
    and returns two lists: one of graph lists and one of experiment names.
    """
    graphs_per_experiment = []
    experiment_names = []

    for fname in os.listdir(directory):
        if fname.startswith("generated_samples") and fname.endswith(".txt"):
            # full path to the file
            file_path = os.path.join(directory, fname)

            # extract the experiment name between 'generated_samples' and '.txt'
            match = re.match(r"generated_samples(.+)\.txt$", fname)
            if not match:
                continue
            exp_name = match.group(1)

            # read graphs and store
            graphs = read_graph_list(file_path)
            graphs_per_experiment.append(graphs)
            experiment_names.append(exp_name)

    return graphs_per_experiment, experiment_names



def compute_experiment_mae_metrics(graphs_per_experiment, experiment_names, test_data):
    """
    For each experiment, parse its condition & phase range from experiment_names,
    compute MAE only for that condition, and aggregate into a table:

      rows   = "copho_start-copho_end"
      cols   = [clustering, assortativity, transitivity, density]
      values = (MAE * 100), rounded to 2 decimals.
               Missing condition -> -1.
    """
    all_conditions = ['density', 'assortativity', 'transitivity', 'clustering']

    # 1. Compute metrics for test_data (ground truth) once
    test_metrics = {name: [] for name in all_conditions}
    for g in test_data:
        E = g.numpy()
        G = nx.from_numpy_array(E)
        test_metrics['density'].append(nx.density(G))
        test_metrics['clustering'].append(nx.average_clustering(G))
        test_metrics['assortativity'].append(nx.degree_assortativity_coefficient(G))
        test_metrics['transitivity'].append(nx.transitivity(G))
    for name in all_conditions:
        test_metrics[name] = np.array(test_metrics[name])

    count_graph = len(test_data)

    # 2. For each experiment, compute MAE only for its own condition
    #    and aggregate by (phase_range, condition)
    #    phase_key = f"{copho_start}-{copho_end}"
    mae_by_phase_and_cond = defaultdict(lambda: defaultdict(list))

    for graphs, exp_name in zip(graphs_per_experiment, experiment_names):
        condition, copho_start, copho_end = parse_experiment_name(exp_name)
        if condition not in all_conditions:
            print(f"[WARN] Unknown condition '{condition}' in experiment name '{exp_name}', skip.")
            continue

        print(f"Experiment: {exp_name} | condition={condition}, phase={copho_start}-{copho_end}")

        bs = len(graphs)
        sub_count = bs // count_graph
        mae_list = []

        for i in range(sub_count):
            sub_graphs = graphs[i * count_graph: (i + 1) * count_graph]
            diffs = []
            for j, g in enumerate(sub_graphs):
                E = np.array(g['E'])
                G = nx.from_numpy_array(E)
                if condition == 'density':
                    val = nx.density(G)
                elif condition == 'clustering':
                    val = nx.average_clustering(G)
                elif condition == 'assortativity':
                    val = nx.degree_assortativity_coefficient(G)
                else:  # 'transitivity'
                    val = nx.transitivity(G)

                diffs.append(abs(val - test_metrics[condition][j]))
            mae_list.append(np.mean(diffs))

        if len(mae_list) == 0:
            print(f"[WARN] No sub-graphs for experiment '{exp_name}', skip.")
            continue

        mae_value = float(np.mean(mae_list))  # scalar
        phase_key = f"{copho_start}-{copho_end}"
        mae_by_phase_and_cond[phase_key][condition].append(mae_value)

    # 3. Build final DataFrame: rows = phase ranges, cols = conditions
    #    If no experiment for a (phase, condition) => -1
    phase_keys = sorted(
        mae_by_phase_and_cond.keys(),
        key=lambda k: (float(k.split('-')[0]), float(k.split('-')[1]))
    )

    rows = []
    for phase_key in phase_keys:
        row = {}
        for cond in all_conditions:
            vals = mae_by_phase_and_cond[phase_key].get(cond, [])
            if len(vals) == 0:
                row[cond] = -1.0
            else:
                # average over multiple experiments, *100 and round to 2 decimals
                m = float(np.mean(vals) * 100.0)
                row[cond] = f"{m:.2f}"
        rows.append(row)

    df = pd.DataFrame(rows, index=phase_keys, columns=all_conditions)
    return df

# load the original data (condition ground truth)
test_data = torch.load("data/comm20/raw/test.pt")

# load the generated data
graphs_per_experiment, experiment_names = load_generated_samples("outputs/")

# compute mae between ground truth and generated data
df = compute_experiment_mae_metrics(graphs_per_experiment, experiment_names, test_data)
print(df)