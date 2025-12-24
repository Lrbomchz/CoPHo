import os
import torch_geometric.utils
from collections import deque
from omegaconf import OmegaConf, open_dict
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch
import omegaconf
import wandb
from typing import List, Dict, Tuple, Optional
import networkx as nx
import numpy as np
import math


def condition_relaxed_H(noisy_data, target, target_type, device, threshold=0.50):

    bs, n, _, _ = noisy_data['E_t'].shape
    E_t = noisy_data['E_t']  # [bs, n, n, 2]
    A_t = E_t[..., 1].bool()  # [bs, n, n], True indicates the presence of an edge

    relaxed_H = torch.zeros(bs, dtype=torch.bool, device=device)
    strict_H = torch.zeros(bs, dtype=torch.bool, device=device)

    # We can extend `target_type` to add more generator-supported condition types.
    # Note: Any new types added here also need to be handled in the classifier/regressor.
    target_type = target_type[0]
    if target_type in ["density", "clustering", "assortativity", "transitivity"]:
        import networkx as nx

        # Compute the conditional metric (\phi(G)_t^i) for each graph in the batch (e.g., density, clustering, assortativity, transitivity)
        # \phi(G)_t^i
        metrics = []
        for i in range(bs):
            # Construct an undirected graph from the adjacency matrix
            A_np = A_t[i].cpu().numpy().astype(int)      # Convert to a 0/1 adjacency matrix (0 = no edge, 1 = edge exists)
            G = nx.from_numpy_array(A_np)

            if target_type == "density":
                m = nx.density(G)
            elif target_type == "clustering":
                m = nx.average_clustering(G, weight=None)
            elif target_type == "assortativity":
                m = nx.degree_assortativity_coefficient(G)
            elif target_type == "transitivity":
                m = nx.transitivity(G)
            if math.isnan(m):
                m=1e5
            metrics.append(m)

        # Convert to tensor and feed into the check function: compute the conditional metric \phi(G)_t^i-y and test whether it satisfies the threshold (1[\phi(G)_t^i-y<=\epsilon])
        metric_tensor = torch.tensor(metrics, dtype=torch.float, device=device)  # [bs]

        condi_metric, indi_func_res = check_struct_condition(metric_tensor, target.reshape(metric_tensor.shape), threshold)  # bool [bs]
    else:
        raise ValueError(f"Unsupported condition type: {target_type}")

    # \phi(G)_t^i-y, 1[\phi(G)_t^i-y<=\epsilon]
    return condi_metric, indi_func_res


def check_struct_condition(metric_tensor: torch.Tensor,
                           target,
                           threshold) -> torch.BoolTensor:
    """
    Check structural condition over a batch by comparing each metric value
    to its corresponding target value within a tolerance threshold.

    Returns a boolean mask of shape [bs], where
        True  if |metric_tensor[i] - target[i]| <= threshold[i],
        False otherwise.

    Args:
        metric_tensor (torch.Tensor): shape [bs], the per-graph metric values.
        target (float or sequence of floats or torch.Tensor):
            - If a single float, the same target is used for every element.
            - If a sequence or 1D tensor of length bs, a per-element target.
        threshold (float or sequence of floats or torch.Tensor):
            - If a single float, the same tolerance for every element.
            - If a sequence or 1D tensor of length bs, a per-element tolerance.

    Returns:
        torch.BoolTensor: shape [bs].
    """
    # bring target and threshold to tensors on same device/dtype
    tgt = torch.as_tensor(target, dtype=metric_tensor.dtype, device=metric_tensor.device)
    thr = torch.as_tensor(threshold, dtype=metric_tensor.dtype, device=metric_tensor.device)

    # expand scalar target to match batch
    if tgt.numel() == 1:
        tgt = tgt.expand_as(metric_tensor)
    elif tgt.shape != metric_tensor.shape:
        raise ValueError(f"target shape {tuple(tgt.shape)} does not match metric_tensor shape {tuple(metric_tensor.shape)}")

    # expand scalar threshold to match batch
    if thr.numel() == 1:
        thr = thr.expand_as(metric_tensor)
    elif thr.shape != metric_tensor.shape:
        raise ValueError(f"threshold shape {tuple(thr.shape)} does not match metric_tensor shape {tuple(metric_tensor.shape)}")

    # |\phi(G)_t^i -y|
    diff = torch.abs(metric_tensor - target)

    if diff.dim() == 1:
        # 1D case: use directly
        diff_out = diff
    elif diff.dim() == 2:
        # 2D case: average over last dimension
        diff_out = diff.mean(dim=-1)
    else:
        raise ValueError(f"Unsupported diff dimensions: {diff.dim()}")

    # compare to threshold (1[\phi(G)_t^i-y<=\epsilon])
    mask = diff_out <= threshold

    # diff_out = |\phi(G)_t^i -y|, mask = 1[\phi(G)_t^i-y<=\epsilon]
    return diff_out, mask


def condition_homo_test_gen(
    grad_e: torch.Tensor,
    noisy_data: Dict[str, torch.Tensor],
    device: torch.device,
    threshold: float = None,
) -> List[Dict[str, torch.Tensor]]:
    '''
    This function constructs a sequence of candidate conditional graphs \hat{G}_t^i based on persistent homology, where i ranges from 1 to T_homo.
    Generate a sequence of noisy_data by gradually removing edges along the gradient,
    producing a new noisy_data after each edge removal.
    The returned list has length homo_step, corresponding to the maximum number of removal steps.
    '''
    grad_e = grad_e.to(device)                                    # [bs, n, n]
    bs, n, _ = grad_e.shape

    # original adjacency matrix
    E0 = noisy_data['E_t'][..., 1].bool().to(device)              # [bs, n, n]

    # For each graph, build a list of candidate edges to delete, sorted by grad in descending order
    removal_lists = []

    # Threshold branch: originally used to select all edges with gradients greater than a given threshold.
    # Since gradients are hard to control, this branch is deprecated and replaced by using T_homo to control the number of homology edges.
    if threshold is not None:
        for i in range(bs):
            # Find all positions where grad_e[i] > threshold
            us, vs = torch.nonzero(grad_e[i] > threshold, as_tuple=True)
            # Sort them by gradient value
            vals = grad_e[i, us, vs]
            order = torch.argsort(vals, descending=True)
            removal = [(int(us[o]), int(vs[o])) for o in order]
            removal_lists.append(removal)

    # Main branch: use T_homo to control the number of homology steps
    else:
        # top-10 gradients
        homo_step = 5
        for i in range(bs):
            flat = grad_e[i].flatten()
            topk = torch.topk(flat, min(homo_step, flat.numel()), sorted=True)
            idxs = topk.indices
            us = (idxs // n)
            vs = (idxs % n)
            removal = [(int(u), int(v)) for u, v in zip(us, vs)]
            removal_lists.append(removal)

    # homo_step = maximum number of edge removal steps
    homo_step = max(len(rem) for rem in removal_lists)

    # Iteratively remove edges and record the noisy_data at each step
    noisy_data_homos = []
    current_E = E0.clone()  # Deletions are accumulated across steps within the loop

    for step in range(homo_step):
        # For each graph, remove the edge corresponding to the current step (if it exists)
        for i, rem in enumerate(removal_lists):
            if step < len(rem):
                u, v = rem[step]
                current_E[i, u, v] = False
                current_E[i, v, u] = False

        # Reconstruct one-hot E_t from current_E
        E_t_step = torch.stack([~current_E, current_E], dim=-1).to(device)  # [bs, n, n, 2]

        # Build a new noisy_data dict (reuse other fields from the original and only move them to device)
        noisy_step = {
            'X_t': noisy_data['X_t'].to(device),
            'E_t': E_t_step,
            'y_t': noisy_data['y_t'].to(device),
            't':   noisy_data['t'].to(device),
            'node_mask': noisy_data['node_mask'].to(device),
        }
        noisy_data_homos.append(noisy_step)

    return noisy_data_homos

def condition_homo_check(
    noisy_data_homos: List[Dict[str, torch.Tensor]],
    target: torch.Tensor,
    target_type: str,
    device: torch.device,
    threshold=0.5,
) -> Tuple[List[Dict], List[Optional[int]], List[Optional[List[int]]], List[Optional[torch.Tensor]]]:
    """
    Compute the final weight (w(G)) for each graph and return the graph with the largest weight:
      - noisy_data_prime: List[Dict] of length bs; if no graph satisfies the condition,
        the corresponding element is set to None.
    """
    bs = noisy_data_homos[0]['E_t'].shape[0]
    # T_homo
    T = len(noisy_data_homos)

    # strict represents the indicator function in the weight,
    # while relaxed represents the conditional weight φ(G) - y
    strict_hist = torch.zeros((T, bs), dtype=torch.bool, device=device)
    relaxed_hist = torch.zeros((T, bs), dtype=torch.float, device=device)
    w_G = torch.zeros((T, bs), dtype=torch.float, device=device)
    for t, nd in enumerate(noisy_data_homos):
        r, s = condition_relaxed_H(nd, target, target_type, device, threshold)
        strict_hist[t] = s
        relaxed_hist[t] = r
        # exp(-|\hat(G)-G|) * (\phi(hat(G))-y)**-2 * 1[\phi(hat(G))-y<epsilon]
        #print(t,r,s)
        factor = s.float()  # True->1.0, False->0.0
        w_G[t] = math.exp(-(t+1)/(2*len(noisy_data_homos))) * (r ** (-2)) * (factor)
        #w_G[t] = math.exp(-(t + 1)) * r ** (-2) * factor

    # If no candidate satisfies the condition, ensure that argmax falls back to the original generated graph (G_t)
    w_G[0]+=1e-10

    #min_steps = relaxed_hist.argmin(dim=0)
    #mins = relaxed_hist.min(dim=0)
    first_steps = w_G.argmax(dim=0)
    # Construct noisy_data_prime
    noisy_prime = []
    for i in range(bs):
        t = first_steps[i]
        if t is None:
            noisy_prime.append(None)
        else:
            # Take the data of the i-th graph at step t (BEST candidate)
            nd = noisy_data_homos[t]
            sliced = {k: v[i].unsqueeze(0) for k,v in nd.items()}
            #print(sliced.keys())
            #print(sliced["E_t"].shape)
            noisy_prime.append(sliced)

    return noisy_prime

