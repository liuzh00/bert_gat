"""
Basic operations on trees.
"""
import torch
import numpy as np
import torch.nn.functional as F

def head_to_adj(sent_len, head, tokens, label, len_, mask, tok, directed=False, self_loop=True):
    """
    Convert a sequence of head indexes into a 0/1 matirx and label matrix.
    """
    adj_matrix = np.zeros((sent_len, sent_len), dtype=np.float32)
    label_matrix = np.zeros((sent_len, sent_len), dtype=np.int64)

    assert not isinstance(head, list)
    tokens = tokens[:len_].tolist()
    head = head[:len_].tolist()
    label = label[:len_].tolist()
    asp_idxs = [idx for idx in range(len(mask)) if mask[idx] == 1]
    for idx, head in enumerate(head):
        if idx in asp_idxs:
            for k in asp_idxs:
                adj_matrix[idx][k] = 1
                label_matrix[idx][k] = 2
        if head != 0:
            adj_matrix[idx, head - 1] = 1
            label_matrix[idx, head - 1] = label[idx]
        else:
            if self_loop:
                adj_matrix[idx, idx] = 1
                label_matrix[idx, idx] = 42  # 自环边
                continue
        if not directed:
            adj_matrix[head - 1, idx] = 1
            label_matrix[head - 1, idx] = label[idx]
        if self_loop:
            adj_matrix[idx, idx] = 1
            label_matrix[idx, idx] = 42
    adj_dict = adj_to_dict(adj_matrix, tokens)  # adj2dict
    return adj_matrix, label_matrix, adj_dict

def adj_to_dict(adj_matrix, tokens):
    """
    将邻接矩阵转换为邻接表
    """
    num_nodes = len(adj_matrix)
    adjacency_list = {}

    for i in range(num_nodes):
        neighbors = []
        for j in range(num_nodes):
            if adj_matrix[i][j] == 1 and i != j:  # discard self-loop
                neighbors.append(j)
        if neighbors:  # None
            adjacency_list[i] = neighbors

    return adjacency_list


def dep_distance_adj(adj, dep_post_adj, aspect_mask, len_, maxlen):
    """
    句法依存距离权重
    """
    weight = np.zeros((maxlen, maxlen), dtype=np.float32)
    dep_post_adj = dep_post_adj[:len_, :len_].add(torch.eye(len_)).numpy()
    max_distance = dep_post_adj.max().item()  # 样本最大句法依赖距离
    aspect_mask = aspect_mask[:len_].tolist()
    for i in range(len_):
        row_aspect = (aspect_mask[i] == 1)  # 判断该行词是否方面词
        for j in range(len_):
            col_aspect = (aspect_mask[j] == 1)  # 判断该列词是否方面词
            # 行是方面词且和列有依赖
            if row_aspect and adj[i][j] == 1:  # 若行是方面词且有依赖关系
                weight[i][j] = 1 - dep_post_adj[i][j] / (max_distance + 1)
            # 列是方面词且与行有依赖
            if col_aspect and adj[i][j] == 1:  # 若列是方面词且有依赖关系
                weight[i][j] = 1 - dep_post_adj[i][j] / (max_distance + 1)
    adj = adj + weight  # A = A * (L + 1)
    padding = -9e15 * np.ones_like(adj)
    adj = np.where(adj > 0, adj, padding)
    return adj  # srd = A * (L + 1)
