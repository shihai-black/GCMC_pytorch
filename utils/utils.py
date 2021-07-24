# -*- coding: utf-8 -*-
# @project：GCMC-Pytorch-dgl
# @author:caojinlei
# @file: utils.py
# @time: 2021/07/14
# -*- coding: utf-8 -*-
# @project：GCMC-Pytorch-dgl
# @author:caojinlei
# @file: utils.py
# @time: 2021/07/13
import csv
import dgl
import re
import torch as th
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict


def load_subtensor(input_nodes, pair_graph, blocks, dataset, parent_graph):
    output_nodes = pair_graph.ndata[dgl.NID]
    head_feat = input_nodes['user'] if dataset.user_feature is None else \
        dataset.user_feature[input_nodes['user']]
    tail_feat = input_nodes['item'] if dataset.item_feature is None else \
        dataset.item_feature[input_nodes['item']]

    for block in blocks:
        block.dstnodes['user'].data['ci'] = \
            parent_graph.nodes['user'].data['ci'][block.dstnodes['user'].data[dgl.NID]]
        block.srcnodes['user'].data['cj'] = \
            parent_graph.nodes['user'].data['cj'][block.srcnodes['user'].data[dgl.NID]]
        block.dstnodes['item'].data['ci'] = \
            parent_graph.nodes['item'].data['ci'][block.dstnodes['item'].data[dgl.NID]]
        block.srcnodes['item'].data['cj'] = \
            parent_graph.nodes['item'].data['cj'][block.srcnodes['item'].data[dgl.NID]]

    return head_feat, tail_feat, blocks


def flatten_etypes(pair_graph, dataset, segment):
    n_users = pair_graph.number_of_nodes('user')
    n_items = pair_graph.number_of_nodes('item')
    src = []
    dst = []
    labels = []
    ratings = []

    for rating in dataset.possible_rating_values:
        src_etype, dst_etype = pair_graph.edges(order='eid', etype=str(rating))
        src.append(src_etype)
        dst.append(dst_etype)
        label = np.searchsorted(dataset.possible_rating_values, rating)
        ratings.append(th.LongTensor(np.full_like(src_etype, rating)))
        labels.append(th.LongTensor(np.full_like(src_etype, label)))
    src = th.cat(src)
    dst = th.cat(dst)
    ratings = th.cat(ratings)
    labels = th.cat(labels)

    flattened_pair_graph = dgl.heterograph({
        ('user', 'rate', 'item'): (src, dst)},
        num_nodes_dict={'user': n_users, 'item': n_items})
    flattened_pair_graph.edata['rating'] = ratings
    flattened_pair_graph.edata['label'] = labels

    return flattened_pair_graph


def to_etype_name(rating):
    return str(rating).replace('.', '_')
