# -*- coding: utf-8 -*-
# @project：GCMC-Pytorch-dgl
# @author:caojinlei
# @file: dataload.py
# @time: 2021/07/13
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import torch

import dgl


class DataSet:
    def __init__(self, path, user_embed_path=None, item_embed_path=None, device='cpu', test_ratio=0.2, valid_ratio=0.05,
                 use_one_hot_fea=False, symm=True):
        self.path = path
        self.user_embed_path = user_embed_path
        self.item_embed_path = item_embed_path
        self.device = device
        self.test_ratio = test_ratio
        self.valid_ratio = valid_ratio
        self.use_one_hot_fea = use_one_hot_fea
        self.symm = symm

        # Load data and spilt data
        data_list, train_data, test_data, valid_data, all_train_data = self._load_data()
        all_rating_pairs, all_rating_values = self._generate_pair_value(data_list)
        all_train_rating_pairs, all_train_rating_values = self._generate_pair_value(all_train_data)
        train_rating_pairs, train_rating_values = self._generate_pair_value(train_data)
        valid_rating_pairs, valid_rating_values = self._generate_pair_value(valid_data)
        test_rating_pairs, test_rating_values = self._generate_pair_value(test_data)

        # Map user/item to the global id
        self.global_user_id_map = {ele: i for i, ele in enumerate(list(set(all_rating_pairs[0])))}
        self.global_item_id_map = {ele: i for i, ele in enumerate(list(set(all_rating_pairs[1])))}
        print('Total user number = {}, item number = {}'.format(len(self.global_user_id_map),
                                                                len(self.global_item_id_map)))
        self._num_user = len(self.global_user_id_map)
        self._num_item = len(self.global_item_id_map)
        self.possible_rating_values = np.unique(all_train_rating_values).astype(int)

        # generate feature
        self._generate_features()

        # generate enc & dec graph
        self.train_enc_graph, self.train_dec_graph, self.train_labels, self.train_truths = self.generate_input_data(
            train_rating_pairs,
            train_rating_values,
            train_rating_pairs,
            train_rating_values,
            add_support=True)
        self.test_enc_graph, self.test_dec_graph, self.test_labels, self.test_truths = self.generate_input_data(
            all_train_rating_pairs,
            all_train_rating_values,
            test_rating_pairs,
            test_rating_values,
            add_support=True)
        self.valid_enc_graph = self.train_enc_graph
        self.valid_dec_graph = self._generate_dec_graph(valid_rating_pairs)
        self.valid_labels = self._make_labels(valid_rating_values)
        self.valid_truths = torch.FloatTensor(valid_rating_values)

        # statistic data
        print("Train enc graph: #user:{} #item:{} #pairs:{}".format(
            self.train_enc_graph.number_of_nodes('user'), self.train_enc_graph.number_of_nodes('item'),
            self.train_enc_graph.number_of_edges() / 2))
        print("Train dec graph: #user:{} #item:{} #pairs:{}".format(
            self.train_dec_graph.number_of_nodes('user'), self.train_dec_graph.number_of_nodes('item'),
            self.train_dec_graph.number_of_edges()))
        print("test enc graph: #user:{} #item:{} #pairs:{}".format(
            self.test_enc_graph.number_of_nodes('user'), self.test_enc_graph.number_of_nodes('item'),
            self.test_enc_graph.number_of_edges() / 2))
        print("test dec graph: #user:{} #item:{} #pairs:{}".format(
            self.test_dec_graph.number_of_nodes('user'), self.test_dec_graph.number_of_nodes('item'),
            self.test_dec_graph.number_of_edges()))
        print("valid enc graph: #user:{} #item:{} #pairs:{}".format(
            self.valid_enc_graph.number_of_nodes('user'), self.valid_enc_graph.number_of_nodes('item'),
            self.valid_enc_graph.number_of_edges() / 2))
        print("valid dec graph: #user:{} #item:{} #pairs:{}".format(
            self.valid_dec_graph.number_of_nodes('user'), self.valid_dec_graph.number_of_nodes('item'),
            self.valid_dec_graph.number_of_edges()))

    def _load_data(self):
        data_list = []
        with open(self.path, 'r') as file:
            for lines in file.readlines():
                line = lines.strip().split(' ')
                data_list.append(line)
        length_data = len(data_list)
        test_index = int(length_data * self.test_ratio)
        valid_index = int(length_data * (self.test_ratio + self.valid_ratio))
        test_data = data_list[:test_index]
        valid_data = data_list[test_index:valid_index]
        train_data = data_list[valid_index:]
        all_train_data = data_list[test_index:]
        return data_list, train_data, test_data, valid_data, all_train_data

    def _load_features(self):
        return torch.FloatTensor(np.load(self.user_embed_path)), torch.FloatTensor(np.load(self.item_embed_path))

    def _generate_pair_value(self, data_info):
        user_list = []
        item_list = []
        rate_list = []
        for line in data_info:
            user, item, rate = line
            user_list.append(user)
            item_list.append(item)
            rate_list.append(rate)
        rating_pairs = (np.array(user_list, dtype=np.int64), np.array(item_list, dtype=np.int64))
        rating_values = np.array(rate_list, dtype=np.float32)
        return rating_pairs, rating_values

    def _generate_features(self):
        if self.use_one_hot_fea:
            self.user_feature = None
            self.item_feature = None
        else:
            # TODO 将训练好的embedding拿过来
            # 随机给予初始纬度参数
            self.user_feature, self.item_feature = self._load_features()
        if self.user_feature is None:
            self.user_feature_shape = (self._num_user, self._num_user)
            self.item_feature_shape = (self._num_item, self._num_item)
        else:
            self.user_feature_shape = self.user_feature.shape
            self.item_feature_shape = self.item_feature.shape
        info_line = "Feature dim: "
        info_line += "\nuser: {}".format(self.user_feature_shape)
        info_line += "\nitem: {}".format(self.item_feature_shape)
        print(info_line)

    @property
    def num_user(self):
        return self._num_user

    @property
    def num_item(self):
        return self._num_item

    def _generate_enc_graph(self, rating_pairs, rating_values, add_support=False):
        user_item_r = np.zeros((self._num_user, self._num_item), dtype=np.float32)
        user_item_r[rating_pairs] = rating_values
        data_dict = {}
        num_nodes_dict = {'user': self._num_user, 'item': self._num_item}
        rating_row, rating_col = rating_pairs
        for rating in self.possible_rating_values:
            ridx = np.where(rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = str(rating)
            data_dict.update({
                ('user', str(rating), 'item'): (rrow, rcol),
                ('item', 'rev-%s' % str(rating), 'user'): (rcol, rrow)
            })
        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
        if add_support:
            def _calc_norm(x):
                x = x.numpy().astype('float32')
                x[x == 0.] = np.inf
                x = torch.FloatTensor(1. / np.sqrt(x))
                return x.unsqueeze(1)

            user_ci = []
            user_cj = []
            item_ci = []
            item_cj = []
            for r in self.possible_rating_values:
                r = str(r)
                user_ci.append(graph['rev-%s' % r].in_degrees())
                item_ci.append(graph[r].in_degrees())
                if self.symm:
                    user_cj.append(graph[r].out_degrees())
                    item_cj.append(graph['rev-%s' % r].out_degrees())
                else:
                    user_cj.append(torch.zeros((self.num_user,)))
                    item_cj.append(torch.zeros((self.num_item,)))
            user_ci = _calc_norm(sum(user_ci))
            item_ci = _calc_norm(sum(item_ci))
            if self.symm:
                user_cj = _calc_norm(sum(user_cj))
                item_cj = _calc_norm(sum(item_cj))
            else:
                user_cj = torch.ones(self.num_user, )
                item_cj = torch.ones(self.num_item, )
            graph.nodes['user'].data.update({'ci': user_ci, 'cj': user_cj})
            graph.nodes['item'].data.update({'ci': item_ci, 'cj': item_cj})
        return graph

    def _generate_dec_graph(self, rating_pairs):
        return dgl.heterograph({('user', 'rate', 'item'): rating_pairs},
                               num_nodes_dict={'user': self.num_user, 'item': self.num_item})

    def _make_labels(self, ratings):
        labels = torch.LongTensor(np.searchsorted(self.possible_rating_values, ratings)).to(self.device)
        return labels

    def generate_input_data(self, rating_pairs_enc, rating_values_enc, rating_pairs_dec, rating_labels,
                            add_support=False):
        enc_graph = self._generate_enc_graph(rating_pairs_enc, rating_values_enc, add_support)
        dec_graph = self._generate_dec_graph(rating_pairs_dec)
        labels = self._make_labels(rating_labels)
        truths = torch.FloatTensor(rating_labels)
        return enc_graph, dec_graph, labels, truths


class DataLoad:
    def __init__(self, path, user_embed_path=None, item_embed_path=None, device='cpu', test_ratio=0.2, valid_ratio=0.05,
                 use_one_hot_fea=False, batch_size=64, num_workers=2):
        self.dataset = DataSet(path, user_embed_path, item_embed_path, device, test_ratio, valid_ratio, use_one_hot_fea)
        # reverse_types = {str(k): 'rev-' + str(k)
        #                  for k in self.dataset.possible_rating_values}
        # reverse_types.update({v: k for k, v in reverse_types.items()})
        self.sampler = dgl.dataloading.MultiLayerNeighborSampler([None], return_eids=True)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

    def data_load(self, data='train'):
        if data == 'train':
            eid = {str(k): torch.arange(self.dataset.train_enc_graph.number_of_edges(etype=str(k))) for k in
                   self.dataset.possible_rating_values}
            data_iter = dgl.dataloading.EdgeDataLoader(self.dataset.train_enc_graph,
                                                       eid, self.sampler,
                                                       batch_size=self.batch_size, shuffle=True, drop_last=False,
                                                       num_workers=self.num_workers)
        elif data == 'test':
            data_iter = dgl.dataloading.EdgeDataLoader(self.dataset.test_dec_graph,
                                                       torch.arange(self.dataset.test_dec_graph.number_of_edges()),
                                                       self.sampler, g_sampling=self.dataset.test_enc_graph,
                                                       batch_size=self.batch_size, shuffle=False, drop_last=False,
                                                       num_workers=self.num_workers)
        else:
            data_iter = dgl.dataloading.EdgeDataLoader(self.dataset.valid_dec_graph,
                                                       torch.arange(self.dataset.valid_dec_graph.number_of_edges()),
                                                       self.sampler, g_sampling=self.dataset.valid_enc_graph,
                                                       batch_size=self.batch_size, shuffle=False, drop_last=False,
                                                       num_workers=self.num_workers)
        return data_iter


if __name__ == '__main__':
    from tqdm import tqdm

    dataloader = DataLoad(path='input/demo/ml-1m_trans.txt', user_embed_path='input/demo/ml_1m_user_feature.npy',
                          item_embed_path='input/demo/ml_1m_item_feature.npy', batch_size=10000, use_one_hot_fea=False)
    train_iter = dataloader.data_load('train')
    count = 0
    with tqdm(train_iter) as tk:
        for step, (input_nodes, pair_graph, blocks) in enumerate(tk):
            pass
