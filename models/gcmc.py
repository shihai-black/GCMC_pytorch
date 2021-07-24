# -*- coding: utf-8 -*-
# @project：GCMC-Pytorch-dgl
# @author:caojinlei
# @file: model.py
# @time: 2021/07/13
import torch
import tqdm
from torch import nn
from torch.nn import init
import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn


def dot_or_identity(A, B, device=None):
    # if A is None, treat as identity matrix
    if A is None:
        return B
    elif len(A.shape) == 1:
        if device == 'cpu':
            return B[A]
        else:
            return B[A].to(device)
    else:
        return A @ B


class GCMCGraphConv(nn.Module):
    """
    GCN layer
    """

    def __init__(self, input_dim, output_dim, weight=True, device='cpu', dropout_rate=0):
        super(GCMCGraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = weight
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)
        if weight:
            self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)

    def forward(self, graph, input_feat, weight=None):
        with graph.local_scope():
            if isinstance(input_feat, tuple):
                input_feat, _ = input_feat
            cj = graph.srcdata['cj']
            ci = graph.dstdata['ci']
            if self.device != 'cpu':
                cj = cj.to(self.device)
                ci = ci.to(self.device)
            if weight is not None:
                if self.weight is not None:
                    raise dgl.DGLError('External weight is provided while at the same time the'
                                       ' module has defined its own weight parameter. Please'
                                       ' create the module with flag weight=False.')
            else:
                weight = self.weight
            if weight is not None:
                input_feat = dot_or_identity(input_feat, weight, self.device)

            input_feat = input_feat * self.dropout(cj)
            graph.srcdata['h'] = input_feat
            graph.update_all(fn.copy_src('h', 'm'),
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            rst = rst * ci

        return rst


class GCMCLayer(nn.Module):
    """
    GCMC layer
    """

    def __init__(self, rating_values, user_in_units, item_in_units, msg_units, out_units,
                 dropout_rate=0, agg='stack', agg_act=None, out_act=None, share_user_item_param=False,
                 device='cpu'):
        super(GCMCLayer, self).__init__()
        self.rating_values = rating_values
        self.agg = agg
        self.share_user_item_param = share_user_item_param
        self.ufc = nn.Linear(msg_units, out_units)
        if share_user_item_param:
            self.ifc = self.ufc
        else:
            self.ifc = nn.Linear(msg_units, out_units)
        if agg == 'stack':
            assert msg_units % len(rating_values) == 0
            msg_units = msg_units // len(rating_values)
        self.dropout = nn.Dropout(dropout_rate)
        self.W_r = nn.ParameterDict()
        subConv = {}
        for rating in rating_values:
            rating = str(rating)
            rev_rating = 'rev-%s' % rating
            if share_user_item_param and user_in_units == item_in_units:
                self.W_r[rating] = nn.Parameter(torch.randn(user_in_units, msg_units))
                self.W_r['rev-%s' % rating] = self.W_r[rating]
                subConv[rating] = GCMCGraphConv(user_in_units,
                                                msg_units,
                                                weight=False,
                                                device=device,
                                                dropout_rate=dropout_rate)
                subConv[rev_rating] = GCMCGraphConv(user_in_units,
                                                    msg_units,
                                                    weight=False,
                                                    device=device,
                                                    dropout_rate=dropout_rate)
            else:
                self.W_r = None
                subConv[rating] = GCMCGraphConv(user_in_units,
                                                msg_units,
                                                weight=True,
                                                device=device,
                                                dropout_rate=dropout_rate)
                subConv[rev_rating] = GCMCGraphConv(item_in_units,
                                                    msg_units,
                                                    weight=True,
                                                    device=device,
                                                    dropout_rate=dropout_rate)
        self.conv = dglnn.HeteroGraphConv(subConv, aggregate=agg)
        self.agg_act = self.get_activation(agg_act)
        self.out_act = self.get_activation(out_act)
        self.device = device
        self.reset_parameters()

    @staticmethod
    def get_activation(act):
        """Get the activation based on the act string

        Parameters
        ----------
        act: str or callable function

        Returns
        -------
        ret: callable function
        """
        if act is None:
            return lambda x: x
        if isinstance(act, str):
            if act == 'leaky':
                return nn.LeakyReLU(0.1)
            elif act == 'relu':
                return nn.ReLU()
            elif act == 'tanh':
                return nn.Tanh()
            elif act == 'sigmoid':
                return nn.Sigmoid()
            elif act == 'softsign':
                return nn.Softsign()
            else:
                raise NotImplementedError
        else:
            return act

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def partial_to(self):
        if self.device != 'cpu':
            self.ufc.to(self.device)
            if not self.share_user_item_param:
                self.ifc.to(self.device)
            self.dropout.to(self.device)

    def forward(self, graph, user_feat, item_feat):
        input_feature = {'user': user_feat, 'item': item_feat}
        mod_args = {}
        for i, rating in enumerate(self.rating_values):
            rating = str(rating)
            rev_rating = 'rev-%s' % rating
            mod_args[rating] = (self.W_r[rating] if self.W_r is not None else None,)
            mod_args[rev_rating] = (self.W_r[rev_rating] if self.W_r is not None else None,)
        output_feature = self.conv(graph, input_feature, mod_args=mod_args)
        user_feat = output_feature['user']
        item_feat = output_feature['item']
        user_feat = user_feat.view(user_feat.shape[0], -1)
        item_feat = item_feat.view(item_feat.shape[0], -1)

        # fc and non-linear
        user_feat = self.agg_act(user_feat)
        item_feat = self.agg_act(item_feat)
        user_feat = self.dropout(user_feat)
        item_feat = self.dropout(item_feat)
        user_feat = self.ufc(user_feat)
        item_feat = self.ifc(item_feat)
        return self.out_act(user_feat), self.out_act(item_feat)


class Bidecoder(nn.Module):
    def __init__(self, input_units, num_classes, num_basis=2, dropout_rate=0.0):
        super(Bidecoder, self).__init__()
        self.num_basis = num_basis
        self.dropout = nn.Dropout(dropout_rate)
        self.Pl = nn.ParameterList(
            nn.Parameter(torch.randn(input_units, input_units))
            for _ in range(num_basis)
        )
        self.combine_basis = nn.Linear(num_basis, num_classes, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, user_feature, item_feature):
        with graph.local_scope():
            user_feature = self.dropout(user_feature)
            item_feature = self.dropout(item_feature)
            graph.nodes['item'].data['h'] = item_feature
            basis_out = []
            for i in range(self.num_basis):
                graph.nodes['user'].data['h'] = user_feature @ self.Pl[i]
                graph.apply_edges(fn.u_dot_v('h', 'h', 'sr'))
                basis_out.append(graph.edata['sr'])
            out = torch.cat(basis_out, dim=1)
            out = self.combine_basis(out)
        return out


class Net(nn.Module):
    def __init__(self, rating_value, src_dim, dst_dim, gcn_agg_dim, gcn_out_dim, gcn_dropout_rate,
                 gcn_agg_accum, share_user_item_param, device):
        super(Net, self).__init__()
        self.encoder = GCMCLayer(rating_value, src_dim, dst_dim,
                                 gcn_agg_dim, gcn_out_dim, gcn_dropout_rate, gcn_agg_accum,
                                 agg_act='leaky', out_act=None, share_user_item_param=share_user_item_param,
                                 device=device)
        self.encoder.to(device)
        self.decoder = Bidecoder(gcn_out_dim, len(rating_value))
        self.decoder.to(device)

    def forward(self, enc_graph, dec_graph, user_feature, item_feature):
        user_out, item_out = self.encoder(enc_graph, user_feature, item_feature)
        predict_label = self.decoder(dec_graph, user_out, item_out)
        return predict_label


if __name__ == '__main__':
    from dataload import DataLoad
    import os
    from utils.utils import load_subtensor, flatten_etypes

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    cuda = True
    device = torch.device("cuda:1" if cuda else "cpu")
    gcn_agg_dim = 200
    gcn_out_dim = 70
    batch_size = 64
    gcn_dropout_rate = 0.3
    train_grad_clip = 1
    gcn_agg_accum = 'sum'
    share_user_item_param = True
    learning_rate = 1e-4

    dataloader = DataLoad(path='../input/demo/ml-1m_trans.txt', device=device, use_one_hot_fea=True, batch_size=64)
    train_iter = dataloader.data_load('train')
    rating_value = dataloader.dataset.possible_rating_values
    src_dim = dataloader.dataset.user_feature_shape[1]
    dst_dim = dataloader.dataset.item_feature_shape[1]
    # 构建模型
    net = Net(rating_value, src_dim, dst_dim, gcn_agg_dim, gcn_out_dim, gcn_dropout_rate, gcn_agg_accum,
              share_user_item_param, device).to(device)
    print(net)
    rating_loss_net = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), learning_rate)
    dataset = dataloader.dataset
    print("Loading network finished ...\n")
    net.train()
    count_loss = 0
    count = 1
    with tqdm.tqdm(train_iter) as tq:
        for step, (input_nodes, pair_graph, blocks) in enumerate(tq):
            user_feat, item_feat, blocks = load_subtensor(input_nodes, pair_graph, blocks, dataset,
                                                          dataset.train_enc_graph)
            enc_graph = blocks[0].to(device)
            dec_graph = flatten_etypes(pair_graph, dataset, 'train').to(device)
            true_label = dec_graph.edata['label']
            true_rate = dec_graph.edata['rating']
            user_feat = user_feat.to(device)
            item_feat = item_feat.to(device)
            pred_label = net(enc_graph, dec_graph, user_feat, item_feat)
            loss = rating_loss_net(pred_label, true_label.to(device)).mean()
            count_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), train_grad_clip)
            optimizer.step()
            tq.set_postfix({'loss': '{:.4f}'.format(count_loss / count)},
                           refresh=False)
            count += 1
