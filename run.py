# -*- coding: utf-8 -*-
# @project：GCMC-Pytorch-dgl
# @author:caojinlei
# @file: run.py
# @time: 2021/07/14
from dataload import DataLoad, DataSet
from models.gcmc import Net
from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error, confusion_matrix
from callback.modelcheckpoint import ModelCheckPoint
from callback.tensorboard_pytorch import net_board, loss_board
import numpy as np
import dgl
import argparse
from torch import nn
import torch
import os
from utils.Logginger import init_logger
from utils.utils import load_subtensor, flatten_etypes

logger = init_logger('gcmc', './output/')


def arguments():
    parser = argparse.ArgumentParser(description='GCMC ml-1m Example')
    parser.add_argument('--module', type=str, default='GCMC', metavar='N',
                        help='Which model to choose(default: GCMC)')
    parser.add_argument('--batch_size', type=int, default=10000, metavar='N',
                        help='input batch size for training (default: 10000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training(default: False)')
    parser.add_argument('--seed', type=int, default=1111, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save', type=str, default='y', metavar='S',
                        help='Whether or not to save(default: y)')
    parser.add_argument('--train', type=str, default='y', metavar='S',
                        help='Train or predict(default: y)')
    parser.add_argument('--log-interval', type=int, default=1024, metavar='N',
                        help='how many batches to wait before logging training status(default: 1024)')
    parser.add_argument('--test_ratio', type=float, default=0.1, metavar='N',
                        help='Proportion of the test set(default: 0.1)')
    parser.add_argument('--valid_ratio', type=float, default=0.05, metavar='N',
                        help='Proportion of the valid set(default: 0.05)')
    parser.add_argument('--gcn_agg_dim', type=int, default=200, metavar='N',
                        help='gcn agg layer dim (default: 200)')
    parser.add_argument('--gcn_out_dim', type=int, default=70, metavar='N',
                        help='gcn out layer dim (default: 70)')
    parser.add_argument('--gcn_dropout_rate', type=float, default=0.5, metavar='N',
                        help='gcn dropout layer rate set(default: 0.5)')
    parser.add_argument('--gcn_agg_accum', type=str, default='sum', metavar='S',
                        help='Gcn agg accum sum or stack (default: sum)')
    parser.add_argument('--train_grad_clip', type=float, default=1., metavar='N',
                        help='Proportion of train gradient clipping set(default: 1.)')
    parser.add_argument('--share_user_item_param', action='store_true', default=False,
                        help='enables share_user_item_param(default: True)')
    parser.add_argument('--use_one_hot_fea', action='store_true', default=False,
                        help='enables share_user_item_param(default: True)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='N',
                        help='learning rate (default:  1e-4)')

    return parser.parse_args()


def train(args, model, train_iter, dataset, optimizer, rating_loss_net, epoch, log, device):
    model.train()
    count = 0
    count_loss = 0
    for step, (input_nodes, pair_graph, blocks) in enumerate(train_iter):
        user_feat, item_feat, blocks = load_subtensor(input_nodes, pair_graph, blocks, dataset,
                                                      dataset.train_enc_graph)
        enc_graph = blocks[0].to(device)
        dec_graph = flatten_etypes(pair_graph, dataset, 'train').to(device)
        true_label = dec_graph.edata['label']
        user_feat = user_feat.to(device)
        item_feat = item_feat.to(device)
        pred_label = model(enc_graph, dec_graph, user_feat, item_feat)
        loss = rating_loss_net(pred_label, true_label.to(device)).mean()
        count_loss += loss.item()
        count += 1
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.train_grad_clip)
        optimizer.step()
        log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\t'.format(
            epoch, step * args.batch_size + len(pred_label), len(train_iter.dataloader.dataset),
                   100 * (step * args.batch_size + len(pred_label)) / len(train_iter.dataloader.dataset),
            loss.item()))
    log.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, count_loss / count))
    result_loss = count_loss / count
    return result_loss


@torch.no_grad()
def evaluate(model, dataset, dataiter, device, rating_loss_net, log, data='valid'):
    if data == 'test':
        model.eval().to(device)
    count_loss = 0
    count = 0
    for input_nodes, pair_graph, blocks in dataiter:
        user_feat, item_feat, blocks = load_subtensor(input_nodes, pair_graph, blocks, dataset,
                                                      dataset.valid_enc_graph if data == 'valid' else dataset.test_enc_graph)
        enc_graph = blocks[0].to(device)
        dec_graph = pair_graph.to(device)
        true_label = dataset.valid_labels[pair_graph.edata[dgl.EID]] if data == 'valid' else \
            dataset.test_labels[pair_graph.edata[dgl.EID]]
        user_feat = user_feat.to(device)
        item_feat = item_feat.to(device)
        pred_label = model(enc_graph, dec_graph, user_feat, item_feat)
        # print('pred_label:{}'.format(pred_label))
        # print('pred_label:{}'.format(torch.softmax(pred_label,dim=1)))
        # l = pred_label.argmax(dim=1)
        # print('pred_label:{}'.format(l))
        # print('true_label:{}'.format(true_label))
        loss = rating_loss_net(pred_label, true_label.to(device)).mean()
        count_loss += loss.item()
        count += 1
    log.info('====>{} Average loss: {:.4f}'.format(data, count_loss / count))
    return count_loss / count


@torch.no_grad()
def predict(model, dataset, dataiter, device, log):
    possible_rating_values = torch.FloatTensor(dataset.possible_rating_values).to(device)
    model.eval().to(device)
    real_pred_ratings = []
    true_rel_ratings = []
    true_y = []
    predict_y = []
    for input_nodes, pair_graph, blocks in dataiter:
        user_feat, item_feat, blocks = load_subtensor(input_nodes, pair_graph, blocks, dataset, dataset.test_enc_graph)
        enc_graph = blocks[0].to(device)
        dec_graph = pair_graph.to(device)
        true_label = dataset.test_labels[pair_graph.edata[dgl.EID]].cpu().tolist()
        true_relation_ratings = dataset.test_truths[pair_graph.edata[dgl.EID]]
        user_feat = user_feat.to(device)
        item_feat = item_feat.to(device)
        pred_label = model(enc_graph, dec_graph, user_feat, item_feat)
        soft_pre = torch.softmax(pred_label, dim=1)
        batch_pred_ratings = (soft_pre * possible_rating_values.view(1, -1)).sum(dim=1)
        pred_label = soft_pre.argmax(dim=1).cpu().tolist()
        real_pred_ratings.append(batch_pred_ratings)
        true_rel_ratings.append(true_relation_ratings)
        true_y.extend(true_label)
        predict_y.extend(pred_label)
    real_pred_ratings = torch.cat(real_pred_ratings, dim=0).to(device)
    true_rel_ratings = torch.cat(true_rel_ratings, dim=0).to(device)
    averages = ["macro",'micro','weighted']
    results = {}
    for average in averages:
        results[average] = f1_score(true_y, predict_y, average=average)
    rmse = ((real_pred_ratings - true_rel_ratings) ** 2.).mean().item()
    auc = roc_auc_score(true_y,predict_y)
    results['rmse'] = np.sqrt(rmse)
    results['auc'] = auc
    cf_matrix = confusion_matrix(true_y, predict_y)
    print(cf_matrix)
    log.info('result: {}'.format(results))


def cmd_entry(args, logger):
    path = 'input/demo/ml-1m_trans3.txt'
    user_json = 'input/demo/ml_1m_user_feature.npy'
    item_json = 'input/demo/ml_1m_item_feature.npy'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda:1" if args.cuda else "cpu")
    logger.info('Loading data started ...')
    dataloader = DataLoad(path=path, user_embed_path=user_json,
                          item_embed_path=item_json, device=device,
                          use_one_hot_fea=args.use_one_hot_fea,
                          batch_size=args.batch_size,
                          test_ratio=args.test_ratio, valid_ratio=args.valid_ratio)
    valid_iter = dataloader.data_load('valid')
    train_iter = dataloader.data_load('train')
    test_iter = dataloader.data_load('test')
    rating_value = dataloader.dataset.possible_rating_values
    src_dim = dataloader.dataset.user_feature_shape[1]
    dst_dim = dataloader.dataset.item_feature_shape[1]
    logger.info('Loading data finished ...')
    # 构建模型
    logger.info('Loading network started ...')
    model = Net(rating_value, src_dim, dst_dim, args.gcn_agg_dim, args.gcn_out_dim, args.gcn_dropout_rate,
                args.gcn_agg_accum,
                args.share_user_item_param, device).to(device)
    print(model)
    rating_loss_net = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    dataset = dataloader.dataset
    logger.info("Loading network finished ...")
    if args.save == 'y':
        save_module = ModelCheckPoint(model=model, optimizer=optimizer, args=args, log=logger)
        state = save_module.save_info(epoch=0)
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, train_iter, dataset, optimizer, rating_loss_net, epoch, logger, device)
        valid_loss = evaluate(model, dataset, valid_iter, device, rating_loss_net, logger, 'valid')
        loss_board('./output/logs', 'train', 'loss', train_loss, valid_loss, epoch)
        if args.save == 'y':
            state = save_module.save_step(state, train_loss)
    predict(model, dataset, test_iter, device, logger)


if __name__ == '__main__':
    args = arguments()
    print(args)
    cmd_entry(args, logger)
