# -*- coding: utf-8 -*-
# @project：GCMC-Pytorch-dgl
# @author:caojinlei
# @file: preprocess.py
# @time: 2021/07/14
import numpy as np
from tqdm import tqdm
import json
from collections import Counter


def trans_data(input_path, out_path, user_json, item_json, augmenta=False):
    result = []
    user_list = []
    item_list = []
    rate_list = []
    with open(input_path, 'r') as f:
        for lines in tqdm(f.readlines()):
            line = lines.strip().split(' ')
            user_list.append(line[0])
            item_list.append(line[1])
            rate_list.append(line[2])
            result.append(line)
    global_user_id_map = {ele: i for i, ele in enumerate(list(set(user_list)))}
    global_item_id_map = {ele: i for i, ele in enumerate(list(set(item_list)))}
    with open(user_json, 'w') as f:
        json.dump(global_user_id_map, f)
    with open(item_json, 'w') as f:
        json.dump(global_item_id_map, f)
    if augmenta:
        augment_list= []
        c1 = Counter(rate_list)
        max_count = 0
        max_rate = 0
        for rate, count in c1.items():
            if count >= max_count:
                max_count = count
                max_rate = rate
        rate_dict = {}
        for rate, count in c1.items():
            if rate != max_rate:
                ratio = int(max_count / count)
                rate_dict[rate] = ratio
        for line in result:
            if line[2] != max_rate:
                ratio = rate_dict[line[2]]
                if ratio !=1:
                    line_list = [line]
                    line_list = line_list*int(ratio - 1)
                    augment_list.extend(line_list)
        result.extend(augment_list)
        tran1 = []
        for a, b, c in result:
            tran1.append(c)
        print(Counter(tran1))
    re_np = np.array(result)
    index = [i for i in range(len(re_np))]  # test_data为测试数据
    np.random.shuffle(index)  # 打乱索引
    re_np = re_np[index].tolist()
    with open(out_path, 'w') as f:
        for re in re_np:
            re[0] = str(global_user_id_map[re[0]])
            re[1] = str(global_item_id_map[re[1]])
            f.write(' '.join(re) + '\n')


def construction_feature(input_path, user_npy, item_npy, dim=256):
    user_list = []
    item_list = []
    with open(input_path, 'r') as f:
        for lines in f.readlines():
            line = lines.strip().split(' ')
            user_list.append(line[0])
            item_list.append(line[1])
    global_user_id_map = {ele: i for i, ele in enumerate(list(set(user_list)))}
    global_item_id_map = {ele: i for i, ele in enumerate(list(set(item_list)))}
    user_feature = np.random.normal(0, 1, (len(global_user_id_map), dim))
    item_feature = np.random.normal(0, 1, (len(global_item_id_map), dim))
    np.save(user_npy, user_feature)
    np.save(item_npy, item_feature)


if __name__ == '__main__':
    input_path = './input/demo/ml-1m.txt'
    out_path = './input/demo/ml-1m_trans3.txt'
    user_json = './input/demo/user_idmap.json'
    item_json = './input/demo/item_idmap.json'
    user_npy = './input/ml_1m_user_feature.npy'
    item_npy = './input/ml_1m_item_feature.npy'
    trans_data(input_path,out_path,user_json,item_json,augmenta=False)
    construction_feature(out_path,user_npy,item_npy)
