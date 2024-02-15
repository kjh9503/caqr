import pickle
from collections import defaultdict
import torch
import numpy as np
import random

def construct_rel_ht(data_path, sample=0):
    train_file = 'train.txt'
    if 'NELL' in data_path.split('/')[-1] :
        remapped=True
    elif 'FB15k' in data_path.split('/')[-1] :
        remapped=False
        train_file = 'fb237_train.txt'
    else :
        raise ValueError('Invalid dataset')

    with open('%s/id2ent.pkl'% data_path, 'rb') as f :
        ind2ent = pickle.load(f)

    with open('%s/id2rel.pkl'% data_path, 'rb') as f :
        ind2rel = pickle.load(f)

    with open('%s/%s'% (data_path, train_file), 'r') as f :
        train = f.readlines()

    print('size of knowledge graph : ', len(train))

    rel2ind = {pair[1] : pair[0] for pair in ind2rel.items()}
    ent2ind = {pair[1] : pair[0] for pair in ind2ent.items()}
    nrelations = len(rel2ind)
    nentities = len(ent2ind)

    rel_h = [set() for _ in range(nrelations)] #kg_triple에서 r의 head가 되는 애들
    rel_t = [set() for _ in range(nrelations)] #kg_triple에서 r의 tail이 되는 애들
    for line in train :
        h, r, t = line.strip().split('\t')
        if not remapped :
            h, r, t = ent2ind[h], rel2ind[r], ent2ind[t]
        else :
            h, r, t = list(map(int, [h, r, t]))
        rel_h[r].add(h)
        rel_t[r].add(t)
    rel_h_freq = [len(rel_h[i]) for i in range(nrelations)]
    rel_t_freq = [len(rel_t[i]) for i in range(nrelations)]
    rel_h_max_freq = max(rel_h_freq)
    rel_t_max_freq = max(rel_t_freq)
    print('rel head map max freq : ', rel_h_max_freq)
    print('rel tail map max freq : ', rel_t_max_freq)
    print('rel head map median freq : ', np.median(rel_h_freq))
    print('rel tail map median freq : ', np.median(rel_t_freq))
    if not sample :
        rel_h_map = [list(rel_h[i]) + [nentities for _ in range(rel_h_max_freq - len(rel_h[i]))] for i in range(nrelations)]
        rel_t_map = [list(rel_t[i]) + [nentities for _ in range(rel_t_max_freq - len(rel_t[i]))] for i in range(nrelations)]
        rel_h_mask = [[False for _ in range(len(rel_h[i]))] + \
                      [True for _ in range(rel_h_max_freq - len(rel_h[i]))] for i in range(nrelations)]
        rel_t_mask = [[False for _ in range(len(rel_t[i]))] + \
                      [True for _ in range(rel_t_max_freq - len(rel_t[i]))] for i in range(nrelations)]

    else :
        rel_h_map = [list(random.sample(list(rel_h[i]), min(sample, len(rel_h[i])))) + [nentities for _ in range(sample - len(rel_h[i]))] for i in range(nrelations)]
        rel_t_map = [list(random.sample(list(rel_t[i]), min(sample, len(rel_t[i])))) + [nentities for _ in range(sample - len(rel_t[i]))] for i in range(nrelations)]
        rel_h_mask = [[False for _ in range(min(len(rel_h[i]), sample))] + \
                      [True for _ in range(sample - len(rel_h[i]))] for i in range(nrelations)]
        rel_t_mask = [[False for _ in range(min(len(rel_t[i]), sample))] + \
                      [True for _ in range(sample - len(rel_t[i]))] for i in range(nrelations)]

    return np.array(rel_h_map), np.array(rel_t_map), np.array(rel_h_freq), np.array(rel_t_freq), np.array(rel_h_mask), np.array(rel_t_mask)




if __name__ == '__main__' :
    datasets = './data/FB15k-237-betae'
    rel_hmap, rel_tmap, rel_h_degree, rel_t_degree, rel_h_mask, rel_t_mask = construct_rel_ht(datasets, 120)
    print('rel_hmap : ', rel_hmap.shape)
    print('rel_tmap : ', rel_tmap.shape)
    print('rel_h_degree : ', rel_h_degree.shape)
    print('rel_t_degree : ', rel_t_degree.shape)
    print('rel_h_mask : ', rel_h_mask.shape)
    print('rel_t_mask : ', rel_t_mask.shape)

    print(np.min(np.sum(rel_h_mask, axis=1)))
