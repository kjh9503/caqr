#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, train_ans, mode):
        assert mode == 'tail-batch'
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples, train_ans)
        self.true_tail = train_ans
        self.qtype = self.triples[0][-1]

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx][0]     # ((0, (0,), 0), '1-chain')에서 (0, (0,), 0)
        head, relations, tail = positive_sample   # head : 0, relations : (0, ), tail : 0 (여기서 tail은 다 0(임시tail)))
        tail = np.random.choice(list(self.true_tail[((head, relations),)]))   # (head, relations)을 조합으로 하는 실제 tail중 sampling
        subsampling_weight = self.count[(head, relations)]   #(head, relations) 조합의 빈도수
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight])) #sqrt 역수 취한 것을 subsampling weight으로
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size: #hyperparam neg_sample_size가 될때까지 반복
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2) #nentity 미만의 숫자중 128*2개의 neg sample 구성
            mask = np.in1d(
                negative_sample,
                self.true_tail[((head, relations),)],     #((0, 0),)에 대한 true tail : {1,6163}
                assume_unique=True, 
                invert=True
            )    #   negative sample에 ``있는 원소가 true_tail에 있는 원소에 있다면, true, 없다면 false인데 inverse이므로  있다면 false
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        positive_sample = torch.LongTensor([head] + [i for i in relations] + [tail])
        return positive_sample, negative_sample, subsampling_weight, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
    @staticmethod
    def count_frequency(triples, true_tail, start=4):
        count = {}
        for triple, qtype in triples:
            head, relations, tail = triple
            assert (head, relations) not in count
            count[(head, relations)] = start + len(true_tail[((head, relations),)])
        return count
    
class TrainInterDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, train_ans, mode):
        assert mode == 'tail-batch'
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples, train_ans)
        self.true_tail = train_ans
        self.qtype = self.triples[0][-1]
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        query = self.triples[idx][:-2]   #((32, (411,)), (6463, (70,)))
        subsampling_weight = self.count[query]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            """
            x = np.array([1,0,2,4,3,1]), y = np.array([1,2])
            np.in1d(x, y) = np.array([True, False, True, False, False, True])
            --> x의 각 원소가 y에 포함되어 있는지 
            """
            mask = np.in1d(
                negative_sample, 
                self.true_tail[query],         # negative_sample array의 원소 중 true_tail에 해당하는게 있는지 : invert되있으므로 없으면 True
                assume_unique=True, 
                invert=True
            )
            negative_sample = negative_sample[mask]   #negative sample로 뽑힌 임의의 entity 중 true_tail인 경우를 제외한 entity들
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        flat_query = np.array([[qi[0], qi[1][0]] for qi in query]).flatten()  #[32, 411, 6463, 70]
        tail = np.random.choice(list(self.true_tail[query]))   #1806 or 2732
        positive_sample = torch.LongTensor(list(flat_query)+[tail])  #[32, 411, 6463, 70, 1806]
        return positive_sample, negative_sample, subsampling_weight, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
    @staticmethod
    def count_frequency(triples, true_tail, start=4):
        count = {}
        for triple in triples:
            query = triple[:-2]
            assert query not in count
            count[query] = start + len(true_tail[query])
        return count

class TestInterDataset(Dataset):
    def __init__(self, triples, test_ans, test_ans_hard, nentity, nrelation, mode):
        self.len = len(triples)
        self.triples = triples
        """
        2i : [((1327, (30,)), (32, (267,)), 0, '2-inter'), ...]
        3i : [((90, (439,)), (2864, (145,)), (2620, (309,)), 0, '3-inter'), ...]
        """
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode
        self.test_ans = test_ans
        self.test_ans_hard = test_ans_hard
        self.qtype = self.triples[0][-1]

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        query = self.triples[idx][:-2]
        tail = self.triples[idx][-2]
        negative_sample = torch.LongTensor(range(self.nentity))
        flat_query = np.array([[qi[0], qi[1][0]] for qi in query]).flatten()
        positive_sample = torch.LongTensor(list(flat_query)+[self.triples[idx][-2]])
        return positive_sample, negative_sample, self.mode, query
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]
        query = data[0][3]
        return positive_sample, negative_sample, mode, query


class TestChainInterDataset(Dataset):
    # tripels : 'test_triples_ci.pkl', test_ans : 'test_ans_ci.pkl', mode = 'tail batch'
    def __init__(self, triples, test_ans, test_ans_hard, nentity, nrelation, mode):
        self.len = len(triples)
        self.triples = triples
        """
        ci : [((8291, (170, 171)), (3724, (39,)), 0, 'chain-inter'), ...]
        """
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode
        self.test_ans = test_ans
        self.test_ans_hard = test_ans_hard
        self.qtype = self.triples[0][-1]

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        query = self.triples[idx][:-2]
        tail = self.triples[idx][-2]
        negative_sample = torch.LongTensor(range(self.nentity))
        positive_sample = torch.LongTensor([query[0][0], query[0][1][0], query[0][1][1], query[1][0], query[1][1][0]]+[self.triples[idx][-2]])
        #  : ((8291, (170, 171)), (3274, (39,))) -> positive sample : [8291, 170, 171, 3724, 39, 0]
        return positive_sample, negative_sample, self.mode, query
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]
        query = data[0][3]
        return positive_sample, negative_sample, mode, query

class TestInterChainDataset(Dataset):
    def __init__(self, triples, test_ans, test_ans_hard, nentity, nrelation, mode):
        self.len = len(triples)
        self.triples = triples
        """
        ic : [((62, (51,)), (2388, (30,)), 381, 0, 'inter-chain'), ...]
        """
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode
        self.test_ans = test_ans
        self.test_ans_hard = test_ans_hard
        self.qtype = self.triples[0][-1]

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        query = self.triples[idx][:-2]
        tail = self.triples[idx][-2]
        negative_sample = torch.LongTensor(range(self.nentity))
        positive_sample = torch.LongTensor([query[0][0], query[0][1][0], query[1][0], query[1][1][0], query[2]]+[self.triples[idx][-2]])
        return positive_sample, negative_sample, self.mode, query
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]
        query = data[0][3]
        return positive_sample, negative_sample, mode, query

class TestDataset(Dataset):
    def __init__(self, triples, test_ans, test_ans_hard, nentity, nrelation, mode):
        self.len = len(triples)
        self.triples = triples
        """
        1c : [((6180, (296,), 0), '1-chain'), ... ]
        2c : [((2865, (5, 63), 0), '2-chain'), ... ]
        """
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode
        self.test_ans = test_ans
        self.test_ans_hard = test_ans_hard
        self.qtype = self.triples[0][-1]

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):

        head, relations, tail = self.triples[idx][0]
        query = ((head, relations),)
        negative_sample = torch.LongTensor(range(self.nentity))
        positive_sample = torch.LongTensor([head] + [rel for rel in relations] + [tail])
        return positive_sample, negative_sample, self.mode, query
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]
        query = data[0][3]
        return positive_sample, negative_sample, mode, query
    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data

class SingledirectionalOneShotIterator(object):
    def __init__(self, dataloader_tail, qtype):
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.qtype = qtype
        self.step = 0
        
    def __next__(self):
        self.step += 1
        data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data


class TrainDatasetQueryWithTargetTriple(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, train_ans, graph_dict, mode):

        # graph_dict = {h : [[r1,t1], [r2,t2], ...]}
        assert mode == 'tail-batch'

        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.graph_dict = graph_dict
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples, train_ans)
        self.true_tail = train_ans
        self.qtype = self.triples[0][-1]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx][0]  # ((0, (0,), 0), '1-chain')  -> (0, (0,), 0)
        head, relations, tail = positive_sample  # h : 0, relations : (0,), tail : 0           (tail은 그냥 placeholder)
        tail = np.random.choice(list(self.true_tail[((head, relations),)]))

        subsampling_weight = self.count[(head, relations)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            mask = np.in1d(
                negative_sample,
                self.true_tail[((head, relations),)],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        positive_sample = torch.LongTensor([head] + [i for i in relations] + [tail])  # 2p 처럼 relation 여러 개인 경우도 있음

        if len(self.graph_dict[tail]) == 0:
            tail = head
        sampled_ind = np.random.choice(len(self.graph_dict[tail]), 1)[0]
        target_triple = torch.LongTensor([tail] + self.graph_dict[tail][sampled_ind])

        return positive_sample, negative_sample, subsampling_weight, target_triple, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        target_triple = torch.stack([_[3] for _ in data], dim=0)
        mode = data[0][4]
        return positive_sample, negative_sample, subsample_weight, target_triple, mode

    @staticmethod
    def count_frequency(triples, true_tail, start=4):
        count = {}
        for triple, qtype in triples:
            head, relations, tail = triple
            assert (head, relations) not in count
            count[(head, relations)] = start + len(true_tail[((head, relations),)])
        return count

if __name__ == '__main__' :
    from torch.utils.data import DataLoader
    tasks = ['2i']
    data_path = '/Users/jeonghoonkim/Desktop/LearnDataLab/query2box_master/data/FB15k-237'
    import pickle

    train_ans, valid_ans, valid_ans_hard, test_ans, test_ans_hard = {}, {}, {}, {}, {}
    with open('%s/stats.txt'%data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])
    if '2i' in tasks:
        with open('%s/train_triples_2i.pkl'%data_path, 'rb') as handle:
            train_triples_2i = pickle.load(handle)
        with open('%s/train_ans_2i.pkl'%data_path, 'rb') as handle:
            train_ans_2i = pickle.load(handle)
        with open('%s/valid_triples_2i.pkl'%data_path, 'rb') as handle:
            valid_triples_2i = pickle.load(handle)
        with open('%s/valid_ans_2i.pkl'%data_path, 'rb') as handle:
            valid_ans_2i = pickle.load(handle)
        with open('%s/valid_ans_2i_hard.pkl'%data_path, 'rb') as handle:
            valid_ans_2i_hard = pickle.load(handle)
        with open('%s/test_triples_2i.pkl'%data_path, 'rb') as handle:
            test_triples_2i = pickle.load(handle)
        with open('%s/test_ans_2i.pkl'%data_path, 'rb') as handle:
            test_ans_2i = pickle.load(handle)
        with open('%s/test_ans_2i_hard.pkl'%data_path, 'rb') as handle:
            test_ans_2i_hard = pickle.load(handle)
        train_ans.update(train_ans_2i)
        valid_ans.update(valid_ans_2i)
        valid_ans_hard.update(valid_ans_2i_hard)
        test_ans.update(test_ans_2i)
        test_ans_hard.update(test_ans_2i_hard)


    # train_dataloader_2i_tail = DataLoader(
    #     TrainInterDataset(train_triples_2i, nentity, nrelation, 7, train_ans, 'tail-batch'),
    #     # TrainInterDataset return : positive_sample, negative_sample, subsampling_weight, self.mode(='tail-batch')
    #     batch_size=4,
    #     shuffle=True,
    #     num_workers=max(1, 5),
    #     collate_fn=TrainInterDataset.collate_fn
    # )
    # train_iterator_2i = SingledirectionalOneShotIterator(train_dataloader_2i_tail, train_triples_2i[0][-1])
    td =TrainInterDataset(train_triples_2i, nentity, nrelation, 7, train_ans, 'tail-batch')
    for i in td[0] :
        print(i)
    # pos, neg, weight, mode = next(train_iterator_2i)
    # print(pos)
    # print(neg)
    # print(weight)
    # print(mode)