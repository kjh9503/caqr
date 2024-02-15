import torch

structure2name = {('e',('r',)): '1p',
                    ('e', ('r', 'r')): '2p',
                    ('e', ('r', 'r', 'r')): '3p',
                    (('e', ('r',)), ('e', ('r',))): '2i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                    ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                    (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                    (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                    (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                    (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                    (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                }

name2structure = {pair[1]:pair[0] for pair in list(structure2name.items())}

qtype2entind = {'1p' : [0], '2p' : [0], '3p' : [0], '2i' : [0,2], '3i' : [0,2,4], 'ip' : [0,2],
                'pi' : [0,3], '2in' : [0,2], '3in' : [0,2,4], 'inp' : [0,2], 'pin' : [0,3],
                'pni' : [0,4], '2u-DNF' : [0,2], 'up-DNF' : [0,2], '2u-DM' : [0,3], 'up-DM' : [0,3]}

qtype2relind = {'1p' : [1], '2p' : [1], '3p' : [1], '2i' : [1,3], '3i' : [1,3,5], 'ip' : [1,3],
                'pi' : [1,4], '2in' : [1,3], '3in' : [1,3,5], 'inp' : [1,3], 'pin' : [1,4],
                'pni' : [1,5], '2u-DNF' : [1,3], 'up-DNF' : [1,3], '2u-DM' : [1,4],
                'up-DM' : [1,4]}

query2table = {'1p' : torch.FloatTensor([[1,0,0,0],[0,0,0,0],[0,1,0,0]]),
               '2p' : torch.FloatTensor([[1,0,0,0],[0,1,0,0],[0,0,1,0]]),
               '3p' : torch.FloatTensor([[1,0,0,0],[0,1,1,0],[0,0,0,1]]),
               '2i' : torch.FloatTensor([[2,0,0,0],[0,0,0,0],[0,1,0,0]]),
               '3i' : torch.FloatTensor([[3,0,0,0],[0,0,0,0],[0,1,0,0]]),
               'ip' : torch.FloatTensor([[2,0,0,0],[0,1,0,0],[0,0,1,0]]),
               'pi' : torch.FloatTensor([[1,1,0,0],[0,1,0,0],[0,0,1,0]]),
               '2in' : torch.FloatTensor([[2,0,0,0],[0,0,0,0],[0,1,0,0]]),
               '3in' : torch.FloatTensor([[3,0,0,0],[0,0,0,0],[0,1,0,0]]),
               'inp' : torch.FloatTensor([[2,0,0,0],[0,1,0,0],[0,0,1,0]]),
               'pin' : torch.FloatTensor([[1,1,0,0],[0,1,0,0],[0,0,1,0]]),
               'pni' : torch.FloatTensor([[1,1,0,0],[0,1,0,0],[0,0,1,0]]),
               '2u-DNF' : torch.FloatTensor([[2,0,0,0],[0,0,0,0],[0,1,0,0]]),
               '2u-DM' : torch.FloatTensor([[2,0,0,0],[0,0,0,0],[0,1,0,0]]),
               'up-DNF' : torch.FloatTensor([[2,0,0,0],[0,1,0,0],[0,0,1,0]]),
               'up-DM' : torch.FloatTensor([[2,0,0,0],[0,1,0,0],[0,0,1,0]])}

global_maximum = max([tensor.sum() for tensor in query2table.values()])

query2table_norm = {'1p' : query2table['1p'] / global_maximum,
                    '2p' : query2table['2p'] / global_maximum,
                    '3p' : query2table['3p'] / global_maximum,
                    '2i' : query2table['2i'] / global_maximum,
                    '3i' : query2table['3i'] / global_maximum,
                    'ip' : query2table['ip'] / global_maximum,
                    'pi' : query2table['pi'] / global_maximum,
                    '2in' : query2table['2in'] / global_maximum,
                    '3in' : query2table['3in'] / global_maximum,
                    'inp' : query2table['inp'] / global_maximum,
                    'pin' : query2table['pin'] / global_maximum,
                    'pni' : query2table['pni'] / global_maximum,
                    '2u-DNF' : query2table['2u-DNF'] / global_maximum,
                    '2u-DM' : query2table['2u-DM'] / global_maximum,
                    'up-DNF' : query2table['up-DNF'] / global_maximum,
                    'up-DM' : query2table['up-DM'] / global_maximum,
                    }

task2len = {'1p' : 2, '2p' : 3, '3p' : 4, '2i' : 2, '3i' : 2, 'ip' : 3, 'pi' : 3,
            '2in' : 2, '3in' : 2, 'inp' : 3, 'pni' : 3, 'pin' : 3, '2u-DM' : 2, '2u-DNF' : 2,
            'up-DM' : 3, 'up-DNF' : 3}


def get_query2onehot(device='cpu', normalization=False):
    #tasks : ['1c', '2c', '3c', ...]
    table = query2table_norm if normalization else query2table
    query2onehot = {qtype : table[qtype].transpose(0, 1).flatten().to(device) for qtype in table}
    return query2onehot

if __name__ == '__main__' :
    import torch.nn as nn

    query_one_hot = get_query2onehot('cpu', normalization=True)

    embedding_range = 32 / 800
    angle_scale = lambda x : x / embedding_range
    q2o = get_query2onehot('cpu', normalization=True)
    pos = nn.Parameter(torch.zeros((len(query2table.keys()), 108)))
    nn.init.uniform_(pos, -embedding_range, embedding_range)
    lin_pos = nn.Linear(108, 800)

    strc_axis = nn.Linear(12, 108)
    onehot = torch.cat([query_one_hot[qtype].unsqueeze(0) for qtype in query2table.keys()], dim=0)
    strc = strc_axis(onehot)

    print(pos.sum(dim=1), strc.sum(dim=1))


