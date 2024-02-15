import torch

query2table = {'1-chain' : torch.FloatTensor([[1,0,0,0],[0,0,0,0],[0,1,0,0]]),
               '2-chain' : torch.FloatTensor([[1,0,0,0],[0,1,0,0],[0,0,1,0]]),
               '3-chain' : torch.FloatTensor([[1,0,0,0],[0,1,1,0],[0,0,0,1]]),
               '2-inter' : torch.FloatTensor([[2,0,0,0],[0,0,0,0],[0,1,0,0]]),
               '3-inter' : torch.FloatTensor([[3,0,0,0],[0,0,0,0],[0,1,0,0]]),
               'inter-chain' : torch.FloatTensor([[2,0,0,0],[0,1,0,0],[0,0,1,0]]),
               'chain-inter' : torch.FloatTensor([[1,1,0,0],[0,1,0,0],[0,0,1,0]]),
               '2-union' : torch.FloatTensor([[2,0,0,0],[0,0,0,0],[0,1,0,0]]),
               'union-chain' : torch.FloatTensor([[2,0,0,0],[0,1,0,0],[0,0,1,0]])}
global_maximum = max([tensor.sum() for tensor in query2table.values()])

query2table_norm = {'1-chain' : query2table['1-chain'] / global_maximum,
                    '2-chain' : query2table['2-chain'] / global_maximum,
                    '3-chain' : query2table['3-chain'] / global_maximum,
                    '2-inter' : query2table['2-inter'] / global_maximum,
                    '3-inter' : query2table['3-inter'] / global_maximum,
                    'chain-inter' : query2table['chain-inter'] / global_maximum,
                    'inter-chain' : query2table['inter-chain'] / global_maximum,
                    '2-union' : query2table['2-union'] / global_maximum,
                    'union-chain' : query2table['union-chain'] / global_maximum,
                    }

task2len = {'1c' : 2, '2c' : 3, '3c' : 4, '2i' : 2, '3i' : 2, 'ic' : 3, 'ci' : 3, '2u' : 2, 'uc' : 3}

task2qtype = {'1c' : '1-chain', '2c' : '2-chain', '3c' : '3-chain', '2i' : '2-inter', '3i' : '3-inter',
              'ic' : 'inter-chain', 'ci' : 'chain-inter', '2u' : '2-union', 'uc' : 'union-chain'}


def get_query2onehot(device='cpu', normalization=False):
    #tasks : ['1c', '2c', '3c', ...]
    table = query2table_norm if normalization else query2table
    query2onehot = {qtype : table[qtype].transpose(0, 1).flatten().to(device) for qtype in table}
    return query2onehot

