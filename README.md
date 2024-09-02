# CaQR
This repository contains Pytorch-based code implementing of paper "Improving Multi-hop Logical Reasoning in Knowledge Graphs with Context-Aware Query Representation Learning (CaQR)"([paper]).

> Improving Multi-hop Logical Reasoning in Knowledge Graphs with Context-Aware Query Representation Learning.\
> Jeonghoon Kim, Heesoo Jung, Hyeju Jang, and Hogun Park\
> Accepted at _Findings of the Association for Computational Linguistics, 2024_

Our CaQR can be applied to query embedding-based methodologies such as [Q2B], [BetaE], and [ConE].

This repository is based on KGReasoning([repo]).

## Requirements
python==3.9.15
torch==1.13.1
tensorboard==2.11.0
tqdm


## Basic Usage
1. Make _data_ directory in _CaQR_ directory.
2. Download the query dataset from [the drive] and move them in _data_ directory (i.e., 'data/FB15k-237-betae', 'data/NELL-betae').
   
### Q2B
To apply `CaQR` to Q2B, execute the shell file, _q2b_caqr.sh_ in the _CaQR/q2b_ directory:
```bash
bash q2b_caqr.sh
```

### BetaE
To apply `CaQR` to BetaE, execute the shell file, _betae_caqr.sh_ in the _CaQR/betae_ directory:
```bash
bash betae_caqr.sh
```

### ConE
To apply `CaQR` to ConE, execute the shell file, _cone_caqr.sh_ in the _CaQR/cone_ directory:
```bash
bash cone_caqr.sh
```

## Citation
If `CaQR` is helpful in your research, we would appreciate it if you could cite our paper as follows:

```
@inproceedings{kim-etal-2024-improving-multi,
    title = "Improving Multi-hop Logical Reasoning in Knowledge Graphs with Context-Aware Query Representation Learning",
    author = "Kim, Jeonghoon  and
      Jung, Heesoo  and
      Jang, Hyeju  and
      Park, Hogun",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    pages = "15978--15991",
}
```






<!--![CaQR applied on _ip_ query.](./fig/caqr.png)-->

[repo]: https://github.com/snap-stanford/KGReasoning
[paper]: https://arxiv.org/abs/2406.07034
[Q2B]: https://arxiv.org/abs/2002.05969](https://iclr.cc/virtual_2020/poster_BJgr4kSFDS.html
[BetaE]: https://proceedings.neurips.cc/paper/2020/hash/e43739bba7cdb577e9e3e4e42447f5a5-Abstract.html
[ConE]: https://proceedings.neurips.cc/paper/2021/hash/a0160709701140704575d499c997b6ca-Abstract.html
[the drive]: https://drive.google.com/drive/folders/1wxMPUrAeSjLiZdDbwAsIQnShJYv6bLdU?usp=drive_link
