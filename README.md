# CaQR
This repository contains Pytorch-based code implementing of paper "Improving Multi-hop Logical Reasoning in Knowledge Graphs with Context-Aware Query Representation Learning (CaQR)"([paper]).

> Improving Multi-hop Logical Reasoning in Knowledge Graphs with Context-Aware Query Representation Learning.\
> Jeonghoon Kim, Heesoo Jung, Hyeju Jang, and Hogun Park\
> Accepted at _Findings of the Association for Computational Linguistics, 2024_

Our CaQR can be applied to query embedding-based methodologies such as [Q2B], [BetaE], and [ConE].

This repository is based on KGReasoning([repo]).

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
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.946",
    pages = "15978--15991",
    abstract = "Multi-hop logical reasoning on knowledge graphs is a pivotal task in natural language processing, with numerous approaches aiming to answer First-Order Logic (FOL) queries. Recent geometry (e.g., box, cone) and probability (e.g., beta distribution)-based methodologies have effectively addressed complex FOL queries. However, a common challenge across these methods lies in determining accurate geometric bounds or probability parameters for these queries. The challenge arises because existing methods rely on linear sequential operations within their computation graphs, overlooking the logical structure of the query and the relation-induced information that can be gleaned from the relations of the query, which we call the context of the query. To address the problem, we propose a model-agnostic methodology that enhances the effectiveness of existing multi-hop logical reasoning approaches by fully integrating the context of the FOL query graph. Our approach distinctively discerns (1) the structural context inherent to the query structure and (2) the relation-induced context unique to each node in the query graph as delineated in the corresponding knowledge graph. This dual-context paradigm helps nodes within a query graph attain refined internal representations throughout the multi-hop reasoning steps. Through experiments on two datasets, our method consistently enhances the three multi-hop reasoning foundation models, achieving performance improvements of up to 19.5{\%}. Our codes are available at https://github.com/kjh9503/caqr.",
}
```






<!--![CaQR applied on _ip_ query.](./fig/caqr.png)-->

[repo]: https://github.com/snap-stanford/KGReasoning
[paper]: https://arxiv.org/abs/2406.07034
[Q2B]: https://arxiv.org/abs/2002.05969](https://iclr.cc/virtual_2020/poster_BJgr4kSFDS.html
[BetaE]: https://proceedings.neurips.cc/paper/2020/hash/e43739bba7cdb577e9e3e4e42447f5a5-Abstract.html
[ConE]: https://proceedings.neurips.cc/paper/2021/hash/a0160709701140704575d499c997b6ca-Abstract.html
[the drive]: https://drive.google.com/drive/folders/1wxMPUrAeSjLiZdDbwAsIQnShJYv6bLdU?usp=drive_link
