## Relational-GCN

An implementation of [RGCN](https://arxiv.org/abs/1703.06103) for Link Prediction task in Pytorch and DGL.

This work is based on [https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn](https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn) and some tricks are added to speed up training and test process.

### Train for link prediction
#### Install dependencies
1. Install Python3
2. Install requirements `pip install -r requirements.txt`

#### Train model
`python run.py --dataset FB15k-237 --filtered --edge-sampler uniform --gpu 0`

- `--dataset` denotes the dataset to use
- `--filtered` denotes the evaluation protocol. `--filtered` or `--raw`
- `edge-sampler` denotes the sampling methods. `uniform` or `neighbor`
- Rest of the arguments can be listed using python run.py -h

### Test Result

| Protocol | MRR | Hits@1 | Hits@3 | Hits@10 | Command |
| --- | --- | --- | --- | --- | --- |
| Filtered | 0.243743 | 0.154622 | 0.263657 | 0.427196 | `python run.py --filtered --gpu 0` |
| Raw | 0.163159 | 0.099262 | 0.165494 | 0.293633 | `python run.py --raw --gpu 0` |