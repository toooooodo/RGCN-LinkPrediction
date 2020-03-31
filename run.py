import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dgl.contrib.data import load_data

from utils import build_graph, node_norm_2_edge_norm, get_adj, generate_sampled_graph_and_labels
from model import LinkPredict


class Runner(object):
    def __init__(self, params):
        self.p = params
        self.data = load_data(self.p.dataset)
        self.num_nodes, self.train_data, self.valid_data, self.test_data, self.num_rels = self.data.num_nodes, self.data.train, self.data.valid, self.data.test, self.data.num_rels
        if torch.cuda.is_available() and params.gpu >= 0:
            self.device = torch.device(f'cuda:{params.gpu}')
        else:
            self.device = torch.device('cpu')
        # self.rel: relations in train set
        self.graph, self.rel, node_norm = build_graph(num_nodes=self.num_nodes, num_rels=self.num_rels,
                                                      edges=self.train_data)
        self.rel = torch.from_numpy(self.rel).to(self.device)
        self.in_deg = self.graph.in_degrees(range(self.graph.number_of_nodes())).float().view(-1, 1)  # used to sample sub-graph
        self.test_node_id = torch.arange(0, self.num_nodes, dtype=torch.long).view(-1, 1).to(self.device)
        self.test_edge_norm = node_norm_2_edge_norm(self.graph, torch.from_numpy(node_norm).view(-1, 1)).to(self.device)
        self.adj_list = get_adj(self.num_nodes, self.train_data)

        self.valid_data = torch.from_numpy(self.valid_data).to(self.device)
        self.test_data = torch.from_numpy(self.test_data).to(self.device)

        self.model = self.get_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr, weight_decay=self.p.regularization)

    def get_model(self):
        return LinkPredict(num_nodes=self.num_nodes,
                           h_dim=self.p.n_hidden,
                           num_rels=self.num_rels,
                           num_bases=self.p.n_bases,
                           num_hidden_layers=self.p.n_layers,
                           dropout=self.p.dropout,
                           reg_param=self.p.regularization)

    def fit(self):
        for epoch in range(self.p.n_epochs):
            loss = self.train()
            mrr = self.evaluate()
            # print(loss, mrr)

    def train(self):
        self.model.to(self.device)
        self.model.train()
        g, node_id, edge_type, node_norm, data, labels = \
            generate_sampled_graph_and_labels(triplets=self.train_data,
                                              sample_size=self.p.graph_batch_size,
                                              split_size=self.p.graph_split_size,
                                              num_rels=self.num_rels,
                                              adj_list=self.adj_list,
                                              degrees=self.in_deg,
                                              negative_rate=self.p.negative_sample,
                                              sampler=self.p.edge_sampler)
        node_id = torch.from_numpy(node_id).view(-1, 1).long().to(self.device)
        edge_type = torch.from_numpy(edge_type).to(self.device)
        data, labels = torch.from_numpy(data).to(self.device), torch.from_numpy(labels).to(self.device)
        edge_norm = node_norm_2_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1)).to(self.device)
        # deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1).to(self.device)

        output = self.model(g, node_id, edge_type, edge_norm)
        loss = self.model.calc_loss(output, data, labels)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.p.grad_norm)
        self.optimizer.step()

        return loss.item()

    def evaluate(self):
        # self.model.cpu()
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.graph, self.test_node_id, self.rel, self.test_edge_norm)
        mrr = self.calc_mrr(output, torch.from_numpy(self.train_data), self.valid_data, self.test_data, hits=[1, 3, 10],
                            batch_size=self.p.eval_batch_size, eval_p=self.p.eval_protocol)

        return mrr

    def calc_mrr(self, output, train_triplets, valid_triplets, test_triplets, hits, batch_size, eval_p="filtered"):
        if eval_p == "filtered":
            pass
        elif eval_p == "raw":
            return self.calc_raw_mrr(output, test_triplets, hits, batch_size)
        else:
            raise KeyError(f'Evaluate protocol {eval_p} not recognized.')

    def calc_raw_mrr(self, output, triplets, hits, batch_size=100):
        subj = triplets[:, 0]
        rel = triplets[:, 1]
        obj = triplets[:, 2]
        test_size = triplets.shape[0]

        subj_rank = self.model.get_raw_rank(output, subj, rel, obj, test_size, batch_size)
        obj_rank = self.model.get_raw_rank(output, obj, rel, subj, test_size, batch_size)
        rank = torch.cat([subj_rank, obj_rank])

        mrr = torch.mean(1. / rank.float())
        print(f"MRR (raw) = {mrr:.6f}")

        for hit in hits:
            avg_count = torch.mean((rank <= hit).float())
            print(f"Hits (raw) @ {hit} = {avg_count:.6f}")

        return mrr.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=500,
                        help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=6000,
                        help="number of minimum training epochs")
    parser.add_argument("--dataset", type=str, default="FB15k-237",
                        help="dataset to use")
    parser.add_argument("--eval-batch-size", type=int, default=500,
                        help="batch size when evaluating")
    parser.add_argument("--eval-protocol", type=str, default="raw",
                        help="type of evaluation protocol: 'raw' or 'filtered' mrr")
    parser.add_argument("--regularization", type=float, default=0.01,
                        help="regularization weight")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")
    parser.add_argument("--graph-batch-size", type=int, default=30000,
                        help="number of edges to sample in each iteration")
    parser.add_argument("--graph-split-size", type=float, default=0.5,
                        help="portion of edges used as positive sample")
    parser.add_argument("--negative-sample", type=int, default=10,
                        help="number of negative samples per positive sample")
    parser.add_argument("--evaluate-every", type=int, default=500,
                        help="perform evaluation every n epochs")
    parser.add_argument("--edge-sampler", type=str, default="uniform",
                        help="type of edge sampler: 'uniform' or 'neighbor'")

    args = parser.parse_args()
    print(vars(args))
    runner = Runner(args)
    runner.fit()
