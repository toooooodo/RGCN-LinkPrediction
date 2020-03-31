import argparse
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
import random
from dgl.contrib.data import load_data
from pprint import pprint
from pathlib import Path

from utils import build_graph, node_norm_2_edge_norm, get_adj, generate_sampled_graph_and_labels, preprocess
from model import LinkPredict
from dataset import TestDataset


class Runner(object):
    def __init__(self, params):
        self.p = params
        self.prj_path = Path(__file__).parent.resolve()
        self.time_stamp = time.strftime('%Y_%m_%d') + '_' + time.strftime('%H:%M:%S')
        self.data = load_data(self.p.dataset)
        self.num_nodes, self.train_data, self.valid_data, self.test_data, self.num_rels = self.data.num_nodes, self.data.train, self.data.valid, self.data.test, self.data.num_rels

        if torch.cuda.is_available() and params.gpu >= 0:
            self.device = torch.device(f'cuda:{params.gpu}')
        else:
            self.device = torch.device('cpu')

        self.val_test_data = preprocess({'train': self.train_data, 'valid': self.valid_data, 'test': self.test_data})
        self.data_iter = self.get_data_iter()

        # self.rel: relations in train set
        self.graph, self.rel, node_norm = build_graph(num_nodes=self.num_nodes, num_rels=self.num_rels,
                                                      edges=self.train_data)
        self.rel = torch.from_numpy(self.rel).to(self.device)
        # used to sample sub-graph
        self.in_deg = self.graph.in_degrees(range(self.graph.number_of_nodes())).float().view(-1, 1)
        self.test_node_id = torch.arange(0, self.num_nodes, dtype=torch.long).view(-1, 1).to(self.device)
        self.test_edge_norm = node_norm_2_edge_norm(self.graph, torch.from_numpy(node_norm).view(-1, 1)).to(self.device)
        self.adj_list = get_adj(self.num_nodes, self.train_data)

        self.model = self.get_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr)

    def get_data_iter(self):
        def get_data_loader(split, batch_size, shuffle=True):
            return DataLoader(
                TestDataset(self.val_test_data[split], self.num_nodes),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=8
            )

        return {
            'valid': get_data_loader('valid', self.p.eval_batch_size),
            'test': get_data_loader('test', self.p.eval_batch_size)
        }

    def get_model(self):
        return LinkPredict(num_nodes=self.num_nodes,
                           h_dim=self.p.n_hidden,
                           num_rels=self.num_rels,
                           num_bases=self.p.n_bases,
                           num_hidden_layers=self.p.n_layers,
                           dropout=self.p.dropout,
                           reg_param=self.p.regularization).to(self.device)

    def fit(self):
        best_val_mrr, best_epoch = -1, 0
        save_root_path = self.prj_path / self.p.save_path
        if not save_root_path.exists():
            save_root_path.mkdir()
        save_path = save_root_path / (self.time_stamp + '.pt')
        for epoch in range(1, 1 + self.p.n_epochs):
            loss = self.train()
            if epoch % self.p.evaluate_every == 0:
                val_mrr, hits_dict = self.evaluate(split='valid')
                if val_mrr > best_val_mrr:
                    best_val_mrr = val_mrr
                    best_epoch = epoch
                    self.save_model(save_path)
                print(
                    f"Epoch: {epoch}, Train Loss: {loss:.6f}, Valid MRR: {val_mrr:.6f}, Best Valid MRR: {best_val_mrr:.6f}")
                for key, value in hits_dict.items():
                    print(f"Hits @ {key} = {value:.6f}")
        val_mrr, hits_dict = self.evaluate(split='valid')
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_epoch = 1 + self.p.n_epochs
            self.save_model(save_path)
        print(
            f"Final Valid MRR: {val_mrr:.6f}, Best Valid MRR: {best_val_mrr:.6f}")
        for key, value in hits_dict.items():
            print(f"Hits @ {key} = {value:.6f}")

        self.load_model(save_path)
        test_mrr, hits_dict = self.evaluate(split='test')
        print(f"Using epoch {best_epoch} with best Valid MRR: {best_val_mrr:.6f}, Test MRR: {test_mrr:.6f}")
        for key, value in hits_dict.items():
            print(f"Hits @ {key} = {value:.6f}")

    def save_model(self, path):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, path)

    def load_model(self, path):
        state = torch.load(path)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])

    def train(self):
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

    def evaluate(self, split='valid'):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.graph, self.test_node_id, self.rel, self.test_edge_norm)  # [ent_num, dim]
        mrr, hits_dict = self.calc_mrr(output, split, hits=[1, 3, 10], filtered=self.p.filtered)
        return mrr, hits_dict

    def calc_mrr(self, output, split, hits, filtered=True):
        ranks = []
        for triple, label in iter(self.data_iter[split]):
            triple, label = triple.to(self.device), label.to(self.device)
            ranks.append(self.model.get_rank(output, triple[:, 0], triple[:, 1], triple[:, 2], label, filtered))
        rank = torch.cat(ranks)
        mrr = torch.mean(1. / rank.float())
        hits_dict = dict()
        for hit in hits:
            avg_count = torch.mean((rank <= hit).float())
            hits_dict[hit] = avg_count

        return mrr.item(), hits_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--random_seed", type=int, default=12345)
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
    parser.add_argument("--evaluate-every", type=int, default=20,
                        help="perform evaluation every n epochs")
    parser.add_argument("--edge-sampler", type=str, default="uniform",
                        help="type of edge sampler: 'uniform' or 'neighbor'")
    parser.add_argument("--save-path", type=str, default='checkpoints')
    parser.add_argument("--filtered", dest='filtered', action='store_true')
    parser.add_argument("--raw", dest='filtered', action='store_false')
    parser.set_defaults(filtered=True)

    args = parser.parse_args()
    pprint(vars(args))

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    runner = Runner(args)
    runner.fit()
