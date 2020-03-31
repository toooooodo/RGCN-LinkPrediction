import torch
from dgl.nn.pytorch import RelGraphConv
import torch.nn.functional as F


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, h_dim)

    def forward(self, g, h, r, norm):
        return self.embedding(h.squeeze())


class BaseRGCN(torch.nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases, hidden_layers=1, dropout=0):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.hidden_layers = hidden_layers
        self.dropout = dropout

        self.layers = self.build_model()

    def build_model(self):
        layers = torch.nn.ModuleList()

        input_layer = self.build_input_layer()
        if input_layer is not None:
            layers.append(input_layer)

        for index in range(self.hidden_layers):
            hidden_layer = self.build_hidden_layer(index)
            layers.append(hidden_layer)

        output_layer = self.build_output_layer()
        if output_layer is not None:
            layers.append(output_layer)

        return layers

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, index):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g, h, r, norm):
        for layer in self.layers:
            h = layer(g, h, r, norm)
        return h


class RGCN(BaseRGCN):
    def build_input_layer(self):
        return EmbeddingLayer(self.num_nodes, self.h_dim)

    def build_hidden_layer(self, index):
        act = F.relu if index < self.hidden_layers - 1 else None
        return RelGraphConv(in_feat=self.h_dim,
                            out_feat=self.h_dim,
                            num_rels=self.num_rels,
                            regularizer='bdd',
                            num_bases=self.num_bases,
                            activation=act,
                            self_loop=True,
                            dropout=self.dropout)


class LinkPredict(torch.nn.Module):
    def __init__(self, num_nodes, h_dim, num_rels, num_bases=-1, num_hidden_layers=1, dropout=0, reg_param=0):
        super(LinkPredict, self).__init__()
        self.rgcn = RGCN(num_nodes=num_nodes,
                         h_dim=h_dim,
                         out_dim=h_dim,
                         num_rels=num_rels * 2,
                         num_bases=num_bases,
                         hidden_layers=num_hidden_layers,
                         dropout=dropout)
        self.reg_param = reg_param
        self.w_relation = torch.nn.Parameter(torch.Tensor(num_rels, h_dim))
        torch.nn.init.xavier_uniform_(self.w_relation, gain=torch.nn.init.calculate_gain('relu'))

    def forward(self, g, h, r, norm):
        """
        :param g: the graph
        :param h: input node ids [v, 1]
        :param r: edge type tensor [e]
        :param norm: edge normalizer tensor [e, 1]
        :return: new node features [v, d]
        """
        return self.rgcn(g, h, r, norm)

    def calc_loss(self, output, triplets, labels):
        """
        :param output: output features of each node
        :param triplets: triplets is a list of data samples (positive and negative)
        :param labels: labels indicating positive or negative
        :return:
        """
        scores = self.calc_score(output, triplets)
        return F.binary_cross_entropy_with_logits(scores, labels) + self.reg_param * (
                    torch.mean(output.pow(2)) + torch.mean(self.w_relation.pow(2)))

    def calc_score(self, output, triplets):
        sub = output[triplets[:, 0]]  # [triple num, dim]
        obj = output[triplets[:, 2]]  # [triple num, dim]
        r = self.w_relation[triplets[:, 1]]  # [triple num, dim]
        # DistMult: sub.T@diag(r)@obj
        score = torch.sum(sub * r * obj, dim=1)  # [triple num]
        return score

    def get_rank(self, output, subj, rel, obj, label, filtered=True):
        """
        calculate ranks of predictions in a mini-batch
        :param output: embedding of each entity: [num_ent, dim]
        :param subj: subject id [batch-size]
        :param rel: relation id
        :param obj: object id
        :param label: indicate valid tails corresponding to head and relation pairs [batch-size, num-ent]
        :param filtered: weather filtered
        :return: rank: [batch-size]
        """
        batch_obj = output[subj] * self.w_relation[rel]
        score = batch_obj @ output.transpose(0, 1)  # [batch_size, dim] @ [dim, entity_num]
        score = torch.sigmoid(score)
        batch_range = torch.arange(score.shape[0])
        if filtered:
            target_score = score[batch_range, obj]
            score = torch.where(label.byte(), torch.zeros_like(score).to(score.device),
                                score)  # filter out other objects with same sub&rel pair
            score[batch_range, obj] = target_score
        rank = 1 + torch.argsort(torch.argsort(score, dim=1, descending=True), dim=1, descending=False)[
            batch_range, obj]
        return rank
