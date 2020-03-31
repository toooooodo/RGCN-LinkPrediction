import numpy as np
import torch
import dgl
from collections import defaultdict as ddict


def preprocess(triplets_dict):
    sr2o = ddict(set)
    for split in ['train', 'valid', 'test']:
        for subj, rel, obj in triplets_dict[split]:
            sr2o[(subj, rel)].add(obj)
            sr2o[(obj, rel)].add(subj)  # reversed triplets
    sr2o = {k: list(v) for k, v in sr2o.items()}  # {(subj, rel): [obj1, obj2..]...}
    data = ddict(list)
    for split in ['valid', 'test']:
        for subj, rel, obj in triplets_dict[split]:
            data[f'{split}'].append({'triple': (subj, rel, obj), 'label': sr2o[(subj, rel)]})
            data[f'{split}'].append({'triple': (obj, rel, subj), 'label': sr2o[(obj, rel)]})
    data = dict(data)
    return data


def compute_degree_norm(g: dgl.DGLGraph):
    g = g.local_var()
    in_degs = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_degs
    norm[np.isinf(norm)] = 0
    return norm


def build_graph_from_triples(num_nodes, num_rels, triples):
    """
    :param num_nodes: number of nodes in graph
    :param num_rels: number of relations in graph
    :param triples: triples in graphs, (head, rel, dst)
    :return: g: dgl-graph built from triples
             rel: relations in graph [num_rel*2]
             nodes_norm: norm of each nodes in graph [num_nodes]
    """
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    src, rel, dst = triples
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))  # add reversed relations
    rel = np.concatenate((rel, rel + num_rels))
    edges = sorted(zip(dst, src, rel))  # [(dst, src, rel), (), ()...]
    dst, src, rel = np.array(edges).transpose()  # [rel_num*2]
    g.add_edges(src, dst)
    nodes_norm = compute_degree_norm(g)  # 1./in-degree
    return g, rel.astype('int64'), nodes_norm.astype('float32')


def build_graph(num_nodes, num_rels, edges):
    """
    :param num_nodes:
    :param num_rels:
    :param edges: [E, 3]
    :return:
    """
    src, rel, dst = edges.transpose()
    return build_graph_from_triples(num_nodes, num_rels, triples=(src, rel, dst))


def node_norm_2_edge_norm(g: dgl.DGLGraph, node_norm):
    g = g.local_var()
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    return g.edata['norm']


def get_adj(num_nodes, triplets):
    adj_list = [[] for _ in range(num_nodes)]
    """
    [ [ [], []..]
      [ [], []..]
      [ [], []..]]
    """
    for i, triple in enumerate(triplets):
        src, dst = triple[0], triple[2]
        # both directions have same id
        adj_list[src].append([i, dst])  # [edge_id, dst]
        adj_list[dst].append([i, src])

    adj_list = [np.array(n) for n in adj_list]
    return adj_list


def negative_sampling(pos_samples, num_entity, negative_rate):
    """
    :param pos_samples: positive triplets [positive num, 3]
    :param num_entity: total number of entities
    :param negative_rate:
    :return: samples: positive&negative triples [pos+neg num, 3]
             labels: labels of triples sampled, 0 or 1
    """
    batch_size = len(pos_samples)
    generate_num = batch_size * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))  # [generate_num, 3]
    labels = np.zeros(batch_size * (negative_rate + 1), dtype=np.float32)
    labels[:batch_size] = 1
    values = np.random.randint(num_entity, size=generate_num)
    choices = np.random.uniform(size=generate_num)
    sub = choices > 0.5
    obj = choices <= 0.5
    # randomly replace sbj or obj
    neg_samples[sub, 0] = values[sub]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels


def sample_edge_uniform(n_triplets, sample_size):
    """
    Sample edges uniformly from all edges.
    :return:
    """
    all_edges = np.arange(n_triplets)
    return np.random.choice(all_edges, sample_size, replace=False)


def sample_edge_neighborhood(adj_list, degrees, n_triplets, sample_size):
    """
    Sampling edges and nodes diffused from one node
    :param adj_list:
    :param degrees:
    :param n_triplets:
    :param sample_size:
    :return:
    """
    edges = np.zeros((sample_size,), dtype=np.int32)

    sample_counts = np.array([d for d in degrees])
    picked = np.array([False for _ in range(n_triplets)])
    seen = np.array([False for _ in degrees])

    for i in range(sample_size):
        weights = sample_counts * seen

        if np.sum(weights) == 0:
            # all nodes are unseen, pick one node uniformly
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0

        p = weights / np.sum(weights)
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]), p=p)
        chosen_adj_list = adj_list[chosen_vertex]

        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))  # chose one edge linked to chosen vertex
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]

        while picked[edge_number]:
            # this edge is already picked
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

        edges[i] = edge_number  # pick this edge
        other_vertex = chosen_edge[1]  # another nodes on this edge
        picked[edge_number] = True  # this edge is picked
        sample_counts[chosen_vertex] -= 1  # in-degree of chosen-vertex minus one (this edge is deleted from graph
        sample_counts[other_vertex] -= 1
        seen[chosen_vertex] = True
        seen[other_vertex] = True

    return edges


def generate_sampled_graph_and_labels(triplets, sample_size, split_size, num_rels, adj_list, degrees, negative_rate,
                                      sampler='uniform'):
    """
    :param triplets: source triplets
    :param sample_size: number of triplets to sample
    :param split_size: portion of edges used as positive sample
    :param num_rels:
    :param adj_list:
    :param degrees: degree of each node
    :param negative_rate: number of negative samples per positive sample
    :param sampler: type of sampler
    :return: g: dgl-graph built from triples
             uniq_v: unique vertex id in graph
             rel: relations in graph [num_rel*2]
             nodes_norm: norm of each nodes in graph [num_nodes]
             samples: positive&negative triples [pos+neg num, 3]
             labels: labels of triples sampled, 0 or 1
    """
    # perform edge neighbor sampling
    if sampler == "uniform":
        edges = sample_edge_uniform(len(triplets), sample_size)  # edge id
    elif sampler == "neighbor":
        edges = sample_edge_neighborhood(adj_list, degrees, len(triplets), sample_size)
    else:
        raise ValueError("Sampler type must be either 'uniform' or 'neighbor'.")

    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    # edges: used to re-number nodes
    uniq_v, edges = np.unique([src, dst], return_inverse=True)
    # relabel nodes to have consecutive node ids
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack([src, rel, dst]).transpose()  # [sample_size, 3]

    # negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_v), negative_rate)

    # further split graph, only part of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples

    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size), size=split_size)

    src = src[graph_split_ids]
    dst = dst[graph_split_ids]
    rel = rel[graph_split_ids]

    g, rel, norm = build_graph_from_triples(len(uniq_v), num_rels, (src, rel, dst))

    return g, uniq_v, rel, norm, samples, labels
