from torch_scatter import scatter, scatter_mean, scatter_add
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax, degree
import math
from torchsampler import ImbalancedDatasetSampler


class GraphCELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, src, index, target):
        N = index.max().item() + 1

        # https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html
        src_max = scatter(src, index, dim=0, dim_size=N, reduce='max')
        src_max = src_max.index_select(dim=0, index=index)
        out = (src - src_max).exp()

        out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')
        out_sum = out_sum.index_select(dim=0, index=index)

        log_sm = src - src_max - out_sum.log()
        
        out = target * log_sm
        out = -1 * scatter(out, index, dim=0, dim_size=N, reduce='sum')
        out = out.mean()

        return out


class GraphBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, src, index, target):
        nof_graphs = index.max().item() + 1

        out = F.binary_cross_entropy_with_logits(src, target, reduction='none')
        # mean per graph
        out = scatter(out, index, dim=0, dim_size=nof_graphs, reduce='mean')
        # mean per batch
        out = out.mean()

        return out

class GraphMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, src, index, target):
        nof_graphs = index.max().item() + 1

        out = F.mse_loss(src, target, reduction='none')
        # mean per graph
        #out = scatter(out, index, dim=0, dim_size=nof_graphs, reduce='mean')
        # mean per batch
        out = out.mean()

        return out


class RadialDistance(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.mu = torch.nn.Parameter(torch.zeros(dim))
        self.sigma = torch.nn.Parameter(torch.ones(dim))
        self.delta = torch.nn.Parameter(torch.ones(dim))
    
    def forward(self, d):
        out = (-1 * torch.pow(d - self.mu, 2)) / self.sigma
        out = self.delta * torch.exp(out)

        return out


def assembled_regularization(logit, batch):
    nof_graphs = batch['part'].x.shape[0]
    index = batch['part'].batch

    assembled = batch['part'].x[:, 0].reshape(nof_graphs, 1)
    N = index.max().item() + 1

    probs = logit.sigmoid()
    out = assembled * probs
    out = scatter_mean(out, index, dim=0, dim_size=N)
    out = out.mean()

    return out


def entropy_regularization(logit, batch):
    nof_graphs = batch['part'].x.shape[0]
    assembled = batch['part'].x[:, 0].reshape(nof_graphs, 1)
    
    index = batch['part'].batch
    N = index.max().item() + 1

    parts_per_graph = degree(batch['part'].batch).reshape(-1, 1)
 
    probs = logit.sigmoid() + 1e-6
    #probs = ((1 - assembled) * logit.sigmoid()) + 1e-6
    norm = scatter_add(probs, index, dim=0, dim_size=N).index_select(dim=0, index=index)
    norm_probs = probs / norm
 
    #norm_probs = softmax(logit, index=index) + 1e-6
    ent = norm_probs * torch.log(norm_probs)
    entropy = -1 * scatter_add(ent, index, dim=0, dim_size=N) #/ parts_per_graph.log()
    entropy = entropy / entropy.max()
    
    #reg_term = (1 - entropy) / parts_per_graph
    out = (1- entropy).mean()

    return out


def graph_kl_div(logit, target, index):
    N = index.max().item() + 1

    pred_dist = softmax(logit, index=index) + 1e-6
    target_dist = softmax(target, index=index) + 1e-6

    out = pred_dist * (pred_dist / target_dist).log()
    out = scatter_add(out, index, dim=0, dim_size=N)
    out = out.mean()

    return out


def confusion(prediction, truth, batch):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    p = truth.sum()
    n = (1 - truth).sum()
    tp = torch.sum(confusion_vector == 1) 
    fp = torch.sum(confusion_vector == float('inf'))
    tn = torch.sum(torch.isnan(confusion_vector))
    fn = torch.sum(confusion_vector == 0)

    recall = tp / p
    precision = tp / (tp + fp)
    fpr = fp / n
    miss = fn / p

    return recall, precision, fpr, miss


# taken from https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class GraphEdgeNormByParts(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        # first, get for each s_dist_s edge its assigment to a graph index
        src_node_idx = batch.edge_index_dict[('surface', 'dist', 'surface')][0]
        graph_idx_per_node = batch['surface'].batch.index_select(dim=0, index=src_node_idx)

        # get number of parts in graph
        degree_per_graph = degree(batch['part'].batch)

        degree_per_edge = degree_per_graph.index_select(dim=0, index=graph_idx_per_node)

        inv_sqrt_deg = degree_per_edge.pow(-0.5)

        out = batch.edge_weight_dict[('surface', 'dist', 'surface')] * inv_sqrt_deg

        return out


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.surface_gnn:
        m.train()


def update_data_dims(config):
    keys = ["part_input_dim", "surface_input_dim"]

    for k in keys:
        if config['pos_encoding'] or config['random_pos_encoding']:
            config[k] += config['encoding_length'] - 1
        if config['one_hot']:
            config[k] += 3

    return config


class ImbalancedPartsDatasetSampler(ImbalancedDatasetSampler):
    def _get_labels(self, dataset):
        print("Using Imbalanced Sampler...")
        lables = [dataset[i]['part'].x.shape[0] for i in range(len(dataset))]
        
        return lables


def set_missing_config_keys(hparams):
    # walkaround for old models
    for key in ["instance_norm_affine", "dropout", "permute_nodes", "instance_norm_running_stats",
     "remove_pos_encodings", "random_pos_encoding", "permute_surfaces", "permute_parts", 
     "one_hot_positional", "learned_embeddings"]:
        if key not in hparams:
            hparams[key] = 0

    if "architecture" not in hparams:
        hparams["architecture"]= "GAT"
    
    return hparams
