from torch_geometric.transforms import BaseTransform
import torch.nn.functional as F
import torch

from source.utils import positionalencoding1d


class NodePositionalEncoding(BaseTransform):

    def __init__(self, position_index: list, encoding_length: int):
        '''
        Replaces the position index of a node feature with sinusidial position encoding of request length.
        Assumes the position index is placed as specified at position_index.
        '''
        # where is the position number located in the feature
        self.position_index = position_index
        self.pos_encoding = positionalencoding1d(encoding_length, 100)

    def __call__(self, data):
        for (node_type, index) in self.position_index:
            pos = data[node_type].x[:, index].long()
            encoding = self.pos_encoding[pos]
            if data[node_type].x.shape[1] == index + 1:
                feature = [data[node_type].x[:, :index], encoding]
            else:
                feature = [data[node_type].x[:, :index], encoding, data[node_type].x[:, (index + 1):]]
            data[node_type].x = torch.cat(feature, dim=1)

        return data

class RemovePositionalEncoding(BaseTransform):

    def __init__(self, position_index: list):
        '''
        Removes the part id field.
        '''
        # where is the position number located in the feature
        self.position_index = position_index

    def __call__(self, data):
        for (node_type, index) in self.position_index:
            if data[node_type].x.shape[1] == index + 1:
                feature = [data[node_type].x[:, :index]]
            else:
                feature = [data[node_type].x[:, :index], data[node_type].x[:, (index + 1):]]
            data[node_type].x = torch.cat(feature, dim=1)

        return data

class RandomPositionalEncoding(BaseTransform):

    def __init__(self, position_index: list, encoding_length: int):
        '''
        Replaces the position index of a node feature with a random vector.
        '''
        # where is the position number located in the feature
        self.position_index = position_index
        self.random_length = encoding_length

    def __call__(self, data):
        for (node_type, index) in self.position_index:
            n = data[node_type].x.shape[0]
            rand = torch.rand((n, self.random_length))
            if data[node_type].x.shape[1] == index + 1:
                feature = [data[node_type].x[:, :index], rand]
            else:
                feature = [data[node_type].x[:, :index], rand, data[node_type].x[:, (index + 1):]]
            data[node_type].x = torch.cat(feature, dim=1)

        return data


class NodeOneHot(BaseTransform):

    def __init__(self, position_index: list, one_hot_max_val=5):
        '''
        Replaces a 1d node feature located in position_index with its one-hot encoding
        '''
        # where is the position number located in the feature
        self.position_index = position_index
        self.one_hot = F.one_hot(torch.arange(0, one_hot_max_val))

    def __call__(self, data):
        for (node_type, index) in self.position_index:
            pos = data[node_type].x[:, index].long()
            encoding = self.one_hot.rep[pos]
            if data[node_type].x.shape[1] == index + 1:
                feature = [data[node_type].x[:, :index], encoding]
            else:
                feature = [data[node_type].x[:, :index], encoding, data[node_type].x[:, (index + 1):]]
            data[node_type].x = torch.cat(feature, dim=1)

        return data


class EdgeNormalizationByParts(BaseTransform):

    def __init__(self):
        '''
        Normalize surface to surface edges by the number of parts.
        '''

    def __call__(self, data):
        nof_parts = torch.tensor(data['part'].x.shape[0])
        data['surface', 'dist', 'surface'].edge_weight /= nof_parts.pow(-0.5)

        return data


class EdgeStandardization(BaseTransform):

    def __init__(self, edge, mean, std):
        '''
        Standartize edges using mean and std
        '''
        self.edge = edge
        self.mean = mean
        self.std = std

    def __call__(self, data):

        value = data[self.edge].edge_weight
        value = (value - self.mean) / self.std
        data[self.edge].edge_weight = value

        return data


class EdgeNormalization(BaseTransform):

    def __init__(self, edge, min, max):
        '''
        Normalize edges to be between 0 and 1
        '''
        self.edge = edge
        self.min = min
        self.max_minus_min = max - min

    def __call__(self, data):

        value = data[self.edge].edge_weight
        value = (value - self.min) / self.max_minus_min
        data[self.edge].edge_weight = value

        return data


class PermuteNodeOrder(BaseTransform):

    def __init__(self, position_index: list, encoding_length: int):
        '''
        Randomly permute the order of the selected node type
        '''
        # where is the position number located in the feature
        self.position_index = position_index
        self.encoding_length = encoding_length

    def __call__(self, data):
        for (node_type, index) in self.position_index:
            nof_nodes = data[node_type].x.shape[0]
            permute = torch.randperm(nof_nodes)       
            data[node_type].x[:, index:] = data[node_type].x[permute, index:]
        
        return data


class FilterNumberParts():
    def __init__(self, nof_parts):
        self.nof_parts = nof_parts
    
    def __call__(self, data):
        if self.nof_parts is None or len(self.nof_parts) == 0:
            return True
        
        return data['part'].x.shape[0] in self.nof_parts
