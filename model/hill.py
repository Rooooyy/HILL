from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.inits import reset
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, Size, PairTensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool


class GNNEncoder(nn.Module):
    def __init__(self, config):
        super(GNNEncoder, self).__init__()
        self.conv_types = {
            'GCN': GCNConv,
            'GAT': GATConv,
            'GIN': GINConv
        }
        self.pool_types = {
            'sum': global_add_pool,
            'avg': global_mean_pool,
            'max': global_max_pool
        }
        self.device = config.device_setting.device
        self.conv = self.conv_types[config.graph_conv]
        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])

        self.num_layers = config.conv_layers
        for _ in range(self.num_layers):
            if config.graph_conv == 'GIN':
                self.convs.append(self.conv(nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(config.hidden_dim, config.hidden_dim)
                )))
            else:
                self.convs.append(self.conv(config.hidden_dim, config.hidden_dim))
            self.bns.append(nn.BatchNorm1d(config.hidden_dim))

        self.pool = self.pool_types[config.graph_pooling_type]  # pool for graph-level representation.
        self.dim_align = nn.Sequential(
            nn.Dropout(config.hidden_dropout),
            nn.Linear(self.num_layers * config.hidden_dim, config.structure_encoder.output_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(self.device)
        xs = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index.to(self.device))
            x = self.bns[i](F.relu(x))
            xs.append(x)

        xpool =[self.pool(x, batch) for x in xs]
        return self.dim_align(torch.cat(xpool, 1))  # [batch, hidden_dim * num_layers] -> [batch, output_dim]



class GTData(Data):
    '''
    The original Graph along with (coding) Tree Data.
    '''
    def __init__(self, x: OptTensor=None, edge_index: OptTensor=None,
                 edge_attr: OptTensor=None, y: OptTensor=None,
                 pos: OptTensor=None, **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key.find('treeEdgeMatLayer') >= 0:
            layer = int(key.replace('treeEdgeMatLayer', ''))
            return torch.tensor([[self.treeNodeSize[0][layer]],
                                 [self.treeNodeSize[0][layer-1]]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key.find('treeEdgeMatLayer') >= 0:
            return 1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


class HRLConv(MessagePassing):
    def __init__(self, nn: Callable, **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'source_to_target')
        super().__init__(**kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: PairTensor, edge_index: Adj, size: Size = None) -> Tensor:
        out = self.propagate(edge_index, x=x, size=size)
        return self.nn(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: Tensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class HRLEncoder(nn.Module):
    def __init__(self, config):
        """
        Hierachical Representation Learning Encoder
        """
        super(HRLEncoder, self).__init__()
        self.num_convs = config.tree_depth
        self.pooling_type = config.tree_pooling_type
        self.num_features = config.hidden_dim  # input_dim
        self.nhid = config.hidden_dim  # hidden dim
        self.output_dim = config.structure_encoder.output_dim  # output dim
        self.dropout_ratio = config.hidden_dropout
        self.link_input = config.structure_encoder.link_input  # whether to readout leaf nodes
        self.drop_root = config.structure_encoder.drop_root  # whether to drop root representation during readout
        self.device = config.device_setting.device
        self.convs = self.get_convs()  # get MLP kernels
        self.pool_types = {
            'sum': global_add_pool,
            'avg': global_mean_pool,
            'max': global_max_pool
        }
        self.pool = self.pool_types[self.pooling_type]
        self.dim_align = self.get_linear()

    def get_linear(self):
        init_dim = self.nhid * self.num_convs
        if self.link_input:
            init_dim += self.num_features
        if self.drop_root:
            init_dim -= self.nhid
        if self.pooling_type == 'root':
            init_dim = self.nhid
        return nn.Sequential(
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(init_dim, self.output_dim),
        )

    @staticmethod
    def __process_layer_batch(data, layer=0):
        if layer == 0:
            return data.batch
        return data['treePHLayer%s_batch' % layer]

    def get_convs(self):
        convs = nn.ModuleList()
        _input_dim = self.num_features
        _output_dim = self.nhid
        for _ in range(self.num_convs - 1):
            conv = HRLConv(
                nn.Sequential(
                    nn.Linear(_input_dim, _output_dim),
                    nn.BatchNorm1d(_output_dim),
                    nn.ReLU(),
                    nn.Linear(_output_dim, _output_dim),
                    nn.BatchNorm1d(_output_dim),
                    nn.ReLU(),
                ))
            convs.append(conv)
            _input_dim = _output_dim
        if not self.drop_root:
            convs.append(
                HRLConv(
                    nn.Sequential(
                        nn.Linear(_input_dim, _output_dim),
                        nn.ReLU(),
                        nn.Linear(_output_dim, _output_dim),
                        nn.ReLU(),
            )))
        return convs

    def forward(self, data):
        x = data.x
        xs = [x] if self.link_input else []
        for i in range(self.num_convs):
            edge_index = data['treeEdgeMatLayer%s' % (i+1)].flip(dims=[0]).to(self.device)
            size = data.treeNodeSize[:, [i, i+1]].sum(dim=0).to(self.device)
            # size = data.treeNodeSize[:, [i, i+1]].sum(dim=0).flip(dims=[0])
            x: PairTensor = (x, torch.zeros((size[-1], x.size(-1))).to(self.device))
            x = self.convs[i](x, edge_index, size=size)
            xs.append(x)

        if self.pooling_type == 'root':
            x = self.dim_align(xs[-1])
            return x

        if self.drop_root:
            xs = xs[:-1]
        pooled_xs = []
        for i, x in enumerate(xs):
            batch = self.__process_layer_batch(data, i if self.link_input else i+1).to(self.device)
            pooled_x = self.pool(x, batch)
            # pooled_x = self.pool(x, torch.zeros(x.size(0), dtype=int).to(self.device))
            pooled_xs.append(pooled_x)

        x = torch.cat(pooled_xs, dim=1)
        x = self.dim_align(x)
        return x

