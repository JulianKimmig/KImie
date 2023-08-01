import torch
from torch import nn
from torch_scatter import (
    scatter_add,
    scatter_max,
    scatter_min,
    scatter_mean,
    scatter_mul,
    # scatter_std,
    scatter_logsumexp,
    # scatter_log_softmax,
    # scatter_softmax,
)
from torch_geometric.nn import Sequential as GCSequential, GCNConv


class PoolingBase(nn.Module):
    @property
    def size(self):
        return 1


class PoolWeightedSum(PoolingBase):
    def __init__(self, n_in_feats=None, n_out_feats=1, normalize=True, bias=True):
        super(PoolWeightedSum, self).__init__()
        self._normalize = normalize
        self._bias = bias
        self._n_out_feats = n_out_feats
        if self._normalize:
            self.weighting_of_nodes = nn.Sequential(
                nn.Linear(n_in_feats, self._n_out_feats, bias=self._bias), nn.Sigmoid()
            )
        else:
            self.weighting_of_nodes = nn.Linear(
                n_in_feats, self._n_out_feats, bias=self._bias
            )

    @property
    def size(self):
        return self._n_out_feats

    def forward(self, feats, batch):
        # feats dims = nodes,last_gcn_feats
        weights = self.weighting_of_nodes(feats)  # dims = nodes,out_feats
        weight_feats = torch.einsum(
            "io, if -> iof", weights, feats
        )  # dims = nodes,out_feats,last_gcn_feats
        weight_feats = weight_feats.flatten(
            start_dim=1, end_dim=-1
        )  # dims = nodes,out_feats*last_gcn_feats
        summed_nodes = scatter_add(
            weight_feats, batch, dim=0
        )  # dims = out_feats*last_gcn_feats
        return summed_nodes


class PoolMax(PoolingBase):
    def forward(self, feats, batch):
        maxed_nodes, _ = scatter_max(feats, batch, dim=0)
        return maxed_nodes


class PoolMin(PoolingBase):
    def forward(self, feats, batch):
        maxed_nodes, _ = scatter_min(feats, batch, dim=0)
        return maxed_nodes


class PoolMean(PoolingBase):
    def forward(self, feats, batch):
        maxed_nodes = scatter_mean(feats, batch, dim=0)
        return maxed_nodes


class PoolProd(PoolingBase):
    def forward(self, feats, batch):
        maxed_nodes = scatter_mul(feats, batch, dim=0)
        return maxed_nodes


class PoolLogSumExp(PoolingBase):
    def forward(self, feats, batch):
        maxed_nodes = scatter_logsumexp(feats, batch, dim=0)
        return maxed_nodes


class PoolSum(PoolingBase):
    def forward(self, feats, batch):
        summed_nodes = scatter_add(
            feats, batch, dim=0
        )  # zeros.scatter_add(0, segment_ids, weight_feats) #dims = number of graphs,last_gcn_feats
        return summed_nodes


class MergedPooling(PoolingBase):
    def __init__(self, pooling_layer_dict):
        super().__init__()
        if isinstance(pooling_layer_dict, list):
            pooling_layer_dict = {str(i): pl for i, pl in enumerate(pooling_layer_dict)}

        pool_names = list(pooling_layer_dict.keys())

        assert len(pool_names) == len(pooling_layer_dict)

        self.pool_names = pool_names
        self.pooling_layer_dict = pooling_layer_dict
        self.pooling_layer = nn.ModuleDict(pooling_layer_dict)

    def forward(self, feats, batch):
        return torch.cat(
            [self.pooling_layer[pl](feats, batch) for pl in self.pool_names], dim=1
        )

    @property
    def size(self):
        return sum([pl.size for pl in self.pooling_layer.values()])

    def __len__(self):
        return len(self.pool_names)
