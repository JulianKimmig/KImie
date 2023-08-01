import numpy as np
import torch
from torch_geometric.data import Data as GeometricData


def graph_input_from_edgelist(edgelist, node_features, y=None, graph_features=None):
    # assert both connection directions
    edgelist = np.concatenate((edgelist, edgelist[[1, 0], :]), axis=1)
    edgelist = np.unique(edgelist, axis=1)
    if y is not None:
        if y.ndim == 1:  # should one apply if passing graph features
            y = y[None, :]

    if graph_features is not None:
        if graph_features.ndim == 1:
            graph_features = graph_features[None, :]

        assert graph_features.shape[0] == 1, "graph_features should be a single row"

    return GeometricData(
        edge_index=torch.from_numpy(edgelist).long(),
        y=None if y is None else torch.from_numpy(y.astype(float)).float(),
        x=torch.from_numpy(node_features.astype(float)).float(),
        num_nodes=len(node_features),
        # edge_index=torch.from_numpy(edge_index).long(),
        graph_features=None
        if graph_features is None
        else torch.from_numpy(graph_features.astype(float)).float(),
    )
