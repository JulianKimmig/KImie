from tests._kimie_test_base import KImieTest


class TorchGeomTest(KImieTest):
    def test_graph_input_from_edgelist(self):
        from KImie.nn.graph.torch_geometric import graph_input_from_edgelist

        raise NotImplementedError()

    def test_pooling(self):
        from KImie.nn.graph.pooling import (
            PoolingBase,
            PoolWeightedSum,
            PoolMax,
            PoolMin,
            PoolMean,
        )

        raise NotImplementedError()
