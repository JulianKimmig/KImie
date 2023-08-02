from tests._kimie_test_base import KImieTest
from KImie.predictor.benchmark import benchmark_molprops


class BenchmarkMolPropTest(KImieTest):
    def setUp(self) -> None:
        super().setUp()

    def test_basic_call(self):
        res = benchmark_molprops()
