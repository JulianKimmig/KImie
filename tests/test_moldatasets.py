from typing import Type

import numpy as np
from tqdm import tqdm

import KImie
from KImie import KIMIE_LOGGER
from KImie.dataloader.molecular.ESOL import ESOL
from KImie.dataloader.molecular.Tox21 import Tox21Train
from KImie.dataloader.molecular.ChEMBLdb import ChemBLdb29
from KImie.dataloader.molecular.FreeSolv import FreeSolv_0_51

from KImie.dataloader.molecular.dataloader import MolDataLoader
from KImie.dataloader.molecular.prepmol import (
    PreparedMolDataLoader,
    PreparedMolAdjacencyListDataLoader,
    PreparedMolPropertiesDataLoader,
)
from KImie.utils.sys import get_temp_dir
from tests._kimie_test_base import KImieTest


class BaseTestClass:
    class DataSetTest(KImieTest):
        DS_NAME: str = None
        DS_CLASS: Type[MolDataLoader] = None
        DS_KWARGS: dict = dict()

        TEST_DL = True
        TEST_PREPMOL = True
        TEST_PREPMOLPROPS = True
        TEST_PREPMOLADJ = True
        TEST_EXPECTED_SIZE = True

        ADJ_TEST_FIRST_SAMPLE = None

        def setUp(self) -> None:
            super().setUp()
            assert self.DS_NAME is not None, "DS_NAME is not set"
            assert self.DS_CLASS is not None, "DS_CLASS is not set"
            self.loader = self.DS_CLASS(data_streamer_kwargs=self.DS_KWARGS)

        def test_dl(self):
            if not self.TEST_DL:
                return
            self.loader.download()

        def test_expected_size(self):
            if not self.TEST_EXPECTED_SIZE:
                return
            self.loader.close()
            mol_count = 0
            iter_count = 0
            expdc = self.loader.expected_data_size
            expmc = self.loader.expected_mol_count
            for m in tqdm(self.loader, total=expdc, mininterval=1):
                if iter_count % 10_000 == 0:
                    print(iter_count, end=" ")
                iter_count += 1
                if m is not None:
                    mol_count += 1
            self.assertEqual(iter_count, expdc), "Expected data size does not match"
            self.assertEqual(mol_count, expmc), "Expected mol count does not match"

        def test_prepmol(self):
            if not self.TEST_PREPMOL:
                return
            self.loader.close()
            loader = PreparedMolDataLoader(self.loader)
            count = 0
            for m in tqdm(loader, total=loader.expected_mol_count, mininterval=1):
                if m is not None:
                    count += 1
            self.assertEqual(
                count, loader.expected_mol_count
            ), "Expected mol count does not match"

        def test_propmolproperties(self):
            if not self.TEST_PREPMOLPROPS:
                return
            if hasattr(self.loader, "mol_properties"):
                if self.loader.mol_properties is not None:
                    if len(self.loader.mol_properties) == 0:
                        return
            self.loader.close()
            loader = PreparedMolPropertiesDataLoader(self.loader)
            count = 0
            i = None
            for i in loader:
                count += len(i)
            self.assertEqual(
                count, loader.expected_mol_count
            ), "Expected mol count does not match"
            assert i is not None, "No data returned"
            self.assertEqual(
                i.index[-1], self.loader.expected_data_size - 1
            ), "Expected data size does not match"

        def test_prepmoladj(self):
            if not self.TEST_PREPMOLADJ:
                return
            self.loader.close()
            loader = PreparedMolAdjacencyListDataLoader(self.loader)
            count = 0

            for m in tqdm(loader, total=loader.expected_mol_count, mininterval=1):
                if m is not None:
                    count += 1
                    if count == 1 and self.ADJ_TEST_FIRST_SAMPLE is not None:
                        np.testing.assert_array_equal(
                            m,
                            self.ADJ_TEST_FIRST_SAMPLE,
                            err_msg=f"Expected adjacency list does not match (is {m.tolist()})",
                        )
            self.assertEqual(
                count, loader.expected_mol_count
            ), "Expected mol count does not match"


class ESOLTest(BaseTestClass.DataSetTest):
    DS_NAME = "ESOL"
    DS_CLASS = ESOL

    TEST_DL = False
    TEST_PREPMOL = True
    TEST_PREPMOLPROPS = True
    TEST_PREPMOLADJ = True

    ADJ_TEST_FIRST_SAMPLE = np.array(
        [[0, 6], [1, 6], [2, 6], [3, 7], [4, 7], [5, 7], [6, 7]]
    )


class FreeSolv_0_51(BaseTestClass.DataSetTest):
    DS_NAME = "FreeSolv_0_51"
    DS_CLASS = FreeSolv_0_51

    TEST_DL = True
    TEST_PREPMOL = True
    TEST_PREPMOLPROPS = True
    TEST_PREPMOLADJ = True

    ADJ_TEST_FIRST_SAMPLE = np.array([[0, 4], [1, 5], [2, 5], [3, 5], [4, 5]])


class Tox21TrainTest(BaseTestClass.DataSetTest):
    DS_NAME = "Tox21Train"
    DS_CLASS = Tox21Train
    DS_KWARGS = dict(iter_None=True)

    TEST_DL = False
    TEST_PREPMOL = True
    TEST_PREPMOLPROPS = True
    TEST_PREPMOLADJ = False

    ADJ_TEST_FIRST_SAMPLE = np.array(
        [
            [1, 2],
            [2, 3],
            [2, 12],
            [3, 4],
            [3, 9],
            [4, 5],
            [5, 6],
            [5, 7],
            [7, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [11, 12],
            [11, 17],
            [12, 13],
            [13, 14],
            [14, 15],
            [14, 16],
            [16, 17],
            [18, 19],
            [18, 28],
            [19, 20],
            [19, 25],
            [20, 21],
            [21, 22],
            [21, 23],
            [23, 24],
            [24, 25],
            [25, 26],
            [26, 27],
            [27, 28],
            [27, 33],
            [28, 29],
            [29, 30],
            [30, 31],
            [30, 32],
            [32, 33],
        ]
    )


class ChemBLdb29Test(BaseTestClass.DataSetTest):
    DS_NAME = "ChemBLdb29"
    DS_CLASS = ChemBLdb29
    DS_KWARGS = dict(iter_None=True)

    TEST_DL = False
    TEST_EXPECTED_SIZE = False
    TEST_PREPMOL = False
    TEST_PREPMOLPROPS = False
    TEST_PREPMOLADJ = False


from KImie.dataloader.molecular.Lipophilicity import Lipo1


class Lipo1(BaseTestClass.DataSetTest):
    DS_NAME = "Lipo1"
    DS_CLASS = Lipo1

    TEST_DL = True
    TEST_PREPMOL = True
    TEST_PREPMOLPROPS = True
    TEST_PREPMOLADJ = True

    ADJ_TEST_FIRST_SAMPLE = np.array(
        [
            [0, 23],
            [1, 24],
            [2, 25],
            [3, 26],
            [4, 27],
            [5, 28],
            [6, 31],
            [7, 32],
            [8, 39],
            [9, 39],
            [10, 40],
            [11, 40],
            [12, 40],
            [13, 41],
            [14, 41],
            [15, 42],
            [16, 42],
            [17, 43],
            [18, 43],
            [19, 44],
            [20, 44],
            [21, 29],
            [22, 33],
            [22, 34],
            [23, 25],
            [23, 29],
            [24, 26],
            [24, 29],
            [25, 30],
            [26, 30],
            [27, 28],
            [27, 31],
            [28, 32],
            [30, 36],
            [31, 34],
            [32, 35],
            [33, 38],
            [33, 39],
            [34, 35],
            [35, 38],
            [36, 41],
            [36, 42],
            [37, 39],
            [37, 43],
            [37, 44],
            [38, 40],
            [41, 43],
            [42, 44],
        ]
    )


from KImie.dataloader.molecular.meltingpoint import BradleyDoublePlusGoodMP


class BradleyDoublePlusGoodMPTest(BaseTestClass.DataSetTest):
    DS_NAME = "BradleyDoublePlusGoodMP"
    DS_CLASS = BradleyDoublePlusGoodMP
    DS_KWARGS = dict(iter_None=True)

    TEST_DL = True
    TEST_PREPMOL = True
    TEST_PREPMOLPROPS = True
    TEST_PREPMOLADJ = True

    ADJ_TEST_FIRST_SAMPLE = np.array(
        [
            [0, 10],
            [1, 10],
            [2, 10],
            [3, 11],
            [4, 11],
            [5, 12],
            [6, 12],
            [7, 13],
            [8, 13],
            [9, 14],
            [10, 14],
            [11, 12],
            [11, 13],
            [12, 14],
            [13, 14],
        ]
    )

from KImie.dataloader.molecular.nmrshiftdb2 import NMRShiftDB2_1H
class NMRShiftDB2_1HTest(BaseTestClass.DataSetTest):
    DS_NAME = "NMRShiftDB2_1H"
    DS_CLASS = NMRShiftDB2_1H

