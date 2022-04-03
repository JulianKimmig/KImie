import os
import unittest
from typing import Type

import numpy as np
from tqdm import tqdm

import KImie
from KImie import KIMIE_LOGGER
from KImie.dataloader.molecular.ESOL import ESOL
from KImie.dataloader.molecular.Tox21 import Tox21Train
from KImie.dataloader.molecular.ChEMBLdb import ChemBLdb29

from KImie.dataloader.molecular.dataloader import MolDataLoader
from KImie.dataloader.molecular.prepmol import (
    PreparedMolDataLoader,
    PreparedMolAdjacencyListDataLoader,
    PreparedMolPropertiesDataLoader,
)
from KImie.utils.sys import get_temp_dir

KIMIE_LOGGER.setLevel("DEBUG")

KImie.utils.sys.enter_test_mode()


class BaseTestClass:
    class DataSetTest(unittest.TestCase):
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
            for m in self.loader:
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
            for m in tqdm(loader, total=loader.expected_mol_count):
                if m is not None:
                    count += 1
            self.assertEqual(
                count, loader.expected_mol_count
            ), "Expected mol count does not match"

        def test_propmolproperties(self):
            if not self.TEST_PREPMOLPROPS:
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
            for m in tqdm(loader, total=loader.expected_mol_count):
                if m is not None:
                    count += 1
                    if count == 1 and self.ADJ_TEST_FIRST_SAMPLE is not None:
                        np.testing.assert_array_equal(
                            m, self.ADJ_TEST_FIRST_SAMPLE
                        ), "Expected adjacency list does not match"
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

    ADJ_TEST_FIRST_SAMPLE = np.array([[0, 1], [1, 2], [2, 3], [2, 4], [2, 5]])


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
    TEST_PREPMOLPROPS = True
    TEST_PREPMOLADJ = False
