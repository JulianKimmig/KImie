import numpy as np
from KImie.dataloader.molecular.ESOL import ESOL
from KImie.dataloader.molecular.dataloader import moldataloader_from_df
from KImie.dataloader.molecular.prepmol import PreparedMolDataLoader
from KImie.featurizer._molecule_featurizer import check_mol_is_prepared

from tests._kimie_test_base import KImieTest


class MolDataloaderTest(KImieTest):
    def test_prepmol(self):
        laoder = PreparedMolDataLoader(ESOL())
        for mol in laoder:
            assert check_mol_is_prepared(mol)

    def test_moldl_from_df(self):
        df = ESOL().to_df()
        dl = moldataloader_from_df(df, "test")
        for mols in zip(dl(), ESOL()):
            self.assertDictEqual(mols[0].GetPropsAsDict(), mols[1].GetPropsAsDict())

    def test_in_memory_lighning(self):
        from KImie.dataloader.lighning import InMemoryLoader
        from KImie.featurizer.prefeaturizer import Prefeaturizer
        from KImie.featurizer.molecule_featurizer import (
            molecule_Descriptors_MolWt_featurizer,
            molecule_AllChem_USR_featurizer,
        )

        featurizer = molecule_Descriptors_MolWt_featurizer
        dataset = ESOL()
        mol_loader = PreparedMolDataLoader(dataset)
        prefeaturizer = Prefeaturizer(mol_loader, featurizer=featurizer)
        loader = InMemoryLoader(prefeaturizer, batch_size=1, split=[0.85, 0.1, 0.05])
        loader.setup()
        exp_train = int(len(dataset) * 0.85)
        exp_val = int(len(dataset) * 0.1 + 1)
        exp_test = int(len(dataset) * 0.05)

        assert (
            len(list(loader.train_dataloader())) == exp_train
        ), f"{len(list(loader.train_dataloader()))} entries in trainloader, expected {exp_train}"
        assert (
            len(list(loader.val_dataloader())) == exp_val
        ), f"{len(list(loader.val_dataloader()))} entries in valloader, expected {exp_val}"
        assert (
            len(list(loader.test_dataloader())) == exp_test
        ), f"{len(list(loader.test_dataloader()))} entries in testloader, expected {exp_test}"

        for i in loader.train_dataloader():
            assert i.ndim == 2, f"{i.ndim} dimensions, expected 2"
            assert i.shape[0] == 1, f"{i.shape[0]} entries, expected 1"
            assert i.shape[1] == 1, f"{i.shape[1]} features, expected 1"
            break

        loader = InMemoryLoader(prefeaturizer, batch_size=32)
        loader.setup()

        for i in loader.train_dataloader():
            assert i.ndim == 2, f"{i.ndim} dimensions, expected 2"
            assert i.shape[0] == 32, f"{i.shape[0]} entries, expected 32"
            assert i.shape[1] == 1, f"{i.shape[1]} features, expected 1"
            break

        featurizer = molecule_AllChem_USR_featurizer
        dataset = ESOL()
        mol_loader = PreparedMolDataLoader(dataset)
        prefeaturizer = Prefeaturizer(mol_loader, featurizer=featurizer)
        loader = InMemoryLoader(prefeaturizer, batch_size=64)
        loader.setup()

        for i in loader.train_dataloader():
            assert i.ndim == 2, f"{i.ndim} dimensions, expected 2"
            assert i.shape[0] == 64, f"{i.shape[0]} entries, expected 64"
            assert i.shape[1] == 12, f"{i.shape[1]} features, expected 12"
            break

        # test seed
        raise NotImplementedError()

    def test_random_split(self):
        dl = ESOL()
        split = [0.8, 0.1, 0.1]
        sdl = dl.split(split=split, method="random", seed=42)
        self.assertEqual(len(sdl), len(split))
        for i in range(len(split)):
            self.assertEqual(len(sdl[i]), int(len(dl) * split[i]))

        dldf = dl.to_df()
        smiles = dldf["smiles"].tolist()
        smiles1 = sdl[0].to_df()["smiles"].tolist()

        # check if smiles0 is randomly distributed in smiles
        for s in smiles1:
            self.assertIn(s, smiles)

        from KImie.utils.tests.dist import wald_wolfowitz_runs_test

        test_result = wald_wolfowitz_runs_test(smiles, smiles1)

        self.assertGreaterEqual(
            test_result[1],
            0.05,
            f"Not randomly distributed,  Test statistic: {test_result[0]}, p-value: {test_result[1]}",
        )

    def test_sorted_split(self):
        dl = ESOL()
        split = [0.8, 0.1, 0.1]
        sdl = dl.split(split=split, method="sorted", seed=42)
        self.assertEqual(len(sdl), len(split))
        for i in range(len(split)):
            self.assertEqual(len(sdl[i]), int(len(dl) * split[i]))

        dldf = dl.to_df()
        smiles = dldf["smiles"].tolist()
        smiles1 = sdl[0].to_df()["smiles"].tolist()

        # check if smiles0 is randomly distributed in smiles
        for s in smiles1:
            self.assertIn(s, smiles)

        from KImie.utils.tests.dist import wald_wolfowitz_runs_test

        test_result = wald_wolfowitz_runs_test(smiles, smiles1)

        self.assertLessEqual(
            test_result[1],
            0.05,
            f"Randomly distributed,  Test statistic: {test_result[0]}, p-value: {test_result[1]}",
        )

        self.assertEqual(
            smiles1,
            smiles[: len(smiles1)],
        )
