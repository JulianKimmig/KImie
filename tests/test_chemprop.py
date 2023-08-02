import os
import KImie
from KImie.predictor import ToManyMolProperties, UnknownMolProperties
from KImie.predictor.chemprop_wrapper import ChempropPredictor
from tests._kimie_test_base import KImieTest
import numpy as np
from KImie.dataloader.dataloader import DataLoader
from KImie.dataloader.molecular.dataloader import moldataloader_from_df
from KImie.dataloader.molecular.ESOL import ESOL


class Dummydl(DataLoader):
    data_streamer_generator = lambda self: iter([None] * 100)


class ChempropPredictorTest(KImieTest):
    def setUp(self) -> None:
        super().setUp()
        self.predictor = ChempropPredictor(
            model_path=os.path.join(KImie.get_user_folder(), "models", "chemproptest")
        )
        esoldf = ESOL().to_df()
        self.esoltrue = esoldf["measured_log_solubility"].values
        self.smiles = esoldf["smiles"].values
        self.dl = moldataloader_from_df(esoldf, "chemproptest")()

    def test_train(self):
        with self.assertRaises(TypeError):
            self.predictor.train(dl=Dummydl())
        # train initial model
        train_res = self.predictor.train(
            dl=self.dl, mol_property="measured_log_solubility", epochs=0
        )
        from KImie.utils import jsonfy

        self.assertGreater(train_res.train_loss, 0)
        self.assertGreater(train_res.val_loss, 0)
        self.assertGreater(train_res.test_loss, 0)

        testloss1 = train_res.test_loss
        self.assertGreater(testloss1, 1)
        train_res = self.predictor.train(
            dl=self.dl, mol_property="measured_log_solubility", epochs=5
        )
        testloss2 = train_res.test_loss
        self.assertGreater(
            testloss1,
            testloss2,
        )

        print(jsonfy.dumps(train_res, indent=2))

    def test_predict(self):
        res = self.predictor.predict(self.smiles[:10])
        self.assertEqual(res.shape, (10,))

    def test_predict_dl(self):
        dl = ESOL()
        res = self.predictor.predict(dl)
        self.assertEqual(res.shape, (dl.expected_mol_count,))

    def test_multiple_props(self):
        with self.assertRaises(ToManyMolProperties):
            self.predictor.train(dl=self.dl)

        with self.assertRaises(UnknownMolProperties):
            self.predictor.train(dl=self.dl, mol_property="picklerick")
