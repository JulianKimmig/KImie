from __future__ import annotations
import os
from typing import List
import sklearn
from KImie.dataloader.molecular.dataloader import MolDataLoader
from KImie.predictor.predictor import MolPropertyPredictor
from KImie.dataloader.dataloader import DataLoader
import KImie
import numpy as np


class ChempropPredictor(MolPropertyPredictor):
    MODEL_PATH_REQUIRED = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_train(
        self,
        dl: MolDataLoader,
        mol_property: str = None,
        epochs: int = 30,
        worker: int = 0,
        gpu: int = 0,
        **kwargs,
    ):
        from chemprop.train import run_training, cross_validate
        from chemprop.args import TrainArgs

        if mol_property is None:
            if len(dl.mol_properties) > 1:
                raise ValueError(
                    f"mol_property must be specified when there are multiple for {dl} with {dl.mol_properties}"
                )
            mol_property = dl.mol_properties[0]

        if mol_property not in dl.mol_properties:
            raise ValueError(
                f"{mol_property} is not a mol_property of {dl} with {dl.mol_properties}"
            )

        traindata = os.path.join(self.model_path, "train.csv")
        dl.to_csv(traindata, smiles="smiles")

        args = TrainArgs()
        args.target_columns = [mol_property]
        args.smiles_columns = ["smiles"]
        args.data_path = traindata
        args.dataset_type = "regression"
        args.save_dir = os.path.join(self.model_path, "checkpoints")
        args.process_args()
        args._parsed = True
        args.epochs = epochs
        args.num_workers = worker
        args.gpu = gpu
        cross_validate(args=args, train_func=run_training)

    def run_predict(self, data: DataLoader | List[str]):
        from chemprop.train import make_predictions
        from chemprop.args import PredictArgs

        args = PredictArgs()

        datapath = os.path.join(self.model_path, "pred.csv")
        if isinstance(data, DataLoader):
            data.to_csv(datapath, smiles="smiles")
        else:
            with open(datapath, "w") as f:
                f.write("smiles\n")
                f.write("\n".join(data))
        args.smiles_columns = ["smiles"]

        args.checkpoint_dir = os.path.join(self.model_path, "checkpoints")
        args.test_path = datapath

        args.preds_path = os.path.join(self.model_path, "pred_out.csv")
        args.process_args()
        preds = make_predictions(args, return_uncertainty=False)
        preds = np.array(preds).flatten()
        return preds
