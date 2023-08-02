from __future__ import annotations
import os
from typing import List
import sklearn
from KImie.dataloader.molecular.dataloader import MolDataLoader
from KImie.predictor import ToManyMolProperties, UnknownMolProperties
from KImie.predictor.predictor import MolPropertyPredictor, TrainingResult
from KImie.dataloader.dataloader import DataLoader
import KImie
from KImie import KIMIE_LOGGER
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd


class ChempropPredictor(MolPropertyPredictor):
    MODEL_PATH_REQUIRED = True
    split_method_map = {"random": "random"}
    metric_map = {
        "rmse": "rmse",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def dl_to_xy(
        self, dl: MolDataLoader, mol_property: str, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        df = dl.to_df()
        x = df["smiles"].values
        y = df[mol_property].values
        return x, y

    def run_train(
        self,
        train_dl: MolDataLoader = None,
        val_dl: MolDataLoader = None,
        test_dl: MolDataLoader = None,
        metric: str = None,
        mol_property: str = None,
        epochs: int = 30,
        worker: int = 0,
        gpu: int = 0,
        model_training_kwargs: dict = None,
    ):
        from chemprop.train import run_training, cross_validate
        from chemprop.args import TrainArgs

        if mol_property is None:
            if len(train_dl.mol_properties) > 1:
                raise ToManyMolProperties(
                    f"mol_property must be specified when there are multiple for {train_dl} with {train_dl.mol_properties}"
                )
            mol_property = train_dl.mol_properties[0]

        if mol_property not in train_dl.mol_properties:
            raise UnknownMolProperties(
                f"{mol_property} is not a mol_property of {train_dl} with {train_dl.mol_properties}"
            )

        traindata = os.path.join(self.model_path, "train.csv")
        train_dl.to_csv(traindata, smiles="smiles")
        testdata = os.path.join(self.model_path, "test.csv")
        test_dl.to_csv(testdata, smiles="smiles")
        valdata = os.path.join(self.model_path, "val.csv")
        val_dl.to_csv(valdata, smiles="smiles")

        args = TrainArgs()
        args.target_columns = [mol_property]
        args.smiles_columns = ["smiles"]
        args.data_path = traindata
        args.separate_val_path = valdata
        args.separate_test_path = testdata
        args.dataset_type = "regression"
        args.save_dir = os.path.join(self.model_path, "checkpoints")
        args.process_args()
        args._parsed = True
        args.epochs = epochs
        args.metric = self.metric_map[metric]
        args.num_workers = worker
        args.gpu = gpu
        args.log_frequency = 1

        if model_training_kwargs is not None:
            for k, v in model_training_kwargs.items():
                setattr(args, k, v)

        mean_score, std_score = cross_validate(args=args, train_func=run_training)
        res = TrainingResult(
            self, chemprop_mean_score=mean_score, chemprop_std_score=std_score
        )

        eventfilesdir = os.path.join(args.save_dir, "model_0")
        KIMIE_LOGGER.debug(f"eventfilesdir: {eventfilesdir}")
        eventfiles = [
            os.path.join(eventfilesdir, f)
            for f in os.listdir(eventfilesdir)
            if f.startswith("events.out.tfevents")
        ]
        if len(eventfiles) == 0:
            return res

        # get the last eventfile by sorting by the timestamp. namestruc= events.out.tfevents.<timestamp>.<machine>, machine can contain "."
        eventfiles.sort(
            key=lambda x: int(os.path.basename(x).split(".")[3]), reverse=True
        )
        eventfile = eventfiles[0]
        KIMIE_LOGGER.debug(f"eventfile: {eventfile}")

        eventacc = EventAccumulator(eventfile).Reload()
        tags = eventacc.Tags()["scalars"]
        data = {}
        for tag in tags:
            events = eventacc.Scalars(tag)
            if tag == "test_" + args.metric or tag == "test_loss":
                tag = "test_loss_data"
            if tag == "validation_" + args.metric or tag == "validation_loss":
                tag = "val_loss_data"
            if tag == "train_" + args.metric or tag == "train_loss":
                tag = "train_loss_data"
            if tag not in data:
                data[tag] = []
            for event in events:
                data[tag].append((event.step, event.value))

        # make np array from data and sort by first column
        for k in data:
            data[k] = np.array(data[k])
            data[k] = data[k][data[k][:, 0].argsort()]
        res.update(data)

        if "test_loss_data" in data:
            res["test_loss"] = data["test_loss_data"][-1][1]
        if "val_loss_data" in data:
            res["val_loss"] = data["val_loss_data"][-1][1]
        if "train_loss_data" in data:
            res["train_loss"] = data["train_loss_data"][-1][1]

        return res

    def run_predict(self, data: DataLoader | List[str]) -> np.ndarray:
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
