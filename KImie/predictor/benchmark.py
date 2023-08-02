from __future__ import annotations
from typing import Literal, Tuple, List, Type, Any, Dict
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import hashlib
import json
from KImie.dataloader.dataloader import DataLoader
from KImie.dataloader.molecular.dataloader import MolDataLoader
from KImie.predictor.predictor import MolPropertyPredictor
from KImie import get_user_folder


def get_default_datasets():
    datasets = []

    from KImie.dataloader.molecular.ESOL import ESOL

    ds = ESOL()
    ds.mol_properties = ["measured_log_solubility"]
    datasets.append(ds)

    from KImie.dataloader.molecular.FreeSolv import FreeSolv

    ds = FreeSolv()
    ds.mol_properties = ["expt"]
    datasets.append(ds)

    from KImie.dataloader.molecular.Lipophilicity import Lipo1

    ds = Lipo1()
    ds.mol_properties = ["exp"]
    datasets.append(ds)

    from KImie.dataloader.molecular.logp import LogPNadinUlrich

    ds = LogPNadinUlrich()
    ds.mol_properties = ["logP_exp"]
    datasets.append(ds)

    from KImie.dataloader.molecular.meltingpoint import BradleyDoublePlusGoodMP

    ds = BradleyDoublePlusGoodMP()
    ds.mol_properties = ["mpC"]
    datasets.append(ds)

    from KImie.dataloader.molecular.pka import IUPAC_DissociationConstantsV1_0T25_5

    ds = IUPAC_DissociationConstantsV1_0T25_5()
    ds.mol_properties = ["pka_value"]
    datasets.append(ds)

    from KImie.dataloader.molecular.flashpoint import MorganFlashpoint

    ds = MorganFlashpoint()
    ds.mol_properties = ["flashpoint"]
    datasets.append(ds)

    return datasets


def get_default_models() -> List[Type[MolPropertyPredictor]]:
    from KImie.predictor.chemprop_wrapper import ChempropPredictor

    return [ChempropPredictor]


def benchmark_molprops(
    dl: MolDataLoader = None,
    output_path: str = None,
    split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    split_method: Literal["random"] = "random",  #
    models: List[Type[MolPropertyPredictor]] = None,
    epochs: int = -1,
    worker=0,
    gpu=0,
    resume=True,
    seed=42,
    model_training_kwargs: List[Dict[str, Any]] = None,
    skip_existing=True,
):
    resultdataframe = pd.DataFrame(
        columns=["model", "dataset", "split", "epochs", "worker", "mae", "rmse", "r2"]
    )

    epochs = int(epochs)

    # default output path is user_folder
    if output_path is None:
        output_path = os.path.join(get_user_folder(), "benchmark")

    # create output path if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    benchmark_csv = os.path.join(output_path, "benchmark.csv")
    if resume and output_path is not None and os.path.exists(benchmark_csv):
        resultdataframe = pd.read_csv(benchmark_csv)

    if dl is None:
        dl = get_default_datasets()

    if models is None:
        models = get_default_models()

    if model_training_kwargs is None:
        model_training_kwargs = [{}] * len(models)

    if len(model_training_kwargs) != len(models):
        raise ValueError("model_training_kwargs must have the same length as models")

    # use multiline tqdm
    import time

    resultdata = []
    for r, d in resultdataframe.iterrows():
        resultdata.append(d.to_dict())

    modelsdir = os.path.join(output_path, "models")
    with tqdm(total=len(models)) as modelbar:
        for mi, modelclass in enumerate(models):
            modelbar.set_description(f"model: {modelclass.__name__}")
            modelbar.update()
            _modeltrainkwargs = model_training_kwargs[mi]

            # create model output path

            with tqdm(total=len(dl)) as dlbar:
                for dataset in dl:
                    dlbar.set_description(f"dataset: {dataset}")
                    dlbar.update()

                    modelstring = modelclass.__name__

                    if (epochs is None) or (epochs < 1):
                        _epochs = max(1, min(100, int(100000 / len(dataset))))
                    else:
                        _epochs = epochs
                    modelhashdata = {
                        "modeltrainkwargs": json.dumps(
                            _modeltrainkwargs, sort_keys=True
                        ),
                        "epochs": _epochs,
                        "split_method": split_method,
                        "split": split,
                        "seed": seed,
                    }

                    # hash modelkwargs using md5
                    modelkwwargshash = hashlib.md5(
                        json.dumps(modelhashdata, sort_keys=True).encode("utf-8")
                    ).hexdigest()
                    modelstring += f"_{modelkwwargshash}"

                    # check if an entry with model=modelstring and dataset=str(dataset) already exists
                    mask = (resultdataframe["model"] == modelstring) & (
                        resultdataframe["dataset"] == str(dataset)
                    )

                    if not resultdataframe.loc[mask].empty and skip_existing:
                        continue
                    basemodeloutputpath = os.path.join(modelsdir, modelstring)

                    modeloutputpath = os.path.join(basemodeloutputpath, str(dataset))

                    model = modelclass(model_path=modeloutputpath)
                    res = model.train(
                        dataset,
                        epochs=_epochs,
                        worker=worker,
                        gpu=gpu,
                        split_method=split_method,
                        split=split,
                        seed=seed,
                        **_modeltrainkwargs,
                    )
                    resultdata.append(
                        {
                            "model": modelstring,
                            "dataset": str(dataset),
                            "split": json.dumps(split),
                            "epochs": _epochs,
                            "metric": res.metric,
                            "train_loss": res.train_loss,
                            "val_loss": res.val_loss,
                            "test_loss": res.test_loss,
                        }
                    )

                    resultdataframe = pd.DataFrame(resultdata)
                    if output_path is not None:
                        resultdataframe.to_csv(benchmark_csv, index=False)
    return resultdataframe
