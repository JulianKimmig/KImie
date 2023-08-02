from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List
import os
import numpy as np
from KImie import KIMIE_LOGGER
from KImie.dataloader.dataloader import DataLoader
from KImie.dataloader.molecular.dataloader import MolDataLoader
from KImie.utils.data import normalize_split
import torchmetrics
import torch
from KImie.utils.jsonfy import dump


def metric_to_loss(metric: str, y: np.ndarray, yhat: np.ndarray) -> str:
    if metric == "mse":
        return torchmetrics.functional.mean_squared_error(
            torch.from_numpy(yhat), torch.from_numpy(y), squared=True
        ).numpy()
    elif metric == "rmse":
        return torchmetrics.functional.mean_squared_error(
            torch.from_numpy(yhat), torch.from_numpy(y), squared=False
        ).numpy()
    elif metric == "mae":
        return torchmetrics.functional.mean_absolute_error(
            torch.from_numpy(yhat), torch.from_numpy(y)
        ).numpy()
    elif metric == "r2":
        return torchmetrics.functional.r2_score(
            torch.from_numpy(yhat), torch.from_numpy(y)
        ).numpy()

    raise ValueError(f"metric {metric} not supported")


class Predictor(ABC):
    MODEL_PATH_REQUIRED = False
    DATALOADER_BASE_CLASS = DataLoader

    def __init__(self, model_path=None):
        if self.MODEL_PATH_REQUIRED and model_path is None:
            raise ValueError("model_path is required")
        self._model_path = model_path

    def check_dl(self, dl: DataLoader, raise_error=True):
        """Check if dl is a subclass of DATALOADER_BASE_CLASS
        Args:
            dl (DataLoader): dataloader to check
            raise_error (bool, optional): raise error if dl is not a subclass of DATALOADER_BASE_CLASS. Defaults to True.
        Returns:
            bool: True if dl is a subclass of DATALOADER_BASE_CLASS, False otherwise

        Raises:
            TypeError: if dl is not a subclass of DATALOADER_BASE_CLASS and raise_error is True
        """
        if not isinstance(dl, self.DATALOADER_BASE_CLASS):
            if raise_error:
                raise TypeError(
                    f"dl must be a subclass of {self.DATALOADER_BASE_CLASS.__name__} but is {type(dl).__name__}"
                )
            else:
                return False
        return True

    @property
    def model_path(self):
        return self._model_path

    def prepare_model(self):
        if self.model_path:
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)

    def prepare_train(self, dl: DataLoader, kwargs):
        if "metric" not in kwargs:
            kwargs["metric"] = dl.default_metric

        props = {}
        # check split and normalize to 1
        if "split" not in kwargs:
            split = (0.8, 0.1, 0.1)
        else:
            split = kwargs.pop("split")

        split = normalize_split(split)

        props["split"] = list(split)

        props["split_method"] = kwargs.pop("split_method", "random")
        props["seed"] = kwargs.pop("seed", None)
        dls = dl.split(split, method=props["split_method"], seed=props["seed"])
        kwargs["train_dl"] = dls[0]

        if len(split) > 1 and split[1] > 0:
            kwargs["val_dl"] = dls[1]
        else:
            kwargs["val_dl"] = None

        if len(split) > 2 and split[2] > 0:
            kwargs["test_dl"] = dls[2]
        else:
            kwargs["test_dl"] = None
        self.prepare_model()
        return kwargs, props

    def prepare_predict(self):
        self.prepare_model()

    @abstractmethod
    def run_train(
        self,
        dl: DataLoader,
        metric: str,
    ) -> TrainingResult:
        pass

    def train(self, dl: DataLoader, **kwargs) -> TrainingResult:
        KIMIE_LOGGER.info(f"Starting training of {self.__class__.__name__}")
        self.check_dl(dl)
        kwargs, props = self.prepare_train(dl, kwargs)
        res = self.run_train(**kwargs)
        res.update(props)
        res.train_dl = kwargs.pop("train_dl", None)
        res.val_dl = kwargs.pop("val_dl", None)
        res.test_dl = kwargs.pop("test_dl", None)
        res.update(kwargs)

        res.eval()

        with open(os.path.join(self.model_path, "training.json"), "w") as f:
            dump(res, f, indent=2)

        return res

    @abstractmethod
    def run_predict(self, data: DataLoader) -> np.ndarray:
        pass

    def predict(self, data: DataLoader | List[Any], *args, **kwargs) -> np.ndarray:
        if isinstance(data, DataLoader):
            self.check_dl(data)
        self.prepare_predict()
        res = self.run_predict(data, *args, **kwargs)
        return res

    @abstractmethod
    def dl_to_xy(self, dl: DataLoader, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        pass

    def calc_loss(self, dl: DataLoader, metric: str, **kwargs) -> float:
        x, y = self.dl_to_xy(dl, **kwargs)
        y_pred = self.predict(x)
        return metric_to_loss(metric, y, y_pred)


class TrainingResult(dict):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.update(kwargs)

    def __getattribute__(self, __name: str) -> Any:
        try:
            return super().__getattribute__(__name)
        except AttributeError as exc:
            return self[__name]

    @property
    def train_loss(self):
        try:
            return self["train_loss"]
        except KeyError as exc:
            pass

        KIMIE_LOGGER.info(
            "Calculating train loss since it is not in the training result"
        )
        self["train_loss"] = self.model.calc_loss(self.train_dl, **self)
        return self["train_loss"]

    @property
    def val_loss(self):
        try:
            return self["val_loss"]
        except KeyError as exc:
            pass

        KIMIE_LOGGER.info("Calculating val loss since it is not in the training result")
        self["val_loss"] = self.model.calc_loss(self.val_dl, **self)
        return self["val_loss"]

    @property
    def test_loss(self):
        try:
            return self["test_loss"]
        except KeyError as exc:
            pass

        KIMIE_LOGGER.info(
            "Calculating test loss since it is not in the training result"
        )
        self["test_loss"] = self.model.calc_loss(self.test_dl, **self)
        return self["test_loss"]

    def eval(self):
        tr, vl, ts = self.train_loss, self.val_loss, self.test_loss


class MolPropertyPredictor(Predictor):
    DATALOADER_BASE_CLASS = MolDataLoader

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def dl_to_xy(
        self, dl: MolDataLoader, mol_property: str, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        x = []
        y = []
        for mol in dl:
            x.append(mol)
            y.append(mol.GetProp(mol_property))

        x = np.array(x)
        y = np.array(y)
        return x, y
