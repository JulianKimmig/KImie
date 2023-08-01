from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List
import os
import numpy as np

from KImie.dataloader.dataloader import DataLoader
from KImie.dataloader.molecular.dataloader import MolDataLoader


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
                    f"dl must be a subclass of {self.DATALOADER_BASE_CLASS.__name__}"
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

    def prepare_train(self):
        self.prepare_model()

    def prepare_predict(self):
        self.prepare_model()

    @abstractmethod
    def run_train(self, dl: DataLoader):
        pass

    def train(self, dl: DataLoader, *args, **kwargs):
        self.check_dl(dl)
        self.prepare_train()
        self.run_train(dl, *args, **kwargs)

    @abstractmethod
    def run_predict(self, data: DataLoader) -> np.ndarray:
        pass

    def predict(self, data: DataLoader | List[Any], *args, **kwargs) -> np.ndarray:
        if isinstance(data, DataLoader):
            self.check_dl(data)
        self.prepare_predict()
        res = self.run_predict(data, *args, **kwargs)
        return res


class MolPropertyPredictor(Predictor):
    DATALOADER_BASE_CLASS = MolDataLoader

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
