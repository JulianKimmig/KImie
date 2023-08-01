import numpy as np

import pytorch_lightning as pl
from torch.utils.data import Subset, DataLoader
from typing import Type


class InMemoryLoader(pl.LightningDataModule):
    def __init__(
        self,
        data,
        split=[0.8, 0.1, 0.1],
        batch_size=32,
        dataloader: Type[DataLoader] = DataLoader,
        seed=None,
        shuffle=True,
        **dataloader_kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.split = np.concatenate([np.array(split).flatten(), np.zeros(3)])[:3]
        self.split = self.split / self.split.sum()
        self.dataloader_kwargs = dataloader_kwargs
        self.dataloader = dataloader
        self.data = [d for d in data]
        if not shuffle:
            seed = -1
        self._seed = seed

    def setup(self, stage=None):
        data = self.data
        l = len(data)
        split = (self.split * l).astype(int)
        while l > split.sum():
            split[((l - split.sum()) % len(split))] += 1

        indices = np.arange(sum(split))
        # randomize indices
        if self._seed is None or self._seed >= 0:
            np.random.RandomState(self._seed).shuffle(indices)

        self.train_indices, self.val_indices, self.test_indices = (
            indices[offset - length : offset]
            for offset, length in zip(np.add.accumulate(split), split)
        )
        self.train_ds, self.val_ds, self.test_ds = [
            Subset(data, ind)
            for ind in (self.train_indices, self.val_indices, self.test_indices)
        ]

    def train_dataloader(self):
        if self.train_ds is not None:
            return self.dataloader(
                self.train_ds, batch_size=self.batch_size, **self.dataloader_kwargs
            )
        return None

    def val_dataloader(self):
        if self.val_ds is not None:
            return self.dataloader(
                self.val_ds, batch_size=self.batch_size, **self.dataloader_kwargs
            )
        return None

    def test_dataloader(self):
        if self.test_ds is not None:
            return self.dataloader(
                self.test_ds, batch_size=self.batch_size, **self.dataloader_kwargs
            )
        return None
