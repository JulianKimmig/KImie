import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from KImie import KIMIE_LOGGER


class DataStreamer:
    def __init__(
        self, dataloader, cached=False, progress_bar_kwargs=None, iter_None=True
    ):
        self._iter_None = iter_None
        if progress_bar_kwargs is None:
            progress_bar_kwargs = {}
        self._progress_bar_kwargs = progress_bar_kwargs
        self.dataloader = dataloader
        self._cached = cached
        self._cache_data = []
        self._all_cached = False

        self._position = 0
        self._in_iter = False

    @classmethod
    def generator(cls, **kwargs):
        def _generator(*args, **skwargs):
            return cls(*args, **{**kwargs, **skwargs})

        return _generator

    def get_all_entries(self, *args, **kwargs):
        return self.get_n_entries(self.dataloader.expected_data_size, *args, **kwargs)

    def get_n_entries(self, n: int, progress_bar=True, desc="load entries"):
        dat = []
        if len(self._cache_data) < n and not self._all_cached:
            if progress_bar:
                g = tqdm(
                    enumerate(self), total=n, **self._progress_bar_kwargs, desc=desc
                )
            else:
                g = enumerate(self)
            if self.cached:
                for j, d in g:
                    if j >= n:
                        break
            else:
                for j, d in g:
                    dat.append(d)
                    if j >= n:
                        break
            g.close()
            self.close()

        else:
            l = len(self._cache_data)
            if l < n:
                n = l
            if progress_bar:
                return [
                    self._cache_data[i]
                    for i in tqdm(range(n), total=n, **self._progress_bar_kwargs)
                ]
        if self.cached:
            return self._cache_data[:n]
        return dat[:n]

    @property
    def cached(self):
        return self._cached

    def clear_cache(self):
        self._cache_data = []
        self._all_cached = False

    @cached.setter
    def cached(self, cached: bool):
        if cached != self._cached:
            self.clear_cache()
            self._cached = cached

    def get_iterator(self):
        raise NotImplementedError()

    # def next(self):

    def __next__(self):
        try:
            k = next(self._iter)
            if k is not None:
                k = self.update_data(k)

            if not self._iter_None:
                while k is None:
                    self.dataloader.expected_data_size -= 1
                    self._removed += 1
                    k = next(self._iter)
                    if k is not None:
                        k = self.update_data(k)

        except StopIteration:
            if self._cached:
                self._all_cached = True
            if self._position != len(self) + self._removed:
                KIMIE_LOGGER.warning(
                    f"{self.dataloader} returns a different size ({self._position}) than expected({self.dataloader.expected_data_size}), {self._removed} entries where removed"
                )
            self.close()
            raise StopIteration

        if self._cached:
            self._cache_data.append(k)
        self._position += self.position_increment(k)

        return k

    def position_increment(self, data):
        return 1

    def __len__(self):
        if self._iter_None:
            return self.dataloader.expected_data_size
        else:
            return len(self.dataloader)

    def close(self):
        if self._in_iter:
            self._iter.close()
            self._in_iter = False

    def __iter__(self):
        if not self._in_iter:
            self._position = 0
            self._removed = 0
            self._iter = self.get_iterator()
            self._in_iter = True
        return self

    def update_data(self, d):
        return d


class NumpyStreamer(DataStreamer):
    def __init__(self, dataloader, folder_getter, *args, cached=False, **kwargs):
        super(NumpyStreamer, self).__init__(
            dataloader,
            *args,
            **kwargs,
            cached=cached,
            progress_bar_kwargs=dict(unit="array", unit_scale=True),
        )

        self._folder_getter = folder_getter

    def get_iterator(self):
        path = self._folder_getter(self)

        def _it():
            for f in sorted(
                [mf for mf in os.listdir(path) if mf.endswith(".npy")],
                key=lambda s: int(s[:-4]),
            ):
                yield np.load(os.path.join(path, f))

        return _it()


class CSVStreamer(DataStreamer):
    def __init__(
        self, dataloader, file_getter, *args, cached=False, chunksize=None, **kwargs
    ):
        super(CSVStreamer, self).__init__(
            dataloader,
            *args,
            **kwargs,
            cached=cached,
            progress_bar_kwargs=dict(unit="lines", unit_scale=True),
        )

        self._file_getter = file_getter

        self._chunksize = chunksize

    def position_increment(self, data):
        return len(data.index)

    @property
    def chunksize(self):
        # if cheunksize is not defined use 10**6 divided by the header length
        if self._chunksize is None:
            with pd.read_csv(
                self._file_getter(self), chunksize=1, index_col=None
            ) as reader:
                for chunk in reader:
                    self._chunksize = max(1, int(10**6 / max(1, len(chunk.columns))))
                    break
        return self._chunksize

    def get_iterator(self):
        file = self._file_getter(self)

        def _it():
            with pd.read_csv(file, chunksize=self.chunksize, index_col=0) as reader:
                for chunk in reader:
                    yield (chunk)

        return _it()


class MemoryStreamer(DataStreamer):
    def __init__(self, dataloader, data, *args, cached=False, **kwargs):
        super(MemoryStreamer, self).__init__(
            dataloader,
            *args,
            **kwargs,
            cached=cached,
            progress_bar_kwargs=dict(unit="array", unit_scale=True),
        )
        self._data = data

    def get_iterator(self):
        def _it():
            for d in self._data:
                yield d

        return _it()


class DfStreamer(MemoryStreamer):
    def get_iterator(self):
        if not isinstance(self._data, pd.DataFrame):
            raise ValueError("data is not a pandas dataframe")

        def _it():
            if not isinstance(self._data, pd.DataFrame):
                raise ValueError("data is not a pandas dataframe")

            for i, d in self._data.iterrows():
                yield d

        return _it()
