from __future__ import annotations
import os
import re
import shutil
from typing import List, Tuple, Dict, Any, Type, Callable
import pandas as pd
import requests
from tqdm import tqdm

from KImie.dataloader.streamer import DataStreamer

from KImie import get_user_folder, KIMIE_LOGGER
from KImie.utils.files import response_to_filename
from KImie.dataloader.streamer import DfStreamer

CITED_SOURCES = []


class BaseDataLoader:
    source: str = None
    expected_data_size: int = -1
    data_streamer_generator: Callable = None
    default_metric = "rmse"
    citation = None

    def __init__(self, data_streamer_kwargs=None):
        if data_streamer_kwargs is None:
            data_streamer_kwargs = {}
        if self.data_streamer_generator is None:
            raise ValueError(
                f"no data_streamer_generator defined for {self.__class__.__name__}"
            )
        self._data_streamer = self.data_streamer_generator(**data_streamer_kwargs)
        if self.citation is not None and self.__class__ not in CITED_SOURCES:
            KIMIE_LOGGER.info(
                f"You are using a citable datasource ('{self.__class__.__name__}'), please consider citing '{self.citation}'!"
            )
            CITED_SOURCES.append(self.__class__)

    @property
    def data_streamer(self) -> DataStreamer:
        return self._data_streamer

    def __str__(self):
        return f"{self.__class__.__module__}.{self.__class__.__name__}"

    def __len__(self):
        return self.expected_data_size

    def __iter__(self):
        return (k for k in self._data_streamer)

    def get_data(self):
        return [d for d in self]

    def close(self):
        self._data_streamer.close()

    def get_n_entries(self, n: int, **kwargs):
        if n > self.expected_data_size:
            KIMIE_LOGGER.warning(
                "try to get %s entriers, but the loader has an expected size of %s",
                n,
                self.expected_data_size,
            )
        return self._data_streamer.get_n_entries(n=n, **kwargs)

    def get_all_entries(self, **kwargs):
        return self._data_streamer.get_all_entries(**kwargs)

    def split(
        self,
        split: list[float],
        method: str | Callable[[DataLoader, List[float]], list[DataLoader]] = "random",
        seed=None,
    ) -> list[DataLoader]:
        raise NotImplementedError("split not implemented for this data source")


class DataLoader(BaseDataLoader):
    raw_file: str = None
    local_source = None
    dl_chunk_size = 2**16

    def __init__(self, parent_dir=None, data_streamer_kwargs=None):
        super().__init__(data_streamer_kwargs=data_streamer_kwargs)
        if parent_dir is None:
            parent_dir = os.path.join(get_user_folder(), "dataloader", str(self))
        os.makedirs(parent_dir, exist_ok=True)
        self._parent_dir = parent_dir

    def _downlaod(self) -> str:
        if self.local_source is not None:
            import shutil

            trg = os.path.join(self.parent_dir, os.path.basename(self.local_source))
            shutil.copyfile(self.local_source, trg)
        else:
            KIMIE_LOGGER.debug(f"downloading {self.source}")
            fname = None

            try:
                if self.dl_chunk_size != 0:
                    response = requests.get(self.source, stream=True, timeout=10)
                    total_length = response.headers.get("content-length")

                    if total_length:
                        total_length = int(total_length)

                    fname = response_to_filename(response)
                    with open(
                        os.path.join(self.parent_dir, fname), "wb"
                    ) as handle, tqdm(
                        total=total_length, unit="byte", unit_scale=True
                    ) as pbar:
                        for data in response.iter_content(
                            chunk_size=self.dl_chunk_size
                        ):
                            handle.write(data)
                            pbar.update(len(data))
                else:
                    response = requests.get(self.source, timeout=10)
                    fn = response_to_filename(response)

                    with open(os.path.join(self.parent_dir, fname), "wb") as handle:
                        handle.write(response.content)
                trg = os.path.join(self.parent_dir, fname)
            except BaseException as e:
                if fname is not None:
                    if os.path.exists(os.path.join(self.parent_dir, fname)):
                        os.remove(os.path.join(self.parent_dir, fname))
                raise e
        KIMIE_LOGGER.debug("processing downloaded data")
        fp = self.process_download_data(trg)

        return fp

    def process_download_data(self, raw_file) -> str:
        return raw_file

    def download(self):
        KIMIE_LOGGER.debug(f"downlaod data from {self.source}")
        dl = self._downlaod()
        dl = os.path.abspath(dl)
        if not dl == self.raw_file_path:
            shutil.move(dl, self.raw_file_path)

    def delete(self):
        if not os.path.exists(self.raw_file_path):
            return
        if os.path.isdir(self.raw_file_path):
            shutil.rmtree(self.raw_file_path)
        else:
            os.remove(self.raw_file_path)

    def _needs_raw(self):
        if not os.path.exists(self.raw_file_path):
            self.download()

    @property
    def parent_dir(self) -> str:
        return self._parent_dir

    @property
    def raw_file_path(self) -> str:
        if self.raw_file is None:
            raise ValueError(f"no raw filename defined for {self.__class__.__name__}")
        return os.path.join(self.parent_dir, self.raw_file)

    def get_raw_file_path(self, load_if_not_exist=True):
        if load_if_not_exist:
            self._needs_raw()
        return self.raw_file_path

    def get_data(self, force_download=False):
        self._needs_raw()
        if force_download:
            self.download()
        return [d for d in self]

    def __iter__(self):
        self._needs_raw()
        return (k for k in self._data_streamer)

    def get_n_entries(self, n: int, **kwargs):
        self._needs_raw()
        if n > self.expected_data_size:
            KIMIE_LOGGER.warning(
                f"try to get {n} entriers, but the loader has an expected size of {self.expected_data_size}"
            )
        return self._data_streamer.get_n_entries(n=n, **kwargs)

    def get_all_entries(self, **kwargs):
        self._needs_raw()
        return self._data_streamer.get_all_entries(**kwargs)

    def to_csv(self, path, *args, **kwargs) -> pd.DataFrame:
        KIMIE_LOGGER.debug("convert to csv")
        df = self.to_df(*args, **kwargs)
        df.to_csv(
            path,
            index=False,
        )
        return df

    def to_df(self, *args, **kwargs):
        raise NotImplementedError("to_df not implemented for this data source")

    def to_df_loader(self, path=None):
        if path is None:
            path = os.path.join(self.parent_dir, "dataframe.csv")
        if not os.path.exists(path):
            self.to_csv(path)
        return DataFrameLoader(
            path=path,
        )


class DataFrameLoader(BaseDataLoader):
    data_streamer_generator = DfStreamer.generator()

    def __init__(self, path=None, df=None, *args, data_streamer_kwargs=None, **kwargs):
        if data_streamer_kwargs is None:
            data_streamer_kwargs = {}

        if path is None and df is None:
            raise ValueError("either path or df must be set")
        if df is None:
            df = pd.read_csv(path)
        self.df = df
        data_streamer_kwargs["data"] = self.df
        super().__init__(*args, data_streamer_kwargs=data_streamer_kwargs, **kwargs)
        self.expected_data_size = len(self.df)
        self.path = path

    def save(self):
        self.df.select_dtypes(exclude=["object"]).to_csv(self.path, index=False)
