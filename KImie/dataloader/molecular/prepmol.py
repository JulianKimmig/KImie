# import relative if not in sys path
import os
import pickle

import numpy as np
import pandas as pd
from KImie.featurizer.molecule_featurizer import prepare_mol_for_featurization

from tqdm import tqdm

from KImie import KIMIE_LOGGER, get_user_folder
from KImie.dataloader.molecular.dataloader import MolDataLoader
from KImie.dataloader.molecular.streamer import PickledMolStreamer
from KImie.dataloader.streamer import NumpyStreamer, CSVStreamer
from KImie.utils.mol import mol_to_graph_data


class PreparedMolDataLoader(MolDataLoader):
    raw_file = "mols"
    data_streamer_generator = PickledMolStreamer.generator(
        folder_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )

    def __init__(self, mdl: MolDataLoader, parent_dir=None, **kwargs):
        assert not isinstance(mdl, PreparedMolDataLoader)
        assert isinstance(mdl, MolDataLoader)

        if parent_dir is None:
            parent_dir = os.path.join(
                get_user_folder(), "dataloader", f"{mdl}_prepared"
            )
        self._mdl = mdl
        super().__init__(parent_dir=parent_dir, **kwargs)
        self.expected_data_size = mdl.expected_mol_count
        self.expected_mol_count = mdl.expected_mol_count

    def __str__(self):
        return str(self._mdl)

    def __len__(self):
        return len(self._mdl)

    def _needs_raw(self):
        if not os.path.exists(self.raw_file_path):
            os.makedirs(self.raw_file_path, exist_ok=True)
        molfiles = [f for f in os.listdir(self.raw_file_path) if f.endswith(".mol")]
        if len(molfiles) < self.expected_mol_count:
            for i, mol in self._raw_gen(desc="generate prepared mols"):
                # for i,mol in enumerate(tqdm(self._mdl,total=self.expected_data_size,desc="generate prepared mols")):
                #    if mol is None:
                #        continue
                _path = os.path.join(self.raw_file_path, f"{i}.mol")
                if not os.path.exists(_path):
                    pmol = prepare_mol_for_featurization(mol)
                    with open(os.path.join(self.raw_file_path, f"{i}.mol"), "w+b") as f:
                        pickle.dump(pmol, f)

    def _raw_gen(self, desc=None):
        KIMIE_LOGGER.debug("generate raw mols")
        pin = self._mdl.data_streamer._iter_None
        self._mdl.data_streamer._iter_None = True
        self._mdl.close()
        for i, mol in enumerate(
            tqdm(self._mdl, total=self._mdl.expected_data_size, desc=desc)
        ):
            if mol is None:
                continue
            yield i, mol
        self._mdl.data_streamer._iter_None = pin


class PreparedMolAdjacencyListDataLoader(PreparedMolDataLoader):
    raw_file = "adjacency_list"
    data_streamer_generator = NumpyStreamer.generator(
        folder_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )

    def __str__(self):
        return f"PreparedMolAdjacencyListDataLoader_{self._mdl}"

    def _needs_raw(self):
        if not os.path.exists(self.raw_file_path):
            os.makedirs(self.raw_file_path, exist_ok=True)
        adj_files = [f for f in os.listdir(self.raw_file_path) if f.endswith(".npy")]
        if len(adj_files) < self.expected_mol_count:
            for i, mol in self._raw_gen(desc="generate prepared mols adjecency list"):
                mol = prepare_mol_for_featurization(mol)
                nodes, edge_list = mol_to_graph_data(mol)
                np.save(os.path.join(self.raw_file_path, f"{i}.npy"), edge_list)


class PreparedMolPropertiesDataLoader(PreparedMolDataLoader):
    raw_file = "mol_props.csv"
    data_streamer_generator = CSVStreamer.generator(
        file_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )

    def __str__(self):
        return f"PreparedMolPropertiesDataLoader_{self._mdl}"

    def _needs_raw(self):
        path = os.path.dirname(self.raw_file_path)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        all_props = set()
        if self._mdl.mol_properties is None:
            KIMIE_LOGGER.warning(
                "No mol properties found in dataloader, will read all mols and check for properties, "
                "this takes a while, depending on the dataset size. You can skip this by predefining "
                "the 'mol_properties' attribute in the dataloader (list of strings)."
            )
            self._mdl.close()
            for i, mol in self._raw_gen(desc="detetct mol properties"):
                all_props.update(mol.GetPropNames())
            all_props = list(all_props)
        else:
            all_props = self._mdl.mol_properties
        datacols = ["mol_idx"] + all_props
        new_file = False

        def _gen_file():
            df = pd.DataFrame(columns=datacols)
            df.to_csv(self.raw_file_path, index=False)

        # make shure that the file exists
        if not os.path.exists(self.raw_file_path):
            KIMIE_LOGGER.debug(f"{self.raw_file_path} does not exist, creating it.")
            _gen_file()
            new_file = True

        # reset data is something is wrong with the columns
        if not new_file:
            for idf in self.data_streamer:
                if len(idf.columns) != len(all_props):
                    KIMIE_LOGGER.warning(
                        f"{self.raw_file_path} has wrong column length, resetting it."
                    )
                    _gen_file()
                    new_file = True
                    break
                if not all(idf.columns[i] == c for i, c in enumerate(all_props)):
                    KIMIE_LOGGER.warning(
                        f"{self.raw_file_path} has wrong columns, will be reset"
                    )
                    _gen_file()
                    new_file = True
                    break
                if idf.index.name != datacols[0]:
                    KIMIE_LOGGER.warning(
                        f"{self.raw_file_path} has wrong index name, will be reset"
                    )
                    _gen_file()
                    new_file = True
                    break
                break
            self.data_streamer.close()

        if not new_file:
            # get line_count
            def _make_gen(reader):
                while True:
                    b = reader(2**16)
                    if not b:
                        break
                    yield b

            with open(self.raw_file_path, "rb") as f:
                count = (
                    sum(buf.count(b"\n") for buf in _make_gen(f.raw.read)) - 1
                )  # minus header

            if count != self.expected_mol_count:
                KIMIE_LOGGER.warning(
                    f"{self.raw_file_path} has wrong number of lines (found {count}, needs {self.expected_mol_count}), will be reset"
                )
                _gen_file()
                new_file = True

        chunk_size = self.data_streamer.chunksize
        if chunk_size is None:
            chunk_size = 10_000

        data = []
        if new_file:
            for i, mol in self._raw_gen(desc="generate mol prop csv"):
                #            self._mdl.close()
                #            for i,mol in enumerate(tqdm(self._mdl,total=self.expected_mol_count,desc="mol_prop_csv")):
                #                if mol is None:
                #                    continue
                mol_dict = mol.GetPropsAsDict()
                data.append([i] + [mol_dict.get(p, None) for p in all_props])
                if len(data) >= chunk_size:
                    df = pd.DataFrame(data, columns=datacols)
                    df.to_csv(self.raw_file_path, mode="a", header=False, index=False)
                    data = []
            if len(data) >= 0:
                df = pd.DataFrame(data, columns=datacols)
                df.to_csv(self.raw_file_path, mode="a", header=False, index=False)
