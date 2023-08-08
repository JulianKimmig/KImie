from __future__ import annotations
from typing import List, Tuple, Dict, Any, Type, Callable, Literal, Generator, TypedDict

from KImie.dataloader.dataloader import DataLoader, DataFrameLoader, DfStreamer
from KImie.dataloader.streamer import MemoryStreamer

from rdkit.Chem.PropertyMol import PropertyMol
import pandas as pd

from KImie.utils.logging import KIMIE_LOGGER
from KImie.utils.data import normalize_split
import os


class MolDataTarget(TypedDict):
    field: str
    task_type: Literal["regression", "classification", "std"]
    unit: str
    description: str


class MolDataLoader(DataLoader):
    mol_properties: List[str] = None
    expected_mol_count: int = None

    targets: List[MolDataTarget] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.expected_mol_count is None:
            self.expected_mol_count = self.expected_data_size

    def __len__(self):
        return self.expected_mol_count

    def __iter__(self) -> Generator[PropertyMol, None, None]:
        return super().__iter__()

    def to_df(self, include_mol=True, smiles="smiles", mols="mol"):
        KIMIE_LOGGER.debug("convert to df")
        from rdkit.Chem import MolToSmiles
        from tqdm import tqdm

        data = []
        datacols = [smiles] + self.mol_properties
        if include_mol:
            datacols.append(mols)

        for m in tqdm(self, total=self.expected_mol_count, mininterval=1):
            if m is None:
                continue
            mol_dict = m.GetPropsAsDict()
            d = [MolToSmiles(m)] + [mol_dict.get(p, None) for p in self.mol_properties]
            if include_mol:
                d.append(m)
            data.append(d)
        return pd.DataFrame(data, columns=datacols)

    def to_csv(self, path, *args, **kwargs):
        return super().to_csv(path, *args, include_mol=False, **kwargs)

    def loads_csv(self, *args, csv_path=None, **kwargs):
        if csv_path is None:
            csv_path = os.path.join(self.parent_dir, "dataframe.csv")
        if not os.path.exists(csv_path):
            return self.to_csv(csv_path, *args, **kwargs)
        df = pd.read_csv(csv_path)
        df.path = csv_path
        return df

    @classmethod
    def df_canonize_smiles(cls, df, smiles="smiles", mols="mol"):
        KIMIE_LOGGER.debug("canonize smiles")
        from rdkit.Chem import MolFromSmiles, MolToSmiles

        df = cls.df_smiles_to_mol(df, smiles=smiles, mols=mols, conformers=False)
        df = cls.df_mol_to_smiles(df, smiles=smiles, mols=mols)
        return df

    @classmethod
    def df_remove_duplicate_smiles(cls, df, smiles="smiles", mols="mol", canonize=True):
        KIMIE_LOGGER.debug("remove duplicate smiles")
        inilength = len(df)
        df.drop_duplicates(subset=[smiles], inplace=True)

        if canonize:
            df = cls.df_canonize_smiles(df, smiles=smiles, mols=mols)
            df.drop_duplicates(subset=[smiles], inplace=True)
        KIMIE_LOGGER.debug(f"removed {inilength - len(df)} duplicate smiles")
        return df

    @classmethod
    def df_mol_to_smiles(cls, df, smiles="smiles", mols="mol"):
        from rdkit.Chem import MolToSmiles

        KIMIE_LOGGER.debug("convert mol to smiles")
        indices = df[df[mols].apply(lambda x: x is not None)].index
        df.loc[indices, smiles] = df.loc[indices, mols].apply(MolToSmiles)
        return df

    @classmethod
    def df_smiles_to_mol(
        cls, df, smiles="smiles", mols="mol", conformers=True, recalc=False
    ):
        KIMIE_LOGGER.debug("convert smiles to mol")
        from KImie.utils.mol.properties import parallel_asset_conformers
        from rdkit.Chem import MolFromSmiles
        from rdkit.Chem.PropertyMol import PropertyMol

        if len(df) == 0:
            return df

        if mols not in df.columns:
            df[mols] = None

        if recalc:
            indices = df.index
        else:
            indices = df[df[mols].apply(lambda x: x is None)].index

        KIMIE_LOGGER.debug(f"convert {len(indices)} smiles to mol")

        df.loc[indices, mols] = df.loc[indices, smiles].apply(MolFromSmiles)

        df.drop(df[df[mols].apply(lambda x: x is None)].index, inplace=True)

        if len(df) == 0:
            return df

        if conformers:
            kwargs = dict()
            if len(df) > 1_000:
                kwargs["split_parts"] = len(df) // 1_000
            df[mols] = parallel_asset_conformers(df[mols], **kwargs)
        df.drop(df[df[mols].apply(lambda x: x is None)].index, inplace=True)
        df[mols] = df[mols].apply(lambda m: PropertyMol(m))
        return df

    @classmethod
    def df_to_sdf(cls, df: pd.DataFrame, file: str = None, mol_col: str = "mol"):
        KIMIE_LOGGER.debug("convert df to sdf")
        from io import StringIO
        from rdkit.Chem.PropertyMol import PropertyMol
        from rdkit.Chem import SDWriter

        # needded string io since per default SDWriter appends $$$$ in the end and then a new line with results in an additional None entrie
        with StringIO() as f:
            with SDWriter(f) as w:
                for m in df[mol_col]:
                    assert isinstance(m, PropertyMol), "mol is not a PropertyMol"
                    w.write(m)
            cont = f.getvalue()

        cont = "$$$$".join([c for c in cont.split("$$$$") if len(c) > 3])
        if file:
            with open(file, "w+") as f:
                f.write(cont)
        return cont

    def _split_sorted(self, split: list[float]) -> list[MolDataLoader]:
        df = self.to_df(smiles="smiles", mols="mol")

        # split
        splits = []
        start = 0
        for s in split:
            end = start + int(s * len(df))
            splits.append(df.iloc[start:end].copy())
            start = end

        # create dataloaders
        dataloaders: List[MolDataLoader] = []
        for dfs in splits:
            dataloaders.append(
                moldataloader_from_df(
                    df=dfs,
                    name=f"{self}_S{'_'.join([str(s) for s in split])}",
                    smiles="smiles",
                    mols="mol",
                    properties=self.mol_properties,
                )()
            )
        return dataloaders

    def _split_random(self, split: list[float], seed=None) -> list[MolDataLoader]:
        df = self.to_df(smiles="smiles", mols="mol")

        # shuffle
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        # split
        splits = []
        start = 0
        for s in split:
            end = start + int(s * len(df))
            splits.append(df.iloc[start:end].copy())
            start = end

        # create dataloaders
        dataloaders: List[MolDataLoader] = []
        for dfs in splits:
            dataloaders.append(
                moldataloader_from_df(
                    df=dfs,
                    name=f"{self}_R{'_'.join([str(s) for s in split])}",
                    smiles="smiles",
                    mols="mol",
                    properties=self.mol_properties,
                )()
            )
        return dataloaders

    def split(
        self,
        split: list[float],
        method: Literal["random", "sorted"]
        | Callable[[DataLoader, List[float]], list[DataLoader]] = "random",
        seed=None,
    ) -> list[MolDataLoader]:
        split = normalize_split(split, n=len(split))
        if callable(method):
            return method(self, split)
        if method == "random":
            return self._split_random(split, seed=seed)
        if method == "sorted":
            return self._split_sorted(split)

        raise NotImplementedError(f"split method {method} not implemented")

    def to_df_loader(self, path=None):
        if path is None:
            path = os.path.join(self.parent_dir, "dataframe.csv")
        if not os.path.exists(path):
            self.to_csv(path)
        return MolDataFrameLoader(
            path=path,
        )


class MolDataFrameLoader(DataFrameLoader):
    data_streamer_generator = DfStreamer.generator()

    def smiles(self, col="smiles"):
        return self.df[col].tolist()

    def selfies(self, col="selfies", smiles="smiles"):
        if col not in self.df.columns:
            from KImie.utils.parallelization.multiprocessing import parallelize
            from KImie.utils.mol.descriptor import batch_smiles_to_selfies

            smiles = self.df[smiles].tolist()
            self.df[col] = parallelize(batch_smiles_to_selfies, smiles)
            self.save()

        selfies = self.df[col].tolist()
        # replace nans with None
        selfies = [None if type(s) == float else s for s in selfies]
        return selfies

    def mol(self, smiles="smiles"):
        if not hasattr(self, "_mols"):
            from KImie.utils.parallelization.multiprocessing import parallelize
            from KImie.utils.mol.descriptor import batch_smiles_to_mol

            smiles = self.df[smiles].tolist()
            self._mols = parallelize(batch_smiles_to_mol, smiles)
        return self._mols

    def save(self):
        super().save()


def moldataloader_from_df(
    df: pd.DataFrame,
    name: str,
    smiles: str = "smiles",
    mols: str = "mol",
    properties: List[str] = None,
) -> Type[MolDataLoader]:
    if properties is None:
        properties = [c for c in df.columns]
    if mols in properties:
        properties.remove(mols)
    if smiles in properties:
        properties.remove(smiles)

    for p in properties:
        assert p in df.columns, f"{p} is not a column of df"

    df = MolDataLoader.df_smiles_to_mol(df, smiles=smiles)
    for r, d in df.iterrows():
        for p in properties:
            d["mol"].SetProp(p, d[p])

    mols = df["mol"].tolist()
    # define new class based on MolDataLoader and set mol_properties to properties and expected_mol_count to len(df)

    new_class = type(
        name,
        (MolDataLoader,),
        {
            "mol_properties": properties,
            "expected_data_size": len(mols),
            "data_streamer_generator": MemoryStreamer.generator(data=mols),
            "__iter__": lambda self: (i for i in self.data_streamer),
            "__str__": lambda self: name,
        },
    )

    return new_class
