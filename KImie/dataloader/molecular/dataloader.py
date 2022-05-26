from typing import List

from KImie.dataloader.dataloader import DataLoader


from rdkit.Chem.PropertyMol import PropertyMol
import pandas as pd


class MolDataLoader(DataLoader):
    mol_properties: List[str] = None
    expected_mol_count: int = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.expected_mol_count is None:
            self.expected_mol_count = self.expected_data_size

    def __len__(self):
        return self.expected_mol_count

    @classmethod
    def df_smiles_to_mol(cls, df, smiles="smiles"):
        from KImie.utils.mol.properties import parallel_asset_conformers
        from rdkit.Chem import MolFromSmiles
        from rdkit.Chem.PropertyMol import PropertyMol

        df["mol"] = df[smiles].apply(lambda s: MolFromSmiles(s))
        df.drop(df[df["mol"].apply(lambda x: x is None)].index, inplace=True)
        df["mol"] = parallel_asset_conformers(df["mol"])
        df.drop(df[df["mol"].apply(lambda x: x is None)].index, inplace=True)
        df["mol"] = df["mol"].apply(lambda m: PropertyMol(m))
        return df

    @classmethod
    def df_to_sdf(cls, df: pd.DataFrame, file: str = None, mol_col: str = "mol"):
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
