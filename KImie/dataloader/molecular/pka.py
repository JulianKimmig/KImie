import pandas as pd

from KImie.dataloader.molecular.dataloader import MolDataLoader
from KImie.dataloader.molecular.streamer import SDFStreamer
from KImie.dataloader.molecular.dataloader import MolDataLoader, moldataloader_from_df
import os
import numpy as np


class IUPAC_DissociationConstantsV1_0(MolDataLoader):
    source = "https://zenodo.org/record/7236453/files/IUPAC/Dissociation-Constants-v1-0_initial-release.zip?download=1"
    raw_file = "IUPAC_DissociationConstantsV1_0.sdf"
    expected_data_size = 11819
    citation = "https://doi.org/10.5281/zenodo.7236453"
    data_streamer_generator = SDFStreamer.generator(
        gz=False, file_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )
    mol_properties = ["pka_value", "T"]

    def process_download_data(self, raw_file):
        # unzip raw file to same folder
        import zipfile

        with zipfile.ZipFile(raw_file, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(raw_file))
        os.remove(raw_file)
        df = pd.read_csv(
            os.path.join(
                os.path.dirname(raw_file),
                "IUPAC-Dissociation-Constants-ea456a0",
                "iupac_high-confidence_v1_0.csv",
            )
        )

        pks_to_keep = ["pKAH1", "pK1"]
        df.drop(df.index[(~df["pka_type"].isin(pks_to_keep))], inplace=True)
        df.reset_index(inplace=True, drop=True)

        def to_float(x):
            try:
                return float(x)
            except:
                return np.nan

        df["T"] = df["T"].apply(to_float)
        df.drop(df.index[(df["T"].isna())], inplace=True)

        df["pka_value"] = df["pka_value"].apply(to_float)
        df.drop(df.index[(df["pka_value"].isna())], inplace=True)

        df = self.df_smiles_to_mol(df, "SMILES")

        for r, d in df.iterrows():
            d["mol"].SetProp("pka_value", d["pka_value"])
            d["mol"].SetProp("T", d["T"])

        self.df_to_sdf(df, file=raw_file, mol_col="mol")
        return raw_file


def IUPAC_DissociationConstantsV1_0T25_5(*args, **kwargs):
    ds = IUPAC_DissociationConstantsV1_0()
    df = IUPAC_DissociationConstantsV1_0().to_df()
    df = df[(df["T"] <= 30) & (df["T"] >= 20)].copy()
    df["distTo25"] = np.abs(df["T"] - 25)
    df.sort_values("distTo25", inplace=True)
    df = MolDataLoader.df_canonize_smiles(df, smiles="smiles")
    df.drop_duplicates(subset=["smiles"], inplace=True)
    df.reset_index(inplace=True, drop=True)

    return moldataloader_from_df(df, f"{ds}T25_5")(*args, **kwargs)
