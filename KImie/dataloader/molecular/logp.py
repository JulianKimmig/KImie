import pandas as pd

from KImie.dataloader.molecular.dataloader import MolDataLoader
from KImie.dataloader.molecular.streamer import SDFStreamer
import os

from .ESOL import ESOL


class LogPNadinUlrich(MolDataLoader):
    source = "https://github.com/nadinulrich/log_P_prediction/raw/main/Dataset_and_Predictions.xlsx"
    raw_file = "logp_nadinulrich.sdf"
    expected_data_size = 50680
    citation = "https://doi.org/10.1038/s42004-021-00528-9"
    data_streamer_generator = SDFStreamer.generator(
        gz=False, file_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )
    mol_properties = ["logP_exp"]

    def process_download_data(self, raw_file):
        df = pd.read_excel(raw_file, sheet_name="model")
        df = self.df_remove_duplicate_smiles(df, smiles="SMILES")
        df = self.df_smiles_to_mol(df, "SMILES")

        for r, d in df.iterrows():
            d["mol"].SetProp("logP_exp", d["logP\nexperimental\n(corrected)"])

        self.df_to_sdf(df, file=raw_file, mol_col="mol")
        return raw_file
