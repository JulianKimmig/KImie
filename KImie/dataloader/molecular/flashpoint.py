import pandas as pd

from KImie.dataloader.molecular.dataloader import MolDataLoader
from KImie.dataloader.molecular.streamer import SDFStreamer
import os


class MorganFlashpoint(MolDataLoader):
    source = "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/18509711/integrated_dataset_public.csv"
    raw_file = "morgan_flashpoint.sdf"
    expected_data_size = 14696
    citation = "https://doi.org/10.1002%2Fminf.201900101"
    data_streamer_generator = SDFStreamer.generator(
        gz=False, file_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )
    mol_properties = ["flashpoint"]

    def process_download_data(self, raw_file):
        df = pd.read_csv(raw_file)
        df = self.df_smiles_to_mol(df, "smiles")

        for r, d in df.iterrows():
            d["mol"].SetProp("_Name", d["compound"])
            d["mol"].SetProp("flashpoint", d["flashpoint"])

        self.df_to_sdf(df, file=raw_file, mol_col="mol")
        return raw_file
