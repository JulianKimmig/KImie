import pandas as pd
import os

from KImie.dataloader.molecular.dataloader import MolDataLoader
from KImie.dataloader.molecular.streamer import SDFStreamer


class Lipo1(MolDataLoader):
    raw_file = "lipo1.sdf"
    expected_data_size = 4200
    citation = "https://doi.org/10.6019/CHEMBL3301361"
    data_streamer_generator = SDFStreamer.generator(
        gz=False, file_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )
    mol_properties = ["exp"]

    local_source = os.path.join(os.path.dirname(__file__), "local", "Lipophilicity.csv")

    def process_download_data(self, raw_file):
        df = pd.read_csv(raw_file)
        df = self.df_smiles_to_mol(df, "smiles")
        for r, d in df.iterrows():
            d["mol"].SetProp("exp", d["exp"])
            d["mol"].SetProp("CMPD_CHEMBLID", d["CMPD_CHEMBLID"])

        self.df_to_sdf(df, file=raw_file, mol_col="mol")
        return raw_file
