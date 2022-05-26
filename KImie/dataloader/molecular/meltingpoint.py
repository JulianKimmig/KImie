import pandas as pd
from KImie.dataloader.molecular.dataloader import MolDataLoader
from KImie.dataloader.molecular.streamer import SDFStreamer


class BradleyDoublePlusGoodMP(MolDataLoader):
    source = "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/1503991/BradleyDoublePlusGoodMeltingPointDataset.xlsx"
    raw_file = "BradleyDoublePlusGoodMP.sdf"
    expected_data_size = 3022
    citation = "http://dx.doi.org/10.6084/m9.figshare.1031637"
    data_streamer_generator = SDFStreamer.generator(
        gz=False, file_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )
    mol_properties = ["mpC"]

    def process_download_data(self, raw_file):
        df = pd.read_excel(raw_file)

        df = self.df_smiles_to_mol(df, "smiles")

        for r, d in df.iterrows():
            d["mol"].SetProp("_Name", d["name"])
            d["mol"].SetProp("mpC", d["mpC"])

        self.df_to_sdf(df, file=raw_file, mol_col="mol")
        return raw_file
