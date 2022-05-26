import pandas as pd

from KImie.dataloader.molecular.dataloader import MolDataLoader
from KImie.dataloader.molecular.streamer import SDFStreamer
import os


class ESOL(MolDataLoader):
    source = "https://pubs.acs.org/doi/suppl/10.1021/ci034243x/suppl_file/ci034243xsi20040112_053635.txt"
    raw_file = "delaney_data.sdf"
    expected_data_size = 1144
    citation = "https://doi.org/10.1021/ci034243x"
    data_streamer_generator = SDFStreamer.generator(
        gz=False, file_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )
    mol_properties = ["measured_log_solubility", "ESOL_predicted_log_solubility"]
    local_source = os.path.join(
        os.path.dirname(__file__), "local", "ci034243xsi20040112_053635.txt"
    )

    def process_download_data(self, raw_file):
        df = pd.read_csv(raw_file)
        df = self.df_smiles_to_mol(df, "SMILES")

        for r, d in df.iterrows():
            d["mol"].SetProp("_Name", d["Compound ID"])
            d["mol"].SetProp(
                "measured_log_solubility", d["measured log(solubility:mol/L)"]
            )
            d["mol"].SetProp(
                "ESOL_predicted_log_solubility",
                d["ESOL predicted log(solubility:mol/L)"],
            )

        self.df_to_sdf(df, file=raw_file, mol_col="mol")
        return raw_file
