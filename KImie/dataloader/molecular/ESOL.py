from io import StringIO
from tempfile import gettempdir

import pandas as pd
from rdkit import Chem
from rdkit.Chem.PropertyMol import PropertyMol
from tqdm import tqdm

from KImie.dataloader.molecular.dataloader import MolDataLoader
from KImie.dataloader.molecular.streamer import SDFStreamer
from KImie.utils.mol.properties import parallel_asset_conformers


class ESOL(MolDataLoader):
    source = "https://pubs.acs.org/doi/suppl/10.1021/ci034243x/suppl_file/ci034243xsi20040112_053635.txt"
    raw_file = "delaney_data.sdf"
    expected_data_size = 1144
    citation = "https://doi.org/10.1021/ci034243x"
    data_streamer_generator = SDFStreamer.generator(
        gz=False, file_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )
    mol_properties = ["measured_log_solubility", "ESOL_predicted_log_solubility"]

    def process_download_data(self, raw_file):
        df = pd.read_csv(raw_file)
        df["mol"] = df["SMILES"].apply(lambda s: Chem.MolFromSmiles(s))
        df.drop(df[df["mol"] == None].index, inplace=True)
        df["mol"] = parallel_asset_conformers(df["mol"])
        df.drop(df[df["mol"] == None].index, inplace=True)
        df["mol"] = df["mol"].apply(lambda m: PropertyMol(m))
        for r, d in df.iterrows():
            d["mol"].SetProp("_Name", d["Compound ID"])
            d["mol"].SetProp(
                "measured_log_solubility", d["measured log(solubility:mol/L)"]
            )
            d["mol"].SetProp(
                "ESOL_predicted_log_solubility",
                d["ESOL predicted log(solubility:mol/L)"],
            )

        # needded string io since per default SDWriter appends $$$$ in the end and then a new line with results in an additional None entrie
        with StringIO() as f:
            with Chem.SDWriter(f) as w:
                for m in df["mol"]:
                    w.write(m)
            cont = f.getvalue()

        cont = "$$$$".join([c for c in cont.split("$$$$") if len(c) > 3])
        with open(raw_file, "w+") as f:
            f.write(cont)
        return raw_file
