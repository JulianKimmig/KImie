import json
import shutil

import pandas as pd

import os


from KImie.dataloader.molecular.dataloader import MolDataLoader
from KImie.dataloader.molecular.streamer import SDFStreamer


class FreeSolv_0_51(MolDataLoader):
    source = "https://escholarship.org/content/qt6sd403pz/supp/FreeSolv-0.51.zip"
    raw_file = "freesolv-0.51.sdf"
    expected_data_size = 643
    citation = "https://doi.org/10.1007/s10822-014-9747-x"
    data_streamer_generator = SDFStreamer.generator(
        gz=False, file_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )
    mol_properties = ["expt", "d_expt", "calc", "d_calc"]

    def process_download_data(self, raw_file):
        import zipfile

        with zipfile.ZipFile(raw_file, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(raw_file))
        files_to_del = list(os.listdir(os.path.dirname(raw_file)))
        with open(
            os.path.join(os.path.dirname(raw_file), "FreeSolv-0.51", "database.json"),
            "r",
        ) as f:
            dict = json.loads(f.read())

        for f in files_to_del:
            if os.path.isfile(os.path.join(os.path.dirname(raw_file), f)):
                os.remove(os.path.join(os.path.dirname(raw_file), f))
            elif os.path.isdir(os.path.join(os.path.dirname(raw_file), f)):
                shutil.rmtree(os.path.join(os.path.dirname(raw_file), f))

        df = pd.DataFrame.from_dict(dict, orient="index")
        df = df[["iupac", "smiles", "expt", "d_expt", "calc", "d_calc"]]

        df = self.df_smiles_to_mol(df, "smiles")

        for r, d in df.iterrows():
            d["mol"].SetProp("_Name", d["iupac"])
            d["mol"].SetProp("expt", d["expt"])
            d["mol"].SetProp("d_expt", d["d_expt"])
            d["mol"].SetProp("calc", d["calc"])
            d["mol"].SetProp("d_calc", d["d_calc"])

        self.df_to_sdf(df, file=raw_file, mol_col="mol")

        return raw_file


FreeSolv = FreeSolv_0_51
