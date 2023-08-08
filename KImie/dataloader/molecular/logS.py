import pandas as pd

from KImie.dataloader.molecular.dataloader import MolDataLoader, MolDataTarget
from KImie.dataloader.molecular.streamer import SDFStreamer
import numpy as np


class AqSolDB(MolDataLoader):
    """
    Curation of nine open source datasets on aqueous solubility.
    The authors also assigned reliability groups.
    """

    source = "https://dataverse.harvard.edu/api/access/datafile/3407241?format=original&gbrecs=true"
    raw_file = "aqsol-curated-solubility-dataset.sdf"
    expected_data_size = 9973
    citation = "https://doi.org/10.1038/s41597-019-0151-1"
    data_streamer_generator = SDFStreamer.generator(
        gz=False, file_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )
    mol_properties = ["logS"]

    targets = [
        MolDataTarget(
            field="logS",
            task_type="regression",
            unit="log(mol/L)",
            description="LogS, where S is the aqueous solubility in mol/L",
        ),
    ]

    def process_download_data(self, raw_file):
        df = pd.read_csv(raw_file)
        df = self.df_smiles_to_mol(df, "SMILES")

        for r, d in df.iterrows():
            d["mol"].SetProp("_Name", d["Name"])
            d["mol"].SetProp("logS", d["Solubility"])
        self.df_to_sdf(df, file=raw_file, mol_col="mol")
        return raw_file
