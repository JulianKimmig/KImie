import os

from KImie.dataloader.molecular.dataloader import MolDataLoader
from KImie.dataloader.molecular.streamer import SDFStreamer


class Tox21Train(MolDataLoader):
    source = "https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_data_allsdf"
    raw_file = "tox21_10k_data_all.sdf"
    expected_data_size = 11764
    expected_mol_count = 11758
    data_streamer_generator = SDFStreamer.generator(
        gz=False, file_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )

    mol_properties = [
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ]

    def process_download_data(self, raw_file):
        import zipfile

        with zipfile.ZipFile(raw_file, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(raw_file))
        os.remove(raw_file)
        return raw_file.rsplit(".zip", 1)[0]


Tox21 = Tox21Train
