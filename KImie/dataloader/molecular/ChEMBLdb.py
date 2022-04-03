from KImie.dataloader.molecular.dataloader import MolDataLoader
from KImie.dataloader.molecular.streamer import SDFStreamer


class ChemBLdbBase(MolDataLoader):
    citation = "https://doi.org/10.1093/nar/gkr777"
    mol_properties = []


class ChemBLdb01(ChemBLdbBase):
    source = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_01/chembl_01.sdf.gz"
    raw_file = "chembl_01.sdf.gz"
    expected_data_size = 440055
    data_streamer_generator = SDFStreamer.generator(
        gz=True, file_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )


class ChemBLdb29(ChemBLdbBase):
    source = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_29/chembl_29.sdf.gz"
    raw_file = "chembl_29.sdf.gz"
    expected_data_size = 2084724
    data_streamer_generator = SDFStreamer.generator(
        gz=True, file_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )


# latest
ChemBLdbLatest = ChemBLdb29
