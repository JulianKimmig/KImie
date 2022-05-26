from KImie.dataloader.molecular.dataloader import MolDataLoader
from KImie.dataloader.molecular.streamer import SDFStreamer


class ChemBLdbBase(MolDataLoader):
    citation = "https://doi.org/10.1093/nar/gkr777"
    mol_properties = []
    data_streamer_generator = SDFStreamer.generator(
        gz=True, file_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )


class ChemBLdb01(ChemBLdbBase):
    source = "http://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_01/chembl_01.sdf.gz"
    raw_file = "chembl_01.sdf.gz"
    expected_data_size = 440_055


class ChemBLdb29(ChemBLdbBase):
    source = "http://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_29/chembl_29.sdf.gz"
    raw_file = "chembl_29.sdf.gz"
    expected_data_size = 2_084_724


class ChemBLdb30(ChemBLdbBase):
    source = "http://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_30/chembl_30.sdf.gz"
    raw_file = "chembl_30.sdf.gz"
    expected_data_size = 2_136_187


# latest
ChemBLdbLatest = ChemBLdb30
