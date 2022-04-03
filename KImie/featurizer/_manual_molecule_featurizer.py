import numpy as np
from rdkit.Chem import GetMolFrags

from KImie.featurizer._molecule_featurizer import (
    MoleculeFeaturizer,
    SingleValueMoleculeFeaturizer,
)


class ExtendKImieFeaturizer(MoleculeFeaturizer):
    def featurize(self, mol):
        r = mol.kimie_features if hasattr(mol, "kimie_features") else []
        return np.array([r]).flatten()


extend_kimie_featurizer = ExtendKImieFeaturizer(name="extend_kimie_featurizer")


class NumFragments_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    # functions

    def featurize(self, mol):
        return len(GetMolFrags(mol))


molecule_num_fragments = NumFragments_Featurizer()

_available_featurizer = {
    "molecule_num_fragments": molecule_num_fragments,
}


def get_available_featurizer():
    return _available_featurizer


__all__ = [
    "NumFragments_Featurizer",
    "molecule_num_fragments",
]
