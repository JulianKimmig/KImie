from rdkit.Chem.rdchem import Atom

from KImie import KIMIE_LOGGER
from KImie.featurizer._molecule_featurizer import (
    prepare_mol_for_featurization,
    check_mol_is_prepared,
)
from KImie.featurizer.featurizer import (
    Featurizer,
    FixedSizeFeaturizer,
    OneHotFeaturizer,
    StringFeaturizer,
)


def atom_pre_featurize(featurizer, atom: Atom):
    mol = atom.GetOwningMol()

    if not check_mol_is_prepared(mol):
        if not featurizer._unprepared_logged:
            KIMIE_LOGGER.warning(
                "you tried to featurize an atom without previous preparation of the molecule. "
                "I will do this for you, but please try to implement this, "
                "otherwise you might end uo with differences in yout molecules and the featurized,"
                " since the preparation creates an copy of the molecule, "
                "adds hydrogens, conformerst etc."
                ""
            )
            featurizer._unprepared_logged = True

        from rdkit.Chem.rdchem import Mol

        mol: Mol = prepare_mol_for_featurization(mol, renumber=False)
        atom = mol.GetAtomWithIdx(atom.GetIdx())
    return atom


class _AtomFeaturizer(Featurizer):
    def __init__(self, *args, **kwargs):
        self._unprepared_logged = False
        super().__init__(*args, **kwargs)
        self.prepend_prefeaturizer(atom_pre_featurize, "atom_pre_featurize")


class VarSizeAtomFeaturizer(_AtomFeaturizer, Featurizer):
    pass


AtomFeaturizer = VarSizeAtomFeaturizer


class FixedSizeAtomFeaturizer(_AtomFeaturizer, FixedSizeFeaturizer):
    pass


class SingleValueAtomFeaturizer(FixedSizeAtomFeaturizer):
    LENGTH = 1


class OneHotAtomFeaturizer(_AtomFeaturizer, OneHotFeaturizer):
    pass


class StringAtomFeaturizer(_AtomFeaturizer, StringFeaturizer):
    pass
