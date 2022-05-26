from KImie import KIMIE_LOGGER

_available_featurizer = {}
__all__ = []

try:
    from KImie.featurizer._autogen import _autogen_rdkit_atomtype_atom_featurizer
    from KImie.featurizer._autogen._autogen_rdkit_atomtype_atom_featurizer import *

    for (
        n,
        f,
    ) in _autogen_rdkit_atomtype_atom_featurizer.get_available_featurizer().items():
        if n in _available_featurizer:
            KIMIE_LOGGER.warning(
                f"encoutered duplicate while collecting atom featurizer: {n}"
            )
            continue
        _available_featurizer[n] = f

    __all__ += _autogen_rdkit_atomtype_atom_featurizer.__all__
except ImportError as e:
    raise e
except Exception as e:
    KIMIE_LOGGER.exception(e)

try:
    from KImie.featurizer._autogen import _autogen_rdkit_feats_numeric_atom_featurizer
    from KImie.featurizer._autogen._autogen_rdkit_feats_numeric_atom_featurizer import *

    for (
        n,
        f,
    ) in (
        _autogen_rdkit_feats_numeric_atom_featurizer.get_available_featurizer().items()
    ):
        if n in _available_featurizer:
            KIMIE_LOGGER.warning(
                f"encoutered duplicate while collecting atom featurizer: {n}"
            )
            continue
        _available_featurizer[n] = f

    __all__ += _autogen_rdkit_feats_numeric_atom_featurizer.__all__
except ImportError as e:
    raise e
except Exception as e:
    KIMIE_LOGGER.exception(e)

try:
    from KImie.featurizer._autogen import _autogen_rdkit_feats_one_hot_atom_featurizer
    from KImie.featurizer._autogen._autogen_rdkit_feats_one_hot_atom_featurizer import *

    for (
        n,
        f,
    ) in (
        _autogen_rdkit_feats_one_hot_atom_featurizer.get_available_featurizer().items()
    ):
        if n in _available_featurizer:
            KIMIE_LOGGER.warning(
                f"encoutered duplicate while collecting atom featurizer: {n}"
            )
            continue
        _available_featurizer[n] = f

    __all__ += _autogen_rdkit_feats_one_hot_atom_featurizer.__all__
except ImportError as e:
    raise e
except Exception as e:
    KIMIE_LOGGER.exception(e)

try:
    from KImie.featurizer._autogen import _autogen_rdkit_feats_str_atom_featurizer
    from KImie.featurizer._autogen._autogen_rdkit_feats_str_atom_featurizer import *

    for (
        n,
        f,
    ) in _autogen_rdkit_feats_str_atom_featurizer.get_available_featurizer().items():
        if n in _available_featurizer:
            KIMIE_LOGGER.warning(
                f"encoutered duplicate while collecting atom featurizer: {n}"
            )
            continue
        _available_featurizer[n] = f

    __all__ += _autogen_rdkit_feats_str_atom_featurizer.__all__
except ImportError as e:
    raise e
except Exception as e:
    KIMIE_LOGGER.exception(e)


def get_available_featurizer():
    return _available_featurizer


def main():
    from rdkit import Chem
    from KImie.featurizer.molecule_featurizer import prepare_mol_for_featurization

    testmol = prepare_mol_for_featurization(Chem.MolFromSmiles("c1ccccc1")).GetAtoms()[
        -1
    ]
    for n, f in get_available_featurizer().items():
        print(n, end=" ")
        print(f(testmol))
    print(len(get_available_featurizer()))


if __name__ == "__main__":
    main()
