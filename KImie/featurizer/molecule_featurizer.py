from typing import Dict

from KImie.featurizer._molecule_featurizer import *
from KImie.featurizer._molecule_featurizer import (
    _MoleculeFeaturizer,
    OneHotMoleculeFeaturizer,
)
from KImie.featurizer.featurizer import FeaturizerList

_available_featurizer: Dict[str, _MoleculeFeaturizer] = {}
__all__ = []
try:
    from KImie.featurizer import _manual_molecule_featurizer
    from KImie.featurizer._manual_molecule_featurizer import *

    for n, f in _manual_molecule_featurizer.get_available_featurizer().items():
        if n in _available_featurizer:
            KIMIE_LOGGER.warning(
                f"encoutered duplicate while collecting molecule featurizer: {n}"
            )
            continue
        _available_featurizer[n] = f

    __all__ += _manual_molecule_featurizer.__all__
except ImportError as e:
    raise e
except Exception as e:
    KIMIE_LOGGER.exception(e)

try:
    from KImie.featurizer import _autogen_molecule_featurizer
    from KImie.featurizer._autogen_molecule_featurizer import *

    for n, f in _autogen_molecule_featurizer.get_available_featurizer().items():
        if n in _available_featurizer:
            n = f"autogen_molecule_featurizer_{n}"
        if n in _available_featurizer:
            KIMIE_LOGGER.warning(
                f"encoutered duplicate while collecting molecule featurizer: {n}"
            )
            continue
        _available_featurizer[n] = f

    __all__ += _autogen_molecule_featurizer.__all__
except ImportError as e:
    raise e
except Exception as e:
    KIMIE_LOGGER.exception(e)


import selfies
from rdkit.Chem import MolToSmiles
from selfies import split_selfies, encoder as selfies_encoder


class SEFIESOneHotMoleculeFeaturizer(OneHotMoleculeFeaturizer):
    POSSIBLE_VALUES = [
        "[C]",
        "[=C]",
        "[Branch1]",
        "[S]",
        "[#Branch1]",
        "[N]",
        "[=N]",
        "[Ring1]",
        "[=Branch2]",
        "[C@H1]",
        "[Branch2]",
        "[Ring2]",
        "[=Branch1]",
        "[=O]",
        "[C@@H1]",
        "[O]",
        "[NH1]",
        "[#Branch2]",
        "[#C]",
        "[P]",
        "[Cl]",
        "[Br]",
        "[/C]",
        "[\\C@@H1]",
        "[C@]",
        "[C@@]",
        "[N+1]",
        "[=Ring1]",
        ".",
        "[Br-1]",
        "[I]",
        "[F]",
        "[/N]",
        "[=Ring2]",
        "[=S]",
        "[O-1]",
        "[/C@@H1]",
        "[\\C]",
        "[\\C@H1]",
        "[#N]",
        "[/O]",
        "[Branch3]",
        "[Na+1]",
        "[/C@H1]",
        "[\\N]",
        "[=N+1]",
        "[Cl-1]",
        "[I-1]",
        "[Ring3]",
        "[\\O]",
        "[Si]",
        "[\\NH1]",
        "[=N-1]",
        "[=P]",
        "[18F]",
        "[\\S]",
        "[B]",
        "[S+1]",
        "[2H]",
        "[P+1]",
        "[/S]",
        "[N-1]",
        "[PH1]",
        "[B-1]",
        "[OH0]",
        "[/Br]",
        "[Se]",
        "[Li+1]",
        "[P@]",
        "[/N+1]",
        "[P@@]",
        "[3H]",
        "[K+1]",
        "[Cl+3]",
        "[S-1]",
        "[C-1]",
        "[#N+1]",
        "[noq]",
    ]

    SHAPE = (len(POSSIBLE_VALUES), 200)

    def featurize(self, mol):
        return list(split_selfies(selfies_encoder(MolToSmiles(mol))))


class AllSingleValueMoleculeFeaturizer(FeaturizerList):
    dtype = np.float32

    def __init__(self, *args, **kwargs):
        super().__init__(
            [
                f
                for n, f in _available_featurizer.items()
                if isinstance(f, SingleValueMoleculeFeaturizer)
            ],
            *args,
            **kwargs,
        )


molecule_all_single_val_feats = AllSingleValueMoleculeFeaturizer(
    name="molecule_all_single_val_feats"
)

__all__.extend(["molecule_all_single_val_feats", "AllSingleValueMoleculeFeaturizer"])
_available_featurizer["molecule_all_single_val_feats"] = molecule_all_single_val_feats


def get_available_featurizer() -> Dict[str, _MoleculeFeaturizer]:
    return _available_featurizer
