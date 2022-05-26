from typing import Dict

from KImie.featurizer._molecule_featurizer import *
from KImie.featurizer._molecule_featurizer import _MoleculeFeaturizer
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
