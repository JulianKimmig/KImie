import numpy as np

from KImie import KIMIE_LOGGER
from KImie.featurizer._atom_featurizer import SingleValueAtomFeaturizer
from KImie.featurizer.featurizer import FeaturizerList

_available_featurizer = {}
__all__ = []
try:
    from KImie.featurizer import _manual_atom_featurizer
    from KImie.featurizer._manual_atom_featurizer import *

    for n, f in _manual_atom_featurizer.get_available_featurizer().items():
        if n in _available_featurizer:
            KIMIE_LOGGER.warning(
                f"encoutered duplicate while collecting moelcule featurizer: {n}"
            )
            continue
        _available_featurizer[n] = f

    __all__ += _manual_atom_featurizer.__all__
except ImportError as e:
    raise e
except Exception as e:

    KIMIE_LOGGER.exception(e)

try:
    from KImie.featurizer import _autogen_atom_featurizer
    from KImie.featurizer._autogen_atom_featurizer import *

    for n, f in _autogen_atom_featurizer.get_available_featurizer().items():
        if n in _available_featurizer:
            n = f"autogen_atom_featurizer_{n}"
        if n in _available_featurizer:
            KIMIE_LOGGER.warning(
                f"encoutered duplicate while collecting moelcule featurizer: {n}"
            )
            continue
        _available_featurizer[n] = f

    __all__ += _autogen_atom_featurizer.__all__
except ImportError as e:
    raise e
except Exception as e:
    KIMIE_LOGGER.exception(e)


class AllSingleValueAtomFeaturizer(FeaturizerList):
    dtype = np.float32

    def __init__(self, *args, **kwargs):
        super().__init__(
            [
                f
                for n, f in _available_featurizer.items()
                if isinstance(f, SingleValueAtomFeaturizer)
            ],
            *args,
            **kwargs,
        )


atom_all_single_val_feats = AllSingleValueAtomFeaturizer(
    name="atom_all_single_val_feats"
)
__all__.extend(["atom_all_single_val_feats", "AllSingleValueAtomFeaturizer"])
_available_featurizer["atom_all_single_val_feats"] = atom_all_single_val_feats


def get_available_featurizer():
    return _available_featurizer

