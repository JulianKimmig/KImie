from __future__ import annotations
from typing import Any, Dict, List, Union
import json
import numpy as np
from collections import defaultdict


def _make_json_serializable(data: Any):
    if isinstance(data, (int, float, str, bool)):
        return data
    if isinstance(data, (dict, defaultdict)):
        return {str(k): _make_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [_make_json_serializable(v) for v in data]

    # check iterable
    try:
        iter(data)
        return [_make_json_serializable(v) for v in data]
    except TypeError:
        pass

    if isinstance(data, np.ndarray):
        return _make_json_serializable(data.tolist())

    return data


def make_json_serializable(data: Any):
    return json.loads(json.dumps(_make_json_serializable(data)))


def dumps(data: Any, **kwargs):
    return json.dumps(make_json_serializable(data), **kwargs)
