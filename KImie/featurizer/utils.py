from KImie.featurizer.prefeaturizer import Prefeaturizer
import numpy as np


def merge_atom_featurizer_data(
    atom_featurizer: Prefeaturizer, return_split_indices: bool = False
):
    """
    Merges the data of the atom_featurizer into a single array and returns
    """
    ipd = []
    split_indices = []
    for p in atom_featurizer:
        ipd.append(p)
        split_indices.append(p.shape[0])
    split_indices = np.cumsum(split_indices)[:-1]
    ipd = np.concatenate(ipd, axis=0)
    return ipd, split_indices


def split_atom_featurizer_data(feature_array: np.ndarray, split_indices: np.ndarray):
    """
    Splits the data of the atom_featurizer into a single array and returns
    """

    assert split_indices.ndim == 1
    split_indices = np.unique(split_indices)
    assert split_indices.shape[0] <= feature_array.shape[0]
    assert split_indices[0] > 0
    assert split_indices[-1] < feature_array.shape[0]

    return np.split(feature_array, split_indices)


def reduce_features(
    data,
    feature_names=None,
    min_rel_content: float = 0.0,
    max_rel_content: float = None,
):
    """
    reduces an N,F feature matrix with N samples and F features to a reduced feature matrix with N samples and K features
    where K is the number of features that are not removed.
    The features are selected based on their relative content. if a feture has an unique element with a relative content larger or equal than max_rel_content or smaller than min_rel_content, it is removed.

    Parameters
    ----------
    data : np.ndarray
        N,F feature matrix
    feature_names : list of str, optional
        list of feature names
    min_rel_content : float, default=0.0
        if a feature has an unique element with a relative content smaller or equal than min_rel_content, it is removed.
    max_rel_content : float, default=1-min_rel_content
        if a feature has an unique element with a relative content larger or equal than max_rel_content, it is removed.
    """

    if min_rel_content > 1:
        min_rel_content = 1
    if min_rel_content < 0:
        min_rel_content = 0
    if max_rel_content is None:
        max_rel_content = 1 - min_rel_content
    if max_rel_content > 1:
        max_rel_content = 1
    if max_rel_content < min_rel_content:
        raise ValueError("max_rel_content must be larger than min_rel_content")

    assert data.ndim == 2, "data must be 2D"

    l = data.shape[0]
    remove_col = np.zeros(data.shape[1], dtype=bool)
    for d in range(data.shape[1]):
        uniques, indices, counts = np.unique(
            data[:, d], axis=0, return_counts=True, return_index=True
        )
        # print(uniques.shape,counts.shape,indices.shape)
        rel_counts = counts / l
        over_th = (rel_counts <= min_rel_content) | (rel_counts >= max_rel_content)
        if np.all(over_th):
            remove_col[d] = True

    data = data[:, ~remove_col]

    if feature_names is not None:
        feature_names = np.array(feature_names)[~remove_col]
        return data, feature_names.tolist()
    return data
