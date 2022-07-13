from rdkit import Chem
import pandas as pd
import os
import numpy as np
from rdkit.Chem import GetAdjacencyMatrix
from typing import Tuple
def _gen_atom_n():
    d = {}
    em = Chem.RWMol()
    em.AddAtom(Chem.Atom(6))
    i = 0
    while True:
        try:
            em.GetAtoms()[0].SetAtomicNum(i)
            d[em.GetAtoms()[0].GetSymbol()] = i
        except RuntimeError:
            break
        i += 1
    return d


ATOMIC_SYMBOL_NUMBERS = _gen_atom_n()

ATOM_PROPS = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "atom_props.csv"), index_col=0
)


def mol_to_graph_data(mol:Chem.Mol) -> Tuple[np.ndarray,np.ndarray]:
    adj_matrix = GetAdjacencyMatrix(mol)
    nodes = np.arange(adj_matrix.shape[0])
    row, col = np.where(adj_matrix)
    adj_list = np.unique(np.sort(np.vstack((row, col)).T, axis=1), axis=0)

    return nodes, adj_list