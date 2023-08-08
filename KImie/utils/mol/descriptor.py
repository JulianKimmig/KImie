from __future__ import annotations
from typing import Iterable, Set, Tuple, Iterator
from rdkit.Chem import Mol, MolFromSmiles, MolToSmiles
from tqdm import tqdm
import selfies as sf


def smiles_to_selfies(smiles: str) -> str:
    return sf.encoder(smiles)


def batch_smiles_to_selfies(smiles: Iterable[str]) -> list[str]:
    selfies = []
    for s in smiles:
        try:
            selfies.append(smiles_to_selfies(s))
        except sf.EncoderError:
            selfies.append(None)
    return selfies


def mol_to_selfies(mol):
    return smiles_to_selfies(mol_to_smiles(mol))


def batch_mol_to_selfies(mols: Iterable[Mol]) -> list[str]:
    selfies = []
    for m in mols:
        try:
            selfies.append(mol_to_selfies(m))
        except sf.EncoderError:
            selfies.append(None)
    return selfies


def selfies_to_smiles(selfies: str) -> str:
    return sf.decoder(selfies)


def batch_selfies_to_smiles(selfies: Iterable[str]) -> list[str]:
    return [selfies_to_smiles(s) for s in selfies]


def selfies_to_mol(selfies: str) -> Mol:
    return smiles_to_mol(selfies_to_smiles(selfies))


def batch_selfies_to_mol(selfies: Iterable[str]) -> list[Mol]:
    return [selfies_to_mol(s) for s in selfies]


def split_selfies(selfies: str) -> Iterator[str]:
    return sf.split_selfies(selfies)


def mol_to_smiles(mol: Mol) -> str:
    return MolToSmiles(mol)


def batch_mol_to_smiles(mols: Iterable[Mol]) -> list[str]:
    return [mol_to_smiles(m) for m in mols]


def smiles_to_mol(smiles: str) -> Mol:
    return MolFromSmiles(smiles)


def batch_smiles_to_mol(smiles: Iterable[str]) -> list[Mol]:
    return [smiles_to_mol(s) for s in smiles]


def selfie_alphabet_from_selfies(
    selfies: Iterable[str], as_dict=False
) -> Tuple[Set[str], int]:
    if as_dict:
        alphabet = dict()
        lens = {}
    else:
        alphabet = set()

    max_len = 0

    pcc = pml = cc = (0,)
    with tqdm(total=len(selfies), desc="Building alphabet") as pbar:
        for sel in selfies:
            if sel is None:
                pbar.update()
                continue
            try:
                split_selfies = list(sf.split_selfies(sel))
                pcc = len(alphabet)
                pml = max_len
                if as_dict:
                    for s in split_selfies:
                        alphabet[s] = alphabet.get(s, 0) + 1
                else:
                    alphabet.update(split_selfies)
                cc = len(alphabet)
                l = len(split_selfies)
                if as_dict:
                    lens[l] = lens.get(l, 0) + 1
                max_len = max(max_len, l)
                # add max_len as info to tqdm
                if cc != pcc or max_len != pml:
                    pbar.set_postfix(max_len=max_len, chars=cc)
                pbar.update()
            except Exception as e:
                print(sel)
                raise e
    if as_dict:
        return alphabet, lens
    return alphabet, max_len
