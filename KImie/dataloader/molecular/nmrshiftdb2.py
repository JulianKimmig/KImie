import os

from KImie.dataloader.molecular.dataloader import MolDataLoader
from KImie.dataloader.molecular.streamer import SDFStreamer


class _HNMRERROR(Exception):
    pass


class NMRShiftDB2_1H(MolDataLoader):
    source = "https://sourceforge.net/projects/nmrshiftdb2/files/data/nmrshiftdb2withsignals.sd/download"
    raw_file = "nmrshiftdb2withsignals.sd"
    data_streamer_generator = SDFStreamer.generator(
        gz=False, file_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )

    dl_chunk_size = 0

    def process_download_data(self, raw_file):
        from KImie.utils.mol.properties import parallel_asset_conformers
        from rdkit import Chem
        import numpy as np
        from rdkit.Chem.PropertyMol import PropertyMol
        from tqdm import tqdm
        import pandas as pd

        mols = []
        with open(raw_file, "rb") as f:
            total_mols = sum([1 for _ in Chem.ForwardSDMolSupplier(f, sanitize=False)])

        with open(raw_file, "rb") as f:
            for mol in tqdm(
                Chem.ForwardSDMolSupplier(f, sanitize=False), total=total_mols
            ):
                if mol is None:
                    continue
                l = [s for s in mol.GetPropNames() if s.startswith("Spectrum 1H")]
                if len(l) > 0:
                    try:
                        nuc_indices = []
                        spec_string = mol.GetProp(l[0])
                        signals = [
                            dict(zip(["ppm", "_", "nuc"], ss.split(";")))
                            for ss in spec_string.split("|")
                            if len(ss) > 0
                        ]
                        for sig in signals:
                            sig["ppm"] = float(sig["ppm"])
                            sig["nuc"] = int(sig["nuc"])
                            sig["used"] = False
                            nuc_indices.append(sig["nuc"])

                        unique_nuc_indices = np.unique(nuc_indices)

                        for atom in mol.GetAtoms():
                            if atom.HasProp("molTotValence"):
                                atom.ClearProp("molTotValence")
                            nrad = atom.GetNumRadicalElectrons()
                            if nrad > 0:
                                atom.SetNumRadicalElectrons(0)
                                atom.SetNumExplicitHs(nrad)

                        Chem.SanitizeMol(mol)

                        sym_nodes = np.array(
                            [
                                list(s)
                                for s in mol.GetSubstructMatches(
                                    mol, uniquify=False, useChirality=True
                                )
                            ]
                        )

                        masks = ~np.diag(np.ones(sym_nodes.shape[1])).astype(bool)

                        unique_nodes = [
                            s
                            for _i, s in enumerate(sym_nodes[0])
                            if not (sym_nodes[1:, masks[_i]][:, _i:] == s).any()
                        ]
                        unique_nodesmap = [
                            np.unique(sym_nodes[:, sym_nodes[0] == n])
                            for n in unique_nodes
                        ]
                        for _i, nm in enumerate(unique_nodesmap.copy()):
                            is_found = False
                            to_rem = []
                            for n in nm:
                                if n not in unique_nuc_indices:
                                    continue
                                if not is_found:
                                    is_found = True
                                    continue
                                unique_nodesmap.append(np.array([n]))
                                to_rem.append(n)
                            for n in to_rem:
                                unique_nodesmap[_i] = np.delete(
                                    unique_nodesmap[_i],
                                    np.where((unique_nodesmap[_i] == n)),
                                )

                        mol = Chem.AddHs(mol)

                        for m in unique_nodesmap:
                            sigs = [
                                (s["ppm"], s["nuc"])
                                for s in signals
                                if (m == s["nuc"]).any()
                            ]
                            if len(sigs) == 0:
                                continue
                            hs = []
                            hs_by_atom = []
                            for a_idx in m:
                                atom = mol.GetAtomWithIdx(int(a_idx))
                                hba = []
                                for bond in atom.GetBonds():
                                    h = bond.GetOtherAtom(atom)
                                    if h.GetAtomicNum() != 1:
                                        continue
                                    hs.append(h)
                                    hba.append(h)
                                hs_by_atom.append(hba)

                            unique_sigs = list(set(sigs))
                            if len(hs) != len(sigs):
                                if len(sigs) == 1:
                                    sigs = sigs * len(hs)
                                # one signal for each proopn on same atom
                                elif np.unique([len(hba) for hba in hs_by_atom]).shape[
                                    0
                                ] == 1 and len(hs_by_atom[0]) == len(sigs):
                                    sigs = sigs * len(hs_by_atom[0])
                                # if all signals are identical but dont match the number of atoms
                                elif (
                                    np.unique([s[0] for s in sigs]).shape[0] == 1
                                    and np.unique([s[1] for s in sigs]).shape[0] == 1
                                ):
                                    sigs = [sigs[0]] * len(hs)
                                # if single atom and too many signals
                                elif len(hs_by_atom) == 1 and len(hs_by_atom[0]) == len(
                                    unique_sigs
                                ):
                                    sigs = unique_sigs
                                ## if single atom and little signals
                                # elif len(hs_by_atom)==1 and len(hs_by_atom[0])>len(sigs):
                                #    sigs=(sigs*len(hs_by_atom[0]))[:len(hs_by_atom[0])]
                                else:
                                    raise _HNMRERROR(f"{len(hs)}!={len(sigs)}")

                            for h, s in zip(hs, sigs):
                                h.SetProp("ppm", str(s[0]))

                        for p in mol.GetPropNames():
                            if (
                                p.startswith("Spectrum")
                                or p.startswith("Solvent")
                                or p.startswith("rawdata")
                            ):
                                mol.ClearProp(p)
                            elif p in [
                                "Temperature [K]",
                                "nmrshiftdb2 ID",
                                "Field Strength [MHz]",
                                "Assignment Method",
                                "NMRBasisSet",
                                "Program",
                                "GeomMethod",
                                "NMRModel",
                                "NMRLocalis",
                                "NMRStandard",
                                "GeomBasisSet",
                                "Machine",
                            ]:
                                mol.ClearProp(p)
                            else:
                                mol.ClearProp(p)

                        mol = PropertyMol(mol)
                        mol.SetProp(
                            "ppm",
                            ",".join(
                                [
                                    a.GetProp("ppm") if a.HasProp("ppm") else ""
                                    for a in mol.GetAtoms()
                                ]
                            ),
                        )
                        mol.RemoveAllConformers()
                        mols.append(mol)

                    except _HNMRERROR as e:
                        continue

        df = pd.DataFrame(mols, columns=["mol"])

        df.drop(df[df["mol"].apply(lambda x: x is None)].index, inplace=True)
        df["mol"] = parallel_asset_conformers(df["mol"])
        df.drop(df[df["mol"].apply(lambda x: x is None)].index, inplace=True)
        self.df_to_sdf(df, file=raw_file, mol_col="mol")
        return raw_file
