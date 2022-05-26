import unittest

from KImie.dataloader.molecular.ESOL import ESOL
from KImie.dataloader.molecular.prepmol import PreparedMolDataLoader
from KImie.featurizer import get_molecule_featurizer_info, get_atom_featurizer_info
from KImie.featurizer.featurizer import FeaturizerList
from tests._kimie_test_base import KImieTest
import numpy as np

class MolecularFeaturizerTest(KImieTest):
    def setUp(self) -> None:
        laoder = PreparedMolDataLoader(ESOL())
        for m in laoder:
            if m is not None:
                self.testmol = m
                break
        assert self.testmol is not None, "No molecule found in ESOL"

    def test_featurizer_info(self):
        assert len(get_molecule_featurizer_info()) > 0, "No molecule featurizer found"
        assert (
            len(get_molecule_featurizer_info()) >= 100
        ), get_molecule_featurizer_info()

    def test_all_featurizer(self):
        for i, d in get_molecule_featurizer_info().iterrows():
            featurizer = d["instance"]
            f = featurizer(self.testmol)
            if hasattr(featurizer, "LENGTH"):
                self.assertEqual(featurizer.LENGTH, d["length"])
                self.assertEqual(len(featurizer), d["length"])
                self.assertEqual(f.shape[0], d["length"])
            else:
                # if isinstance(featurizer,FeaturizerList):
                #    print(len(featurizer))
                pass

            if hasattr(featurizer, "dtype"):
                self.assertEqual(
                    featurizer.get_dtype(),
                    d["dtype"],
                    f"Wrong dtype for featurizer {i} ({featurizer.get_dtype()}, {d['dtype']})",
                )
                if f.dtype.char == "U":
                    self.assertEqual(
                        featurizer.get_dtype(),
                        str,
                        f"Wrong dtype for featurizer {i} ({featurizer.get_dtype()}, {f.dtype})",
                    )
                else:
                    self.assertEqual(
                        featurizer.get_dtype(),
                        f.dtype,
                        f"Wrong dtype for featurizer {i} ({featurizer.get_dtype()}, {f.dtype})",
                    )
                # print(i)

    def test_prefeaturizer(self):
        raise NotImplementedError()

    def test_normalization(self):
        raise NotImplementedError()


class AtomFeaturizerTest(KImieTest):
    def setUp(self) -> None:
        laoder = PreparedMolDataLoader(ESOL())
        for m in laoder:
            if m is not None:
                if len(m.GetAtoms()) > 0:
                    self.testatom = m.GetAtoms()[0]
                    break
        assert self.testatom is not None, "No molecule found in ESOL"

    def test_featurizer_info(self):
        assert len(get_atom_featurizer_info()) > 0, "No atom featurizer found"
        assert len(get_atom_featurizer_info()) >= 100, get_atom_featurizer_info()

    def test_all_featurizer(self):
        for i, d in get_atom_featurizer_info().iterrows():
            print(i)
            featurizer = d["instance"]
            f = featurizer(self.testatom)
            if hasattr(featurizer, "LENGTH"):
                if featurizer.LENGTH >= 0:
                    self.assertEqual(featurizer.LENGTH, d["length"])
                    self.assertEqual(len(featurizer), d["length"])
                    self.assertEqual(f.shape[0], d["length"])
            else:
                # if isinstance(featurizer,FeaturizerList):
                #    print(len(featurizer))
                pass

            if hasattr(featurizer, "dtype"):
                self.assertEqual(
                    featurizer.get_dtype(),
                    d["dtype"],
                    f"Wrong dtype for featurizer {i} ({featurizer.get_dtype()}, {d['dtype']})",
                )
                if f.dtype.char == "U":
                    self.assertEqual(
                        featurizer.get_dtype(),
                        str,
                        f"Wrong dtype for featurizer {i} ({featurizer.get_dtype()}, {f.dtype})",
                    )
                else:
                    self.assertEqual(
                        featurizer.get_dtype(),
                        f.dtype,
                        f"Wrong dtype for featurizer {i} ({featurizer.get_dtype()}, {f.dtype})",
                    )
                # print(i)

    def test_prefeaturizer(self):
        raise NotImplementedError()

    def test_normalization(self):
        raise NotImplementedError()


    def test_split_and_merge(self):
        from  KImie.featurizer.utils import merge_atom_featurizer_data, split_atom_featurizer_data
        from KImie.featurizer.atom_featurizer import atom_ConnectedAtoms_featurizer
        from KImie.featurizer.prefeaturizer import Prefeaturizer
        feat = atom_ConnectedAtoms_featurizer
        ds=ESOL()

        laoder = PreparedMolDataLoader(ds)
        pref = Prefeaturizer(laoder,featurizer=feat)

        pref_data=[p for p  in pref]
        assert len(pref_data)==ds.expected_mol_count 

        ipd,split_indices = merge_atom_featurizer_data(pref)
        assert ipd.shape[1] == len(feat) 
        assert ds.expected_mol_count == split_indices.shape[0]+1

        for i,e in enumerate(np.split(ipd,split_indices)):
            np.testing.assert_array_equal(e,pref_data[i])

        for i,e in enumerate(split_atom_featurizer_data(ipd,split_indices)):
            np.testing.assert_array_equal(e,pref_data[i])

    def test_reduce_features(self):
        from  KImie.featurizer.utils import reduce_features
        a = np.zeros((10,4))
        a[0,0]=0.1
        a[0,2]=0.9  
        a[1,2]=0.2
        a[6,1]=0.3
        redf,nf_names = reduce_features(a,feature_names=["a","b","c","d"])
        assert redf.shape[1]==3, redf.shape
        assert redf.shape[0]==a.shape[0], f"{redf.shape[0]},{a.shape[0]}"
        assert nf_names == ["a","b","c"], nf_names

        redf,nf_names = reduce_features(a,min_rel_content=0.1,max_rel_content=0.9,feature_names=["a","b","c","d"])
        assert redf.shape[1]==1, redf.shape
        assert redf.shape[0]==a.shape[0], f"{redf.shape[0]},{a.shape[0]}"
        assert nf_names == ["c"], nf_names

        redf = reduce_features(a,min_rel_content=0.1,max_rel_content=0.9)
        assert redf.shape[1]==1, redf.shape
        assert redf.shape[0]==a.shape[0], f"{redf.shape[0]},{a.shape[0]}"