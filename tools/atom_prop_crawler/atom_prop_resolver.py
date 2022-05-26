import os
import pandas as pd
import numpy as np


def main():
    os.chdir(os.path.dirname(__file__))
    subdfs = [f for f in os.listdir() if f.startswith("periodic_table_")]

    entries = []
    for fn in subdfs:
        f = fn.replace("periodic_table_", "")
        an = int(f.split("_")[0])
        subsite = f.split("_", 1)[1].split(".html")[0]
        index = int(f.rsplit("_", 1)[1].replace(".csv", ""))
        entries.append([fn, an, subsite, index])
    df = pd.DataFrame(entries, columns=["fn", "an", "subsite", "index"])

    subsite_lengths = {ss: [] for ss in df.subsite.unique()}
    for (ss, an), sg in df.groupby(["subsite", "an"]):
        subsite_lengths[ss].append(len(sg))

    for ss, l in subsite_lengths.items():
        l = np.array(l)
        assert np.all(l == l[0]), "not all legths_equal"

    df.sort_values(["an", "subsite", "index"], inplace=True)

    df["csv"] = None

    for (ss, idx), sg in df.groupby(["subsite", "index"]):
        print(ss, idx)
        sg["csv"] = sg["fn"].apply(lambda f: pd.read_csv(f, index_col=0))

        for i, d in sg.iterrows():
            c = d["csv"]
            if "Periodicity link" in c.columns:
                c.drop(columns=["Periodicity link"], inplace=True)
            c.columns = [(d["an"], col) for col in c.columns]

        indices = [c.index.values for c in sg["csv"]]
        indices_equal = np.all(np.array(indices) == indices[0])
        if indices_equal:
            c = pd.concat(sg["csv"].values, axis=1)
            c.replace("(no data)", np.nan, inplace=True)
            print(c)
        else:
            continue
            for c in sg["csv"].values:
                print(c)
            raise NotImplementedError()


if __name__ == "__main__":
    main()
