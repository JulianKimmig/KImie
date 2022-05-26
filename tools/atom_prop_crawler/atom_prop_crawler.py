try:
    import KImie
except ModuleNotFoundError:  #
    import os, sys

    _cd = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    while os.path.dirname(_cd) != _cd:
        if "KImie" in os.listdir(_cd):
            if os.path.basename(_cd) == "KImie":
                pass
            else:
                _cd = os.path.join(_cd, "KImie")
            sys.path.append(_cd)
            break
        _cd = os.path.dirname(_cd)
    print(_cd)
    import KImie


import requests
from bs4 import BeautifulSoup as bs
from KImie.utils.mol import ATOMIC_SYMBOL_NUMBERS
import os
import pandas as pd


def main():
    os.chdir(os.path.dirname(__file__))
    print(ATOMIC_SYMBOL_NUMBERS)

    base_url = "https://www.webelements.com/periodicity/atomic_number/"
    res = requests.get(base_url)
    soup = bs(res.text, "html.parser")
    # finde table by class
    table = soup.find("table", attrs={"class": "periodic-table"})

    df = pd.DataFrame(columns=["symbol", "link"])
    df.index.name = "atomic_number"

    for symbol, num in ATOMIC_SYMBOL_NUMBERS.items():

        # find link by text
        link = table.find("a", text=symbol)
        if link is None:
            continue
        # get href
        href = link.get("href")
        if num not in df.index:
            df.loc[num] = [symbol, base_url + href]

    for r, d in df.iterrows():
        sub_dfs = {}
        res = requests.get(d["link"])
        print(d["link"])
        soup = bs(res.text, "html.parser")
        # find all ul_facts_table classes
        ul_facts_table = soup.find_all("ul", attrs={"class": "ul_facts_table"})
        # iterate ofer ul entries
        for ul in ul_facts_table:
            # find all li entries
            for li in ul.find_all("li"):
                # get text
                text = li.get_text()
                # split by :
                text = text.split(":")
                # get key
                key = text[0].strip()
                # get value
                value = text[1].strip()
                # add to df
                # add column if needed
                if key not in df.columns:
                    df[key] = None
                # add value
                df.loc[r, key] = value
        sub_dfs = {}
        # get sub properties:
        for subsite in [
            "physics.html",
            "thermochemistry.html",
            "atom_sizes.html",
            "electronegativity.html",
            "atoms.html",
            "geology.html",
        ]:
            sub_dfs[subsite] = []
            res = requests.get(d["link"] + subsite)
            ssoup = bs(res.text, "html.parser")
            # find all spark_table_list classes
            spark_table_list = ssoup.find_all("ul", attrs={"class": "spark_table_list"})
            # iterate ofer ul entries
            for ul in spark_table_list:
                # find all li entries
                for li in ul.find_all("li"):
                    # get text
                    text = li.get_text()
                    # split by :
                    text = text.split(":")
                    # get key
                    key = text[0].strip()
                    # get value
                    value = text[1].strip()
                    # add to df
                    # add column if needed
                    if key not in df.columns:
                        df[key] = None
                    # add value
                    df.loc[r, key] = value

            table_list = ssoup.find_all("table", attrs={"class": "pure-table"})

            # iterate over table
            for i, table in enumerate(table_list):
                sdf = pd.read_html(str(table))
                sub_dfs[subsite].extend(sdf)
                continue
                # get table header
                header = table.find("thead")
                ignore_indices = []
                if header is not None:
                    header_texts = [x.get_text().strip() for x in header.find_all("th")]
                    for i, text in enumerate(header_texts):
                        if text == "Periodicity link":
                            ignore_indices.append(i)
                print(header_texts)
                # find all tr entries
                for tr in table.find_all("tr"):
                    # get all td entries
                    tds = tr.find_all("td")
                    if len(tds) < 2:
                        continue
                    if header is None:
                        header_texts = [""] * len(tds)

                    base_key = (
                        header_texts[0] + " " + tds[0].get_text().strip()
                    ).strip()
                    print(base_key)
                    for i, td in enumerate(tds):
                        if i == 0:
                            continue
                        if i in ignore_indices:
                            continue

                        value = td.get_text().strip()
                        key = (base_key + " " + header_texts[i]).strip()
                        print("\t", key)
                        if key not in df.columns:
                            df[key] = None
                        # add value
                        df.loc[r, key] = value

        for subsite, sdfs in sub_dfs.items():
            for i, sdf in enumerate(sdfs):
                sdf.to_csv(f"periodic_table_{r}_{subsite}_{i}.csv", index=False)

        df.to_csv("periodic_table.csv")


if __name__ == "__main__":
    main()
