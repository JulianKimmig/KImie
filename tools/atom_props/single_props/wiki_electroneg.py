import requests
from bs4 import BeautifulSoup as bs
import os
import pandas as pd
import numpy as np

os.chdir(os.path.dirname(__file__))

res = requests.get(
    "https://en.wikipedia.org/wiki/Electronegativities_of_the_elements_(data_page)"
)
soup = bs(res.text, "html.parser")

# find by id
content = soup.find(id="bodyContent")

# find header containing text
o_header = content.findAll("h2")
n_header = len(o_header)
o_header_text = [o.text for o in o_header]

search_header = ["Pauling scale", "Allen scale"]
use_headers = {}
for i in range(n_header):
    for s in search_header:
        if s in o_header_text[i]:
            use_headers[s] = i
            break
print(use_headers)

html = content.prettify()

df = pd.DataFrame(columns=["Symbol"])
df.index.name = "atomic_number"

for k, i in use_headers.items():
    soup = bs(html, "html.parser")

    all_header = soup.findAll("h2")
    header = all_header[i]
    # delet every element before header
    preparent = header

    while preparent.parent:
        parent = preparent.parent

        for child in parent.findChildren(recursive=False):
            if child == preparent:
                break
            child.decompose()
        preparent = parent
        break

    if i < n_header - 1:
        header = all_header[i + 1]
        # delet every element after header
        preparent = header
        while preparent.parent:
            parent = preparent.parent
            go = False
            for child in parent.findChildren(recursive=False):
                if child == preparent:
                    go = True
                    continue
                if go:
                    child.decompose()
            preparent = parent

    # find tables with class
    tables = soup.findAll("table", attrs={"class": "wikitable sortable"})
    if len(tables) == 0:
        continue

    for table in tables:
        sdf = pd.read_html(str(table))[0]
        sdf.index = sdf["Number"]
        sdf.drop(columns=["Number"], inplace=True)
        sdf.index.name = "atomic_number"

        l_sdf = sdf.drop(columns=["Symbol"])
        df = pd.concat([df, l_sdf], axis=1)
        df.update(sdf)

#        print(sdf)
# print(soup.prettify())
# break
df.drop(
    columns=[c for c in df.columns if c not in ["use", "Electronegativity"]],
    inplace=True,
)
df = df[["use", "Electronegativity"]]
# print(df)
df.columns = ["EN_pauling_scale", "EN_allen_scale"]
df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
df.replace("no data", np.nan, inplace=True)
df.to_csv("wiki_electroneg.csv")
print(df)
# print(header)
