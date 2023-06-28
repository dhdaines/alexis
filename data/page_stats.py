#!/usr/bin/env python3

import itertools
import pandas as pd
from pathlib import Path


max_page = 0
for p in Path(".").iterdir():
    if p.suffix != ".csv":
        continue
    print(p)
    df = pd.read_csv(p)
    for idx, group in itertools.groupby(df["page"]):
        nwords = sum(1 for x in group)
        print(idx, nwords)
        max_page = max(nwords, max_page)
print("Max words per page:", max_page)

