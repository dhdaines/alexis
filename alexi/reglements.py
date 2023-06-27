#!/usr/bin/env python3

"""
Extraire la liste de PDFS pour une catégorie de règlements.
"""

from bs4 import BeautifulSoup
from pathlib import Path
import argparse
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("publications", help="Page HTML avec liste de publications", type=Path)
args = parser.parse_args()

with open(args.publications) as infh:
    soup = BeautifulSoup(infh, 'lxml')
    for h2 in soup.find_all('h2', string=re.compile(r"règlements", re.I)):
        ul = h2.find_next('ul')
        for li in ul.find_all('li'):
            path = Path(li.a["href"])
            print(path.relative_to("/"))
