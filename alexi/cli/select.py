"""
Fonction de sélection de fichiers source d'ALEXI.
"""

import re
from pathlib import Path

from bs4 import BeautifulSoup


def main(args):
    """Trouver une liste de fichiers dans la page web des documents."""
    with open(args.infile) as infh:
        soup = BeautifulSoup(infh, 'lxml')
        for h2 in soup.find_all('h2', string=re.compile(args.section, re.I)):
            ul = h2.find_next('ul')
            for li in ul.find_all('li'):
                path = Path(li.a["href"])
                print(path.relative_to("/"))

