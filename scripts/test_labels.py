"""
Use rule-based features and evaluate with CRF evaluator
"""

from pathlib import Path

import alexi.crf
from alexi.label import Classificateur
from alexi.segment import Segmenteur
from sklearn_crfsuite import metrics

test_set = list(alexi.crf.load(Path("data/test").glob("*.csv")))
truth = alexi.crf.page2labels(test_set)
pred = alexi.crf.page2labels(Classificateur()(Segmenteur()(test_set)))

sorted_labels = "O B-Alinea B-Enumeration B-Pied B-TOC B-Tableau B-Tete B-Titre".split()
print(metrics.flat_classification_report([truth], [pred], labels=sorted_labels))
