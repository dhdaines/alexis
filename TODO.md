Tracking some time
------------------

- fix sous-section links 1h
  - test case 15 min
  - fuzzy matching of element numbers w/sequence 30 min
  - deploy 15 min

- links to categories 1h
  - test case 15 min
  - collect cases and implement 30 min
  
- links to zones 2h
  - implement zone query in ZONALDA (with centroid) 1h30
  - extract zone links (multiple usually) 30min

- links to usages 1h30
  - test case 15 min
  - analysis function 45 min
  - linking as above 30 min

Immediate fixes/enhancements
----------------------------

- redo serafim CI
  - download alexi files with https://github.com/dawidd6/action-download-artifact
- better features / training
  - numeric sequence features
  - gazetteers possibly
  - optimize feature extraction
- better sequence tagging
  - use main train_crf.py
  - identify pages for training/testing (for now, those with at least one sequence tag)
- support more ways of addressing stuff
  - Titre I, II, III in that one stupid bylaw (WTF)
  - Paragraph numbers (actually items in lists)
  
DERP LERNING
------------

- Segmentation
  - Retokenize CSVs using CamemBERT tokenizer (spread features on pieces)
  - Train PyTorch-CRF: https://pytorch-crf.readthedocs.io/en/stable/
  - possibly use Skorch to do evaluation: https://skorch.readthedocs.io/en/stable/

Documentation
-------------

- Blog part 3: Overview of heuristic-based system
- Blog part 4: Architecture of CRF-based system

Unprioritized future stuff
--------------------------

- span-level tagging:
  - more options for training/testing specific pages
  - tags:
    - document title DONE
    - chapter/section/article numbers
    - dates (avis, adoption, etc)
    - article references
    - zones et types de milieu 1314-2021-Z
    - usages 1314-2021-Z
    - cross-references to items in list (and in preface to list) in 1314-2021-PC for instance
- annotations:
  - eliminate Tableau everywhere there is no Table (magical thinking)
  - All "Figure N", "Tableau N" tagged as Titre (if outside Table)
- investigate possibility of removing impossible transitions in crfsuite
  - have to hack the crfsuite file to do this
  - not at all easy to do with sklearn-crfsuite magical pickling
  - otherwise ... treat I-B as B-B when following O or I-A (as before)
- workflow for correcting individual pages
  - convenience functions for "visual debugging" in pdfplumber style
  - instructions to identify and extract CSV for page
- tune regularization (some more)
- compare memory footprint of main branch versus html_output
- levels of lists
  - List+, List-, List
  - 1314-2021-PC could be used to train this, then try to relabel the others accordingly
