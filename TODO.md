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
  - Retokenize CSVs using CamemBERT tokenizer (spread features on pieces) DONE
    - doesn't work well for CRFs, possibly due to:
      - all subwords have the same position, so layout features are wrong
      - hand-crafted features maybe don't work the same on subwords (leading _ thing)
  - Train a BiLSTM model with vsl features DONE
    - Learning rate decay and early stopping DONE
    - Embed words and categorical features DONE
    - Use same evaluator as CRF training for comparison DONE
    - Scale layout features by page size and include as vector DONE
  - Things that helped
    - use all the manually created features and embed them with >=4 dimensions
    - deltas and delta-deltas
    - scale all the things by page size (slightly less good than by
      abs(max(feats)) but probably more robust)
    - upweight B- tags by 2.0
    - smaller word embeddings (not enough data to train them well, and
      they are not reliable)
    - taking the best model using f1_macro
  - Things that did not help
    - weighting classes by inverse frequency (just upweight B as it's what we care about)
    - more LSTM layers
    - much wider LSTM
    - much narrower LSTM
    - dropout on LSTM layers
  - Things yet to be tried
    - CRF output layer (should help a lot)
    - label smoothing
    - dropout in other places
    
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
