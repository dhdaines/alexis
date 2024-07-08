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

Segmentation
============

- Retokenize CSVs using CamemBERT tokenizer (spread features on pieces) DONE
- Train a BiLSTM model with vsl features DONE
  - Learning rate decay and early stopping DONE
  - Embed words and categorical features DONE
  - Use same evaluator as CRF training for comparison DONE
  - Scale layout features by page size and include as vector DONE
- CRF output layer DONE
- Tokenize from chars
- Use Transformers for embeddings
  - Heuristic pre-chunking as described below
  - Either tokenize from chars (above) or use first embedding per word
  - Probably project 768 dimensions down to something smaller
- Do prediction with Transformers
  - not expecting this to work very well due to sequence length limits
  - what happens if we split a page in the middle of a paragraph?
  - or in the middle of a word?
  - could do a heuristic based pre-chunking (e.g. on a "large
    enough" vertical whitespace)

Segmentation results
====================

- Things that helped
  - use all the manually created features and embed them with >=4 dimensions
  - deltas and delta-deltas
  - scale all the things by page size (slightly less good than by
    abs(max(feats)) but probably more robust)
  - upweight B- tags by 2.0
  - taking the best model using f1_macro
- Inconclusive
  - GRU or plain RNN with lower learning rate
    - LSTM is maybe overparameterized?
    - Improves label accuracy quite a lot but mean F1 not really
    - This seems to be a consequence of lower learning rate not cell typpe
  - wider word embeddings (maybe or maybe not, doing grid search...)
- Things that did not help
  - CamemBERT tokenizer doesn't work well for CRFs, possibly due to:
    - all subwords have the same position, so layout features are wrong
    - hand-crafted features maybe don't work the same on subwords (leading _ thing)
  - weighting classes by inverse frequency (just upweight B as it's what we care about)
  - more LSTM layers
  - much wider LSTM
  - much narrower LSTM
  - dropout on LSTM layers
  - extra feedforward layer
  - CRF output layer
    - training becomes very unstable and macro f1 is really bad
    - probably due to:
      - very imbalanced classes (lots of I, very little O or B)
        - can possibly be solved using AllenNLP implementation
      - very long sequences (another aspect of the same problem)
        - path score for a long sequence will converge to 0 if
          transition weights are too small
        - can possibly be solved by initializing the transition
          weights with something non-random (I think the AllenNLP
          implementation also does this)
- Things yet to be tried
  - better CRF implementation (AllenNLP modules lite)
  - pre-trained or pre-computed word embeddings
  - hyperparameter t00ning
  - label smoothing
  - feedforward layer before RNN
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
- investigate using a different CRF library
- tune regularization (some more)
- compare memory footprint of main branch versus html_output
- levels of lists
  - List+, List-, List
  - 1314-2021-PC could be used to train this, then try to relabel the others accordingly
