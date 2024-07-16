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
  - Retrain from full dataset + patches
    - early stopping? sample a dev set?
  - Do extraction and qualitative evaluation
    - sort for batch processing then unsort afterwards
- CRF output layer DONE
- Ensemble RNN DONE
- Viterbi decoding (with allowed transitions only) on RNN outputs or
  ensemble RNNs
  - Could *possibly* train a CRF to do this, in fact
- Tokenize from chars
  - Add functionality to pdfplumber
- Use Transformers for embeddings
  - Heuristic pre-chunking as described below
  - Either tokenize from chars (above) or use first embedding per word
  - Probably project 768 dimensions down to something smaller
- Do prediction with Transformers (LayoutLM) DONE
  - heuristic chunking based on line gap (not indent) DONE
- Do prediction with Transformers (CamemBERT)
- Do prediction with Transformers (CamemBERT + vector feats)


Segmentation results
====================

- Things that helped
  - RNN helps overall, particularly on unseen data (using the
    "patches" as a test set)
  - use all the manually created features and embed them with >=4 dimensions
  - deltas and delta-deltas
  - scale all the things by page size (slightly less good than by
    abs(max(feats)) but probably more robust)
  - upweight B- tags by 2.0
  - weight all tags by inverse frequency (works even better than B- * 2.0)
  - taking the best model using f1_macro (can't do for full training
    unless we sample a dev set!)
  - ensemble of cross-validation folds (allows early stopping as well)
    - in *theory* dropout would give us this benefit too
- Inconclusive
  - GRU or plain RNN with lower learning rate
    - LSTM is maybe overparameterized?
    - Improves label accuracy quite a lot but mean F1 not really
    - This seems to be a consequence of lower learning rate not cell typpe
  - LayoutLM
    - pretrained on wrong language
    - layout features possibly suboptimal for this task
    - but need to synchronize evaluation metrics to be sure!
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
  - dropout on extra feedforward layer
  - wider word embeddings
  - CRF output layer
    - Training is *much* slower
    - Raw accuracy is consistently a bit better.  It is a better model.
    - Macro-F1 though is not as good (over B- tags)
      - Imbalanced data is an issue and weighting is more difficult
      - Definitely weight transitions and emissions (helps)
      - Have to weight "up", can't weight "down"
      - Weighting by exp(1.0 / count) better than nothing
      - Weighting by exp(1.0 / B-count) not helpful
      - Weighting by exp(1.0 / (B-count + I-count)) not helpful
    - Applying Viterbi to RNN output shows why
      - Sequence constraints favour accuracy of I over B
      - Weighted RNN training favours correct Bs, but Is can change
        mid-sequence, which we don't care about
      - Decoding with constraints forces B and I to agree, improving
        overall acccuracy by fixing incorrect Is but flipping some
        correct Bs in the process
      - Confirmed, Viterbi with --labels bonly gives (nearly) same
        results as non-Viterbi
- Things yet to be tried
  - pre-trained or pre-computed word embeddings
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
