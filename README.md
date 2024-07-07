ALexi, EXtracteur d'Information adèlois
=======================================

ALEXI est une bibliothèque de code Python qui extrait la structure et
le contenu des documents officiels de la ville de Sainte-Adèle afin de
faciliter leur indexation par un moteur de recherche,
[SÈRAFIM](https://github.com/dhdaines/serafim) par exemple.

Extraction des documents
------------------------

L'extraction de la structure et du contenu des documents se fait par
cette série d'étapes:

1. Mettre à jour le répertoire local des documents.  Ceci se fait avec
   [wget](https://www.gnu.org/software/wget/) par la commande `alexi
   download`.  Par défaut, tous les documents dans la section des
   règlements d'urbanisme sont sélectionnés.  Il est conseillé d'utiliser
   l'option `--exclude` pour exclure certains documents moins utiles:

        alexi -v download --exclude=Plan --exclude=/derogation \
              --exclude='\d-[aA]dopt' --exclude='Z-\d'

   Les documents téléchargés seront déposés dans le repertoire
   `download` par défaut.  Pour les diriger ailleurs vous pouvez
   fournir l'option `--outdir` avec le nom du repertoire désiré.
   
   Il est également possible de sélectionner un autre filtre pour les
   documents en fournissant une expression régulière comme argument à
   `alexi download`, par exemple, pour avoir tous les règlements:
   
        alexi -v download Reglements
   
2. Extraire et analyser le contenu des documents, ce qui se fait par
   la commande `alexi extract`:
   
        alexi -v extract download/*.pdf
   
   Les fichiers générés seront déposés dans le repertoire `export` par
   défaut.  Encore il est possible de fournir l'option `--outdir` pour
   modifier cela.

Correction des structures extraites
-----------------------------------

Il arrive parfois (ou souvent) qu'ALEXI n'interprète pas correctement
le formatage d'un PDF.  Pour le moment la manière de corriger consiste
à sortir les traits distinctifs utilisés par le classificateur en
format CSV, corriger un exemplaire du texte non pris en charge (ne
serait-ce qu'une seule page), et l'ajouter à l'entrainement du modèle.
La commande `alexi annotate` vise à faciliter ce processus.

Par exemple, si l'on veut corriger l'extraction de la page 1 du
règlement 1314-2023-DEM, on peut d'abord extraire les données et une
visualisation de la segmentation et classification avec cette commande
(on peut spécifier le nom de base de fichiers comme deuxième argument):

    alexi annotate --pages 1 \
        download/2023-03-20-Rgl-1314-2023-DEM-Adoption-_1.pdf \
        1314-page1

Cela créera les fichers `1314-page1.pdf` et `1314-page1.csv`. Notez
qu'il est possible de spécifier plusieurs pages à extraire et
annoter, par exemple:

    --pages 1,2,3

Dans le PDF, pour le moment, des rectangles colorés sont utiliser pour
représenter les blocs annotés et aider à répérer les erreurs.
Notamment:

- Les chapitres et annexes sont en rouge
- Les sections et articles sont en rose (plus foncé plus le type
  d'élément est large)
- Les listes sont en bleu-vert (parce qu'elles sont souvent confondues
  avec les articles)
- Les en-têtes et pieds de page sont en jaune-vert-couleur-de-bile
- Tout le reste est en noir (alinéas, tableaux, figures)

Pour les éléments de séquence (il y a juste les titres et les numéros)
ceux-ci sont indiqués par un remplissage vert clair transparent.

Avec un logiciel de feuilles de calcul dont LibreOffice ou Excel, on
peut alors modifier `1314-page1.csv` pour corriger la segmentation.
Il est *très important* de spécifier ces paramètres lorsqu'on ouvre et
sauvegarde le fichier CSV:

- La colonne "text" doit avoir le type "Texte" (et pas "Standard")
- Le seul séparateur de colonne devrait être la virgule (pas de
  point-virgule, tab, etc)

Une fois les erreurs corrigés, le résultat peut être vu avec:

    alexi annotate --pages 1 \
        --csv 1314-page1.csv \
        download/2023-03-20-Rgl-1314-2023-DEM-Adoption-_1.pdf
        1314-page1

Cela mettra à jour le fichier `1314-page1.pdf` avec les nouvelles
annotations.

Une fois satisfait du résultat, il suffira de copier `1314-page1.csv`
vers le repertoire `data` et réentrainer le modèle avec
`hatch run train`.

Extraction de catégories pertinentes du zonage
----------------------------------------------

Quelques éléments du règlement de zonage ont droit à un traitement
particulier d'ALEXI pour faciliter la génération d'hyperliens internes
ainsi que fournir des informations à des applications externes dont
ZONALDA et SÈRAFIM.  Ces informations se retrouvent dans le fichier
`zonage.json` dans le repertoire `export`.  Actuellement pris en
charge sont:

- Les catégories de milieux (ex. "T2 OCCUPATION DE LA FORÊT")
- Les types de milieux (ex. "T2.1 AGROFORESTIER")
- Les catégories d'usages (ex. "H HABITATION")
- Les classes d'usages (ex. "H-01 Habitation unifamiliale")

À l'avenir il pourrait être utile d'extraire aussi les exceptions
spécifiques a des zones individuelle pour les référencer.

Extraction d'hyperliens
-----------------------

Un ensemble limité d'hyperliens internes et externes est pris en
charge par ALEXI, spécifiquement:

- Des liens vers d'autres articles du même règlement
- Des liens vers d'autres chapitres ou annexes du même règlement
- Des liens vers les catégories et types de milieu et usages
- Des liens vers des articles de la loi sur l'aménagement et
  l'urbanisme, par exemple
  https://www.legisquebec.gouv.qc.ca/fr/document/lc/A-19.1#se:148_0_1



Génération d'un index
---------------------

Il est maintenant possible de générer un index pour faire des
recherches dans les documents, ce qui se fait avec `alexi index`:

    alexi index export

L'index sera généré dans le repertoire `indexdir`.  Maintenant vous
pouvez lancer des recherches!  Par exemple:

    alexi search poulailler

Ce qui devrait donner une sortie comme:

    https://ville.sainte-adele.qc.ca/upload/documents/Rgl-1314-2021-Z-en-vigueur-20231013.pdf#page=77 Article 99: Poulailler et parquet
    https://ville.sainte-adele.qc.ca/upload/documents/Rgl-1314-2021-Z-en-vigueur-20231013.pdf#page=73 SousSection 15: USAGES COMPLÉMENTAIRES À UN USAGE DU GROUPE « HABITATION (H) »
    https://ville.sainte-adele.qc.ca/upload/documents/Rgl-1314-2021-Z-en-vigueur-20231013.pdf#page=73 Section 5: USAGES COMPLÉMENTAIRES
    https://ville.sainte-adele.qc.ca/upload/documents/Rgl-1314-2021-Z-en-vigueur-20231013.pdf#page=18 Chapitre 3: DISPOSITIONS GÉNÉRALES AUX USAGES
    https://ville.sainte-adele.qc.ca/upload/documents/Rgl-1314-2021-Z-en-vigueur-20231013.pdf#page=1 Document : Règlement de zonage numéro 1314-2021-Z

Les liens ci-haut vous amèneront dans les documents PDF, mais dans les
fichiers exportés il est également possible de naviguer la structure
des règlements, par exemple l'article 99 du règlement 1314-2021-Z se
trouvera dans le repertoire
`export/Rgl-1314-2021-Z-en-vigueur-20231013/Article/99/` en formats
HTML et MarkDown, avec des metadonnées en JSON.
