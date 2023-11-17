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
