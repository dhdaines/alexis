Liste de tâches
===============

- Tests unitaires (extraire des pages individuelles des PDFS en data/train)
  - exemple de TOC
  - exemple de tableau
  - exemples de chapitres
  - exemples de sections
  - exemples de sous-sections
- Bugfix:
  - Extra "chapters" everywhere (00, 1314-C, 1314-2023-DEM)
  - Missing chapters (1314-2021-L, 1314-2021-PC, 1314-2021-TM, 1314-2021-Z, etc)
  - Titre complet PAE
  - Leading dashes in chapters RGL-1132
  - Adoption vs. objet RGL-1132
  - Missing content in 1314-2021-PU
  - Weird column titles in PIIA
- Documents à ajouter:
  - SQ
  - Animal
- Évaluation de la segmentation, CRF vs. heuristique
  - Loading CSV features into CRFsuite
  - Réannotation des CSV avec Tag, MCID
  - Billet de blogue pour rationaliser la façon de faire
