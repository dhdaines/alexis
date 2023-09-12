Guide d'annotation
==================

En théorie (où il fait bon vivre) la tâche se résume à une analyse
(parsing) du texte, puisque celui-ci se décline en une hiérarchie de
chapitres, sections, articles, alinéas, etc.

En pratique (où on vit vraiment) on ne veut pas passer les mots et
leurs traits caractéristiques directement dans un analyseur puisque la
complexité computationnelle en est trop importante.

Nous allons alors décomposer le problème pour en extraire la
ségmentation syntaxique (ou visuelle) du texte, c'est à dire les
alinéas, titres, tableaux, et autre éléments textuelles, puis procéder
à une classification des segments extraits, puis enfin une analyse
structurelle sur la base de ces segments.

Une phase à part est imaginé pour l'extraction d'entités tels que les
dates d'adoption, le numéro et le titre du règlement.

Cette approche est implémenté d'abord à base de règles, ou bien des
heuristiques, dans les modules `segment.py` et `label.py`, où l'on
bâtit progressivement des annotations de type BIO pour une panoplie de
catégories à la fois syntaxiques et sémantiques.  L'extraction des
entités est fait de façon *très* heuristique (*broche à foin* en bon
québécois) en-dedans de ces modules

Pour transférer cette démarche dans un cadre conceptuel
d'apprentissage automatique, il faut réimaginer les annotations pour
faciliter cet apprentissage.  Surtout, il faut éviter de mélanger des
catégories syntaxiques et sémantiques, et de trop diviser l'ensemble
des annotations puisqu'il faut un certain "support" pour chaque
catégorie.

`Titre`: cette catégorie réprésente les titres de documents,
chapitres, sections, ou autres étendues de texte.  Elle *n'est pas*
nécessairement le titre du document au complet.  Les catégories
`Chapitre`, `Annexe`, `Section`, `Article`, etc. sont retenues mais
pourraient être transformées en `Titre` pour fins de modélisation.

`Alinea`: cette catégorie représente un bloc de texte qui n'a pas une
présentation tabulaire.

`Liste`: cette catégorie représente un item d'une liste ou
énumération.  Elle est traîtée séparément des alinéas puisque ces
items ont généralement une forme visuelle particulière.

`Tableau`: cette catégorie représente un bloc de texte avec
présentation tabulaire.

`Tete`: en-tête de page (pourrait être transformé en `O`)

`Pied`: pied de page (pourrait être transformé en `O`)

`TOC`: tableaux de matières, tableaux et figures


Entrainement d'un modèle CRF pour segmentation
==============================================

- Modèle de base, utilisant tout simplement les mêmes traits qu'on
  utilisait auparavant pour les heuristiques.
- Quantisation des traits numeriques.
- Normalisation des traits textuels.
- Recherche d'hyperparametres
- 
