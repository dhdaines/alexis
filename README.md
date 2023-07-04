ALexi, EXtracteur d'Information adèlois
---------------------------------------------------------

ALEXI est une bibliothèque de code Python qui extrait la structure et
le contenu des documents officiels de la ville de Sainte-Adèle afin de
faciliter leur indexation par un moteur de recherche, SÈRAFIM par
exemple.

Actuellement, sa mode de fonctionnement est de consommer des fichiers
PDF et de sortir des fichiers JSON.  Il est envisageable qu'à l'avenir
il pourra prendre en charge aussi des documents Word, HTML, ou autres,
et que sa sortie pourra être dirigé vers une base de données
relationnelle.

Extraction des documents
========================

L'extraction de la structure et du contenu des documents se fait par
une série d'étapes:

1. Mettre à jour le répertoire local des documents.  Ceci se fait avec
[wget](https://www.gnu.org/software/wget/) par la commande `alexi
download`.  Il téléchargera les documents plus récents à partir du
site web de la Ville, ainsi que la page principale avec l'index des
documents, ce qui se trouvera dans
[ville.sainte-adele.qc.ca/publications.php](./ville.sainte-adele.qc.ca/publications.php).
2. Générer la liste de documents d'intérêt.
