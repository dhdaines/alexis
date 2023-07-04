ALexi, EXtracteur d'Information adèlois
=======================================

ALEXI est une bibliothèque de code Python qui extrait la structure et
le contenu des documents officiels de la ville de Sainte-Adèle afin de
faciliter leur indexation par un moteur de recherche, [SÈRAFIM](https://github.com/dhdaines/serafim) par
exemple.

Actuellement, sa mode de fonctionnement est de consommer des fichiers
PDF et de sortir des fichiers JSON.  Il est envisageable qu'à l'avenir
il pourra prendre en charge aussi des documents Word, HTML, ou autres,
et que sa sortie pourra être dirigé vers une base de données
relationnelle.

Extraction des documents
------------------------

L'extraction de la structure et du contenu des documents se fait par
cette série d'étapes:

1. Mettre à jour le répertoire local des documents.  Ceci se fait avec
[wget](https://www.gnu.org/software/wget/) par la commande `alexi
download`.  Il téléchargera les documents plus récents à partir du
site web de la Ville, ainsi que la page principale avec l'index des
documents.
2. Générer la liste de documents d'intérêt.  Ceci se fait avec `alexi
select` qui écrit une liste sur la sortie standarde.  Par défaut tous
les règlements y sont compris, si vous voulez par exemple seulement
les règlements d'urbanisme, l'option `-s` peut être utilisé, par
exemple:

       alexi select -s urbanisme

3. Convertir les documents du format PDF en format CSV, avec `alexi convert`.
4. Extraire les blocs de texte brut avec `alexi segment`.
5. Extraire la structure du texte (sections, articles, alinéas) avec `alexi extract`.

À partir de cet étape les fichiers JSON peuvent être utilisés par un
moteur de recherche, dont
[SÈRAFIM](https://github.com/dhdaines/serafim).  Il est également
possible de générer un index avec ALEXI et faire des recherches sur la
ligne de commande, un API REST sera aussi disponible sous peu.
