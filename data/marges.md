Marges, en-têtes, pieds de page
-------------------------------

Malheureusement les marges des documents et règlements sont très
variables, alors il ne suffit pas de simplement rogner les pages pour
éliminer les numéros de page, en-têtes, et pieds.

Les règlements d'urbanisme (fichiers Rgl-1314-2021-*) ont
habituellement une en-tête de 45 points, par exemple:

    page,tag,text,x0,x1,top,doctop,bottom,upright,direction
    1,,Règlement,85.104,116.91828,36.48840000000007,1044.6084,44.52840000000003,True,1
    1,,sur,118.68708,127.83659999999999,36.48840000000007,1044.6084,44.52840000000003,True,1
    1,,les,129.6054,138.05544,36.48840000000007,1044.6084,44.52840000000003,True,1
    1,,ententes,139.79208,165.02160000000003,36.48840000000007,1044.6084,44.52840000000003,True,1
    1,,relatives,166.87884000000003,191.16768000000002,36.48840000000007,1044.6084,44.52840000000003,True,1
    1,,à,193.02492,196.69116,36.48840000000007,1044.6084,44.52840000000003,True,1
    1,,des,198.54036000000002,209.04060000000004,36.48840000000007,1044.6084,44.52840000000003,True,1
    1,,travaux,210.89784000000003,232.42895999999996,36.48840000000007,1044.6084,44.52840000000003,True,1
    1,,municipaux,234.28619999999995,267.44315999999986,36.48840000000007,1044.6084,44.52840000000003,True,1
    1,,numéro,269.1797999999999,291.43451999999985,36.48840000000007,1044.6084,44.52840000000003,True,1
    1,,1314-2021-TM,293.2837199999998,336.52135999999996,36.48840000000007,1044.6084,44.52840000000003,True,1
    1,,Table,389.23,405.74416,36.48840000000007,1044.6084,44.52840000000003,True,1
    1,,des,407.44864,418.06144,36.48840000000007,1044.6084,44.52840000000003,True,1
    1,,"matières,",419.95,446.97244,36.48840000000007,1044.6084,44.52840000000003,True,1
    1,,des,448.74124,459.35404,36.48840000000007,1044.6084,44.52840000000003,True,1
    1,,figures,461.21128,480.82084000000003,36.48840000000007,1044.6084,44.52840000000003,True,1
    1,,et,482.55748,488.10508000000004,36.48840000000007,1044.6084,44.52840000000003,True,1
    1,,des,489.87388,500.48668000000004,36.48840000000007,1044.6084,44.52840000000003,True,1
    1,,tableaux,502.34392,527.11516,36.48840000000007,1044.6084,44.52840000000003,True,1

Ils ont aussi un numéro de page seul en bas à droite:

    page,tag,text,x0,x1,top,doctop,bottom,upright,direction
    1,,2,523.54,527.20624,964.6884,1972.8084,972.7284,True,1

Par contre, les règlements adopté en séance de conseil ont un format
tout à fait différent.  Par exemple
`2022-04-19-Rgl-1324-redevances-adopte.csv` où il n'y a aucune en-tête
et le texte commence tout de suite à 31 points (environ 1cm) du haut
de page.  Par contre, ce document possède un pied de page avec le
numéro intégré:

    page,tag,text,x0,x1,top,doctop,bottom,upright,direction
    0,,Règlement,113.42,150.08132,938.98488,938.98488,947.0248799999999,True,1
    0,,1324,151.94,169.71408,938.98488,938.98488,947.0248799999999,True,1
    0,,–Contribution,171.41,219.2684,938.98488,938.98488,947.0248799999999,True,1
    0,,pour,221.01308,237.29408,938.98488,938.98488,947.0248799999999,True,1
    0,,financer,239.11112,267.2672,938.98488,938.98488,947.0248799999999,True,1
    0,,une,269.21,282.00968,938.98488,938.98488,947.0248799999999,True,1
    0,,dépense,283.7222,312.21596,938.98488,938.98488,947.0248799999999,True,1
    0,,pour,314.05712,330.33007999999995,938.98488,938.98488,947.0248799999999,True,1
    0,,"l’ajout,",332.14712,355.14955999999995,938.98488,938.98488,947.0248799999999,True,1
    0,,l’agrandissement,356.7414799999999,415.4736799999999,938.98488,938.98488,947.0248799999999,True,1
    0,,ou,417.1862,425.93372,938.98488,938.98488,947.0248799999999,True,1
    0,,la,427.60604,433.7084,938.98488,938.98488,947.0248799999999,True,1
    0,,modification,435.40484,478.73239999999987,938.98488,938.98488,947.0248799999999,True,1
    0,,d’infrastructures,113.42,171.20348,948.34488,948.34488,956.38488,True,1
    0,,ou,172.90796,181.65548,948.34488,948.34488,956.38488,True,1
    0,,équipements,183.3278,227.789,948.34488,948.34488,956.38488,True,1
    0,,municipaux,229.48543999999998,269.70955999999995,948.34488,948.34488,956.38488,True,1
    0,,–,271.49,275.51,948.34488,948.34488,956.38488,True,1
    0,,19,277.39,286.29832,948.34488,948.34488,956.38488,True,1
    0,,avril,287.94651999999996,303.67276,948.34488,948.34488,956.38488,True,1
    0,,2022,305.35,323.12404000000004,948.34488,948.34488,956.38488,True,1
    0,,(Adoption),324.79,362.66128,948.34488,948.34488,956.38488,True,1
    0,,page,498.48,514.64232,948.34488,948.34488,956.38488,True,1
    0,,1,516.48,520.93416,948.34488,948.34488,956.38488,True,1

Il faut alors soit des heurstiques, soit un modèle entraîné pour
étiquetter (et éliminer) ces blocs de texte.  L'autre option est tout
simplement d'intégrer leur détection dans un modèle général
d'extraction d'information.

Ils ont certains caractéristiques qui les rend assez facile de détecter:

- taille de police plus petite (9 points ou moins)
- dégagement assez importante avec la ligne précédente (24 points ou plus)
- situation au début ou à la fin de la page

On a alors l'option soit de les étiquetter pour un modèle en utilisant
des features pertinents (hauteur de ligne, distance avec précédent,
distance du bord de page), soit les détecter avec un script.

Dans ce cas (comme d'autres) il vaudrait la peine d'essayer la façon
programmatique d'abord, ce qui nous guidera pour la suite.

Notez que dans certains cas les en-têtes et pieds sont marqués dans le
PDF par des "marked content sections" avec le tag "Artifact".
