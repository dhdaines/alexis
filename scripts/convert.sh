#!/bin/sh

for i in $(cat data/urbanisme.txt); do
    echo $i
    bn=$(basename $i .pdf)
    alexi extract --images "../serafim/public/img/$bn" \
          ville.sainte-adele.qc.ca/$i \
          > "../serafim/data/$bn.json"
done
