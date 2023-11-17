#!/bin/sh

rm -rf ../serafim/public/img ../serafim/data
mkdir -p ../serafim/data
for i in $(cat data/urbanisme.txt); do
    echo $i
    bn=$(basename $i .pdf)
    python scripts/convert.py --serafim \
           --outdir ../serafim \
          "ville.sainte-adele.qc.ca/$i"
done
