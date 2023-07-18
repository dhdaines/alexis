#!/bin/sh

for i in $(cat data/urbanisme.txt); do
    echo $i
    csv=data/train/$(basename $i .pdf).csv
    if [ -e "$csv" ]; then
	echo $csv
	alexi json -n $(basename $i) $csv > ../serafim/data/$(basename $i .pdf).json
    else
	alexi extract ville.sainte-adele.qc.ca/$i > ../serafim/data/$(basename $i .pdf).json
    fi
done
