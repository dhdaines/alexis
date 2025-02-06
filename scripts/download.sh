#!/bin/sh

set -e

alexi -v download --exclude=/derogation \
      --exclude='\d-[aA]dopt' \
      --exclude='-Z-\d' \
      --exclude='-\d-redevances' \
      --exclude='-[rR]eso' \
      --exclude=Plan-de-zonage \
      --exclude=AnnexeZonage
for d in download/*.pdf; do
    bn=$(basename $d .pdf)
    for dd in data/train data/dev data/test; do
        if [ -e "${dd}/${bn}.csv" ]; then
            cp "${dd}/${bn}.csv" download/
        fi
    done
done
alexi -v download -u https://vsadm.ca/citoyens/reglementation/reglementation-durbanisme/ -o download/vsadm --all-links
alexi -v download -u https://www.vss.ca/services-aux-citoyens/services/reglementation-durbanisme/ \
      -o download/vss --all-links -x '[Aa]nnexe'
alexi -v download -u https://www.ville.prevost.qc.ca/guichet-citoyen/services/urbanisme/ \
      -o download/prevost --all-links -x Annexe -x Formulaires -x PUMD -x PMAD
