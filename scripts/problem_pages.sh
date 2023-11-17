#!/bin/sh

alexi convert --pages 10,11,12,13,14,15,16,17 ville.sainte-adele.qc.ca/upload/documents/Rgl-1314-2021-Z-en-vigueur-20230614.pdf | alexi crf - > zonage_sections.csv
alexi convert --pages 189,190,205,206,214,222,223,231 ville.sainte-adele.qc.ca/upload/documents/Rgl-1314-2021-Z-en-vigueur-20230614.pdf | alexi crf - > zonage_zones.csv
