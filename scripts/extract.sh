#!/bin/sh

set -e

alexi -v extract -m download/index.json -o export/vdsa download/*.pdf
alexi -v extract -m download/vsadm/index.json -o export/vsadm download/vsadm/*.pdf
alexi -v extract -m download/vss/index.json -o export/vss download/vss/*.pdf
alexi -v extract -m download/prevost/index.json -o export/prevost download/prevost/*.pdf
echo '<meta http-equiv="refresh" content="0; url=vdsa/index.html">' > export/index.html
