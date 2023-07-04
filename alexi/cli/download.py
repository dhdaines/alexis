"""
Fonction de téléchargement d'ALEXI.
"""

import subprocess


def main(args):
    """Télécharger les fichiers avec wget"""
    subprocess.run(["wget",
                    "--no-check-certificate",
                    "--timestamping",
                    "--recursive",
                    "--level=1",
                    "--accept-regex",
                    r".*upload/documents/.*\.pdf",
                    "https://ville.sainte-adele.qc.ca/publications.php"],
                   check=True)

