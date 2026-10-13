"""Chaque dossier du projet a un README.md qui cite tous ses fichiers (exigence de lisibilité du dépôt public)."""
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[1]
DOSSIERS = ["mclab", "tests", "scripts", "notebooks", "figures", "results", "docs"]


@pytest.mark.parametrize("dossier", DOSSIERS)
def test_readme_de_dossier_cite_tous_les_fichiers(dossier):
    d = RACINE / dossier
    readme = (d / "README.md").read_text(encoding="utf-8")
    fichiers = [f.name for f in d.iterdir() if f.is_file() and f.name != "README.md" and not f.name.startswith(".")
                and f.suffix != ".pyc"]
    absents = [f for f in fichiers if f not in readme]
    assert not absents, f"{dossier}/README.md ne cite pas : {absents}"
