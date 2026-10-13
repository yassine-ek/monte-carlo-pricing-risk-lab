"""Hygiène du dépôt public : aucune trace interne ni chemin de machine, liens des docs résolus, attributs Git, données absentes gérées.

Le parcours des fichiers ignore les environnements virtuels et les caches : créer un `.venv` dans le dépôt ne fait pas échouer ce test.
"""
import os
import re
import sys
from pathlib import Path

import matplotlib
import pytest

from mclab import risque as rk

RACINE = Path(__file__).resolve().parents[1]
IGNORES = {".venv", "venv", "env", ".git", "build", "dist", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".mypy_cache",
           ".ruff_cache", "node_modules", "site-packages", "cache"}
EXTENSIONS = {".py", ".md", ".json", ".yml", ".yaml", ".txt", ".toml", ".ini", ".cfg", ".ipynb"}
# motifs assemblés par morceaux : ce fichier ne doit pas contenir lui-même les traces qu'il cherche
INTERDITS = [
    re.escape("C:" + chr(92) + "Users"), re.escape("C:/" + "Users"),
    "smart" + "Trade", "X" + "rays", "Liquidity" + "FX", "Algo" + "box", "Attijari" + "wafa", r"\bAW" + r"B\b", "stage " + "awb",
    "projets" + "_cv", "_rev" + "ue/", "_recher" + "che/", "note de recherche " + "interne", "PIV" + "OT", "Route " + "A",
    "orchestr" + "ateur", "agent ind" + "épendant", "SPEC" + "_GITHUB",
]


def fichiers_du_depot():
    for dossier, sous_dossiers, noms in os.walk(RACINE):
        sous_dossiers[:] = [d for d in sous_dossiers if d not in IGNORES]
        for nom in noms:
            chemin = Path(dossier) / nom
            if chemin.suffix in EXTENSIONS:
                yield chemin


def test_aucune_trace_interne_ni_chemin_de_machine():
    trouves = []
    for f in fichiers_du_depot():
        texte = f.read_text(encoding="utf-8", errors="ignore")
        trouves += [f"{f.relative_to(RACINE)} : {m}" for m in INTERDITS if re.search(m, texte)]
    assert not trouves, trouves


@pytest.mark.parametrize("md", [RACINE / "README.md", *sorted((RACINE / "docs").glob("*.md"))], ids=lambda p: p.name)
def test_liens_relatifs_resolus(md):
    texte = md.read_text(encoding="utf-8")
    cibles = re.findall(r"\]\(([^)\s]+)\)", texte) + re.findall(r'src="([^"]+)"', texte)
    cassees = [c for c in cibles if not c.startswith(("http", "#", "mailto:")) and not (md.parent / c.split("#")[0]).exists()]
    assert not cassees, cassees


def test_gitattributes_declare_les_notebooks_comme_documentation():
    attributs = (RACINE / ".gitattributes").read_text(encoding="utf-8").splitlines()
    for regle in ("*.ipynb linguist-documentation", "*.md text eol=lf", "*.json text eol=lf"):
        assert regle in attributs


def test_donnees_absentes_message_guide(monkeypatch, tmp_path):
    """Sans données de marché : une erreur lisible (variable à définir, source), pas une trace brute de pandas."""
    monkeypatch.delenv("UNIVERS_DIR", raising=False)
    monkeypatch.delenv("MCLAB_DATA_DIR", raising=False)
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    with pytest.raises(FileNotFoundError) as e:
        rk.trouver_dossier_donnees()
    assert "UNIVERS_DIR" in str(e.value) and "histdata.com" in str(e.value)
    monkeypatch.undo()
    with pytest.raises(FileNotFoundError, match="UNIVERS_DIR"):
        rk.cloture_quotidienne("EURUSD", tmp_path)


def test_run_risque_sort_avec_un_message_d_une_ligne(monkeypatch):
    sys.path.insert(0, str(RACINE / "scripts"))
    import run_risque

    def absent():
        raise FileNotFoundError("dossier absent")

    monkeypatch.setattr(run_risque.rk, "trouver_dossier_donnees", absent)
    monkeypatch.setattr(sys, "argv", ["run_risque.py"])
    with matplotlib.rc_context(), pytest.raises(SystemExit) as e:   # main() change le style matplotlib : on le restaure
        run_risque.main()
    assert isinstance(e.value.code, str) and "absentes" in e.value.code and "\n" not in e.value.code
