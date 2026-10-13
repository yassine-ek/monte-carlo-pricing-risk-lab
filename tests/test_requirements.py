"""Reproductibilité : tout paquet tiers importé par mclab doit figurer dans requirements.txt."""
import ast
import re
import sys
from pathlib import Path


def test_requirements_couvre_les_imports_de_mclab():
    """Régression (relecture critique) : `arch` (GARCH du volet risque) manquait, donc `pip install -r requirements.txt`
    suivi de pytest échouait avec ImportError sur tests/test_risque.py."""
    racine = Path(__file__).resolve().parents[1]
    lignes = (racine / "requirements.txt").read_text(encoding="utf-8").splitlines()
    requis = {re.split(r"[<>=!~\[ ]", l.strip(), maxsplit=1)[0].lower() for l in lignes if l.strip() and not l.startswith("#")}
    importes = set()
    for f in (racine / "mclab").glob("*.py"):
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Import):
                importes |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                importes.add(n.module.split(".")[0])
    externes = {m for m in importes if m not in sys.stdlib_module_names and m != "mclab"}
    assert externes <= requis, f"absents de requirements.txt : {externes - requis}"
