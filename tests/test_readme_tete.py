"""Tête du README (SPEC_GITHUB) : sections imposées présentes, tableaux remplis, nombres égaux à ceux des JSON de résultats."""
import json
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
README = (RACINE / "README.md").read_text(encoding="utf-8")


def _json(nom):
    return json.loads((RACINE / "results" / nom).read_text(encoding="utf-8"))


def test_sections_imposees_dans_l_ordre():
    ordre = ["# Laboratoire Monte Carlo", "img.shields.io/badge/tests-", "## Hypothèses", "## Métriques de validation",
             "hero_convergence_et_var.png", "## Fiable ou pas ?", "## Toutes les variantes", "## Pourquoi ce projet",
             "## Approfondissement"]
    positions = [README.index(m) for m in ordre]
    assert positions == sorted(positions)
    assert (RACINE / "figures" / "hero_convergence_et_var.png").is_file()
    for cle in ("tete_validation", "tete_risque", "tete_lecture", "variantes_pricing", "variantes_vr", "variantes_var"):
        debut, fin = f"<!-- AUTO:{cle}:DEBUT -->", f"<!-- AUTO:{cle}:FIN -->"
        assert README.index(debut) < README.index(fin)
        assert len(README.split(debut, 1)[1].split(fin, 1)[0].strip()) > 100, f"bloc {cle} vide"


def test_pas_de_chemin_absolu():
    for interdit in ("C:" + chr(92) + "Users", "C:/" + "Users"):          # construits ici pour ne pas apparaître dans le code
        assert interdit not in README


def test_tableau_de_validation_lit_validation_json():
    val = _json("validation.json")
    tete = README.split("<!-- AUTO:tete_validation:DEBUT -->", 1)[1].split("<!-- AUTO:tete_validation:FIN -->", 1)[0]
    assert f"{len(val['cas'])} cas au total" in tete
    cas = {c["id"]: c for c in val["cas"]}
    assert f"{cas['bs_qmc']['estimation']:.4f}" in tete and f"{cas['bs_qmc']['reference']:.4f}" in tete


def test_tableau_de_risque_lit_risque_backtest_json():
    risque = _json("risque_backtest.json")
    tete = README.split("<!-- AUTO:tete_risque:DEBUT -->", 1)[1].split("<!-- AUTO:tete_risque:FIN -->", 1)[0]
    assert len(risque["methodes"]) == 5
    for m in risque["methodes"].values():
        b = m["backtest_var"]
        assert f"{b['n_exceptions']} (attendu" in tete
        assert f"{b['kupiec_p']:.3f}" in tete
