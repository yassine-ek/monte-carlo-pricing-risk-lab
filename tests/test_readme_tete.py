"""README : structure imposée, blocs générés égaux aux JSON de résultats, premier écran, figure héro lisible, longueur."""
import json
import re
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
README = (RACINE / "README.md").read_text(encoding="utf-8")
sys.path.insert(0, str(RACINE / "scripts"))


def _json(nom):
    return json.loads((RACINE / "results" / nom).read_text(encoding="utf-8"))


def _bloc(cle):
    return README.split(f"<!-- AUTO:{cle}:DEBUT -->", 1)[1].split(f"<!-- AUTO:{cle}:FIN -->", 1)[0].strip()


def test_sections_imposees_dans_l_ordre():
    ordre = ["# Laboratoire Monte Carlo", "AUTO:tete_glance:DEBUT", "actions/workflows/tests.yml/badge.svg", "## Résultats clés",
             "hero_convergence_et_var.png", "**Aller plus loin**", "## Frais et hypothèses", "## Métriques", "## Overfit ou pas ?",
             "## Toutes les variantes", "## Pourquoi ce projet", "AUTO:approfondissement:DEBUT", "## Limites honnêtes",
             "## Comment ce projet a été construit", "## Reproduire", "### Obtenir les données"]
    positions = [README.index(m) for m in ordre]
    assert positions == sorted(positions)
    assert (RACINE / "figures" / "hero_convergence_et_var.png").is_file()
    for cle in ("tete_glance", "tete_cles", "tete_validation", "tete_risque", "tete_lecture", "variantes_pricing", "variantes_vr",
                "variantes_var", "approfondissement"):
        assert len(_bloc(cle)) > 100, f"bloc {cle} vide"


def test_premier_ecran_et_longueur():
    lignes = README.splitlines()
    assert len(lignes) <= 400
    tete = "\n".join(lignes[:45])
    assert "## Résultats clés" in tete and "figures/hero_convergence_et_var.png" in tete
    assert len([l for l in _bloc("tete_glance").splitlines() if l.strip()]) == 3 and "At a glance" in _bloc("tete_glance")
    assert len([l for l in _bloc("tete_cles").splitlines() if l.startswith("- ")]) == 3
    assert README.count("<details>") >= 3
    assert not re.search(r"tests-\d+%20passed", README), "le badge de tests doit être le badge de CI vivant, pas un nombre saisi"


def test_tableaux_hors_details_de_7_colonnes_au_plus():
    visible = re.sub(r"<details>.*?</details>", "", README, flags=re.S)
    for l in visible.splitlines():
        if l.startswith("|"):
            assert l.count("|") - 1 <= 7, l[:80]


def test_pas_de_chemin_absolu_ni_de_45_cas():
    for interdit in ("C:" + chr(92) + "Users", "C:/" + "Users"):          # construits ici pour ne pas apparaître dans le code
        assert interdit not in README
    assert not re.search(r"\b45 cas\b", README), "« 42 cas validés + 3 contre-exemples volontaires » partout"


def test_comptes_de_tete_lisent_validation_json():
    val = _json("validation.json")
    demos = [c for c in val["cas"] if c["attendu_echec"]]
    valides = [c for c in val["cas"] if not c["attendu_echec"] and c["verdict_ok"]]
    assert len(valides) + len(demos) == len(val["cas"])
    for bloc in ("tete_cles", "tete_validation"):
        assert f"{len(valides)} cas validés + {len(demos)} contre-exemples volontaires" in _bloc(bloc)
    assert f"{len(valides)} validated cases" in _bloc("tete_glance") and f"{len(demos)} deliberate failure demos" in _bloc("tete_glance")


def test_tableau_de_validation_lit_validation_json():
    val = _json("validation.json")
    tete = _bloc("tete_validation")
    cas = {c["id"]: c for c in val["cas"]}
    assert f"{cas['bs_qmc']['estimation']:.4f}" in tete and f"{cas['bs_qmc']['reference']:.4f}" in tete


def test_tableau_de_risque_lit_risque_backtest_json():
    risque = _json("risque_backtest.json")
    tete = _bloc("tete_risque")
    assert len(risque["methodes"]) == 5
    for m in risque["methodes"].values():
        b = m["backtest_var"]
        assert f"{b['n_exceptions']} (attendu" in tete
        assert f"{b['kupiec_p']:.3f}" in tete


def test_figure_hero_deux_panneaux_et_police_lisible():
    """Réduite de moitié par GitHub, la figure garde un texte d'au moins 11 px (et la police source dépasse 11 pt)."""
    import build_readme_tete_01 as b

    fig = b.figure_hero(b.charger("validation.json"), b.charger("risque_backtest.json"))
    try:
        assert len(fig.axes) == 2
        police = b.taille_police_min(fig)
        assert police >= 11
        assert police * b.DPI_HERO / 72 * 0.5 >= 11
    finally:
        b.plt.close(fig)
