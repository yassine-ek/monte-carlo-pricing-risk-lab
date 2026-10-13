"""Grecques MC : chaque estimateur applicable doit être à moins de 3 erreurs-types de la valeur exacte."""
import pytest

from mclab.grecques import COMBINAISONS, grecque_exacte, grecque_mc, toutes_les_grecques

P = dict(S0=100.0, K=100.0, T=1.0, r=0.02, sigma=0.2)


def test_toutes_les_methodes_applicables_sont_calibrees():
    for (payoff, grecque, methode), (res, exacte) in toutes_les_grecques(n_chemins=200_000, graine=3).items():
        if payoff == "digitale" and methode == "pathwise":
            continue  # traité dans le test suivant : l'estimateur est faux par construction
        assert abs(res.z(exacte)) < 3, (payoff, grecque, methode, res.prix, exacte)


def test_pathwise_echoue_sur_la_digitale_mais_pas_le_rapport_de_vraisemblance():
    exacte = grecque_exacte("delta", "digitale", **P)
    pw = grecque_mc("delta", "pathwise", "digitale", n_chemins=50_000, graine=1, **P)
    lr = grecque_mc("delta", "lr", "digitale", n_chemins=50_000, graine=1, **P)
    assert pw.prix == 0.0 and abs(pw.prix - exacte) > 0.019  # la dérivée de l'indicatrice est nulle p.s.
    assert abs(lr.z(exacte)) < 3


def test_variances_relatives_pathwise_lr():
    """Pour le call, pathwise a une variance bien plus faible que LR (il exploite la régularité du payoff)."""
    pw = grecque_mc("delta", "pathwise", "call", n_chemins=100_000, graine=2, **P)
    lr = grecque_mc("delta", "lr", "call", n_chemins=100_000, graine=2, **P)
    assert pw.erreur_type < lr.erreur_type / 1.5


def test_combinaisons_interdites_et_reproductibilite():
    with pytest.raises(ValueError):
        grecque_mc("gamma", "pathwise", "call", n_chemins=100, **P)
    a = grecque_mc("vega", "lr", "call", n_chemins=1000, graine=9, **P)
    assert a.prix == grecque_mc("vega", "lr", "call", n_chemins=1000, graine=9, **P).prix
    assert ("digitale", "gamma") not in COMBINAISONS
