"""Chaque technique de réduction de variance doit rester sans biais ET faire gagner de la précision."""
import pytest

from mclab.analytique import asiatique_geo_prix, bs_prix
from mclab.reduction_variance import (facteur_efficacite, facteur_variance, mc_antithetique, mc_controle,
                                      mc_importance_call, mc_naif)

A = dict(S0=100.0, K=100.0, T=1.0, r=0.02, sigma=0.2)
N = 100_000


def test_europeenne_sans_biais_et_plus_precise():
    ref = bs_prix(**A)
    naif = mc_naif("call_europeen", n_pas=1, n_chemins=N, graine=1, **A)
    anti = mc_antithetique("call_europeen", n_pas=1, n_chemins=N, graine=1, **A)
    ctrl = mc_controle("call_europeen", n_pas=1, n_chemins=N, graine=1, **A)
    for res in (naif, anti, ctrl):
        assert abs(res.z(ref)) < 3
    assert anti.erreur_type < naif.erreur_type
    assert ctrl.erreur_type < naif.erreur_type / 2


def test_asiatique_geometrique_mc_contre_formule():
    ref = asiatique_geo_prix(100.0, 100.0, 1.0, 0.02, 0.2, 12)
    assert abs(mc_naif("call_asiatique_geo", n_pas=12, n_chemins=N, graine=2, **A).z(ref)) < 3


def test_variable_de_controle_asiatique_arithmetique():
    naif = mc_naif("call_asiatique_arith", n_pas=12, n_chemins=N, graine=3, **A)
    ctrl = mc_controle("call_asiatique_arith", n_pas=12, n_chemins=N, graine=3, **A)
    assert facteur_variance(naif, ctrl) > 10
    assert facteur_efficacite(naif, ctrl) > 10  # y compris le temps du pilote
    assert abs(naif.prix - ctrl.prix) < 3 * naif.erreur_type  # les deux estiment la même quantité
    assert ctrl.ic_bas < ctrl.prix < ctrl.ic_haut


def test_importance_sampling_call_tres_otm():
    b = dict(S0=100.0, K=180.0, T=1.0, r=0.02, sigma=0.2)
    ref = bs_prix(**b)
    naif = mc_naif("call_europeen", n_pas=1, n_chemins=N, graine=4, **b)
    imp = mc_importance_call(n_chemins=N, graine=4, **b)
    assert abs(imp.z(ref)) < 3
    assert facteur_efficacite(naif, imp) > 10


def test_reproductibilite_par_graine():
    a = mc_controle("call_asiatique_arith", n_pas=12, n_chemins=5000, graine=7, **A)
    b = mc_controle("call_asiatique_arith", n_pas=12, n_chemins=5000, graine=7, **A)
    assert a.prix == b.prix and a.erreur_type == b.erreur_type
    with pytest.raises(ValueError):
        mc_controle("put_europeen", n_pas=1, n_chemins=1000, **A)
