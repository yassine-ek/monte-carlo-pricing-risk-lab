"""Approfondissement Q2 : MLMC avec Milstein, exposants de Giles."""
import numpy as np

from mclab.analytique import bs_prix
from mclab.mlmc import (echantillonneur_gbm_euler, echantillonneur_gbm_milstein, echantillonneur_heston_euler,
                        echantillonneur_heston_milstein, estimer_exposants, mlmc, regime_giles)
from mclab.utils import ParamsHeston
from dataclasses import replace

K, T, R, SIG = 100.0, 1.0, 0.02, 0.2


def test_beta_euler_vaut_1_et_beta_milstein_vaut_2():
    e = estimer_exposants(echantillonneur_gbm_euler(100.0, K, T, R, SIG), L=6, n=40_000, graine=1)
    m = estimer_exposants(echantillonneur_gbm_milstein(100.0, K, T, R, SIG), L=6, n=40_000, graine=1)
    assert 0.8 < e["beta"] < 1.15 and e["ic_beta"][0] < e["beta"] < e["ic_beta"][1]
    assert 1.7 < m["beta"] < 2.2
    assert m["beta"] > e["beta"] + 0.7
    assert 0.8 < m["alpha"] < 1.2                      # ordre faible 1 de Milstein
    assert all(m["niveaux"][i] == i for i in range(7))


def test_regime_de_giles():
    assert regime_giles(1.0, 2.0, 1.0)["forme"] == "eps^-2"
    assert "log" in regime_giles(1.0, 1.0, 1.0)["forme"]
    assert regime_giles(1.0, 0.5, 1.0)["exposant_cout"] == 2.0 + 0.5


def test_mlmc_milstein_a_la_tolerance_et_moins_cher_qu_euler():
    ref = bs_prix(100.0, K, T, R, SIG)
    eps = 0.03
    me = mlmc(echantillonneur_gbm_milstein(100.0, K, T, R, SIG), eps, graine=2, beta=2.0)
    eu = mlmc(echantillonneur_gbm_euler(100.0, K, T, R, SIG), eps, graine=2)
    assert abs(me.resultat.prix - ref) < 3 * eps and me.resultat.erreur_type < eps
    assert me.cout < eu.cout                            # beta = 2 : moins de chemins sur les niveaux fins
    assert me.var_Y[3] < eu.var_Y[3] / 3                # variance par niveau nettement plus faible


def test_heston_milstein_beta_ne_gagne_que_si_feller_et_bruit_commutatif():
    p0 = ParamsHeston()
    sans_gain = estimer_exposants(echantillonneur_heston_milstein(p0, K, T), L=6, n=30_000, graine=3)["beta"]
    sans_gain_euler = estimer_exposants(echantillonneur_heston_euler(p0, K, T), L=6, n=30_000, graine=3)["beta"]
    p1 = replace(p0, xi=0.3, rho=-0.999)               # Feller respecté (F = 1,33) et aire de Lévy ~ 0
    gain = estimer_exposants(echantillonneur_heston_milstein(p1, K, T), L=6, n=30_000, graine=3)["beta"]
    gain_euler = estimer_exposants(echantillonneur_heston_euler(p1, K, T), L=6, n=30_000, graine=3)["beta"]
    assert sans_gain < sans_gain_euler + 0.45 and sans_gain < 1.4        # cas du projet : pas de beta ~ 2
    assert gain > 1.7 and gain > gain_euler + 0.5                        # les deux obstacles levés : beta ~ 2
