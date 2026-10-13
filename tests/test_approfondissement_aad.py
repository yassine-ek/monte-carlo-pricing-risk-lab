"""Approfondissement Q6 : AAD en mode inverse codée à la main, contre pathwise, analytique et bump."""
import numpy as np
import pytest

from mclab import aad
from mclab.analytique import asiatique_geo_prix, bs_grecques

S0, K, T, R, SIG = 100.0, 100.0, 1.0, 0.02, 0.2


@pytest.mark.parametrize("payoff", aad.PAYOFFS)
@pytest.mark.parametrize("d", [1, 12])
def test_aad_egale_pathwise_a_1e_10_pres_sur_les_memes_chemins(payoff, d):
    Z = np.random.default_rng(d).standard_normal((20_000, d))
    a = aad.aad_boucle(payoff, S0, K, T, R, SIG, Z)
    b = aad.aad_vectorise(payoff, S0, K, T, R, SIG, Z)
    p = aad.pathwise(payoff, S0, K, T, R, SIG, Z)
    for cle in ("prix", "delta", "rho", "dK"):
        assert np.max(np.abs(a[cle] - b[cle])) < 1e-10, cle                       # boucle == somme cumulée inverse
    assert np.max(np.abs(a["delta"] - p["delta"])) < 1e-10
    assert np.max(np.abs(a["vega_seaux"].sum(axis=1) - p["vega"])) < 1e-10        # somme des vegas par seaux = vega total
    assert np.max(np.abs(a["rho"] - p["rho"])) < 1e-10
    assert np.max(np.abs(a["dK"] - p["dK"])) < 1e-10
    assert np.max(np.abs(a["vega_seaux"] - b["vega_seaux"])) < 1e-10


def test_aad_europeen_contre_black_scholes():
    Z = np.random.default_rng(1).standard_normal((600_000, 1))
    a = aad.aad_vectorise("europeen", S0, K, T, R, SIG, Z)
    g = bs_grecques(S0, K, T, R, SIG)
    for x, ref in ((a["delta"], g["delta"]), (a["vega_seaux"][:, 0], g["vega"]), (a["rho"], g["rho"])):
        assert abs(x.mean() - ref) < 4 * x.std(ddof=1) / np.sqrt(x.size)


def test_aad_geometrique_contre_differences_finies_de_la_formule_fermee():
    d = 8
    Z = np.random.default_rng(2).standard_normal((600_000, d))
    a = aad.aad_vectorise("asiatique_geo", S0, K, T, R, SIG, Z)
    h = 1e-4
    G = lambda **kw: asiatique_geo_prix(**{**dict(S0=S0, K=K, T=T, r=R, sigma=SIG, n_dates=d), **kw})
    ref = {"delta": (G(S0=S0 + h) - G(S0=S0 - h)) / (2 * h), "vega": (G(sigma=SIG + h) - G(sigma=SIG - h)) / (2 * h),
           "rho": (G(r=R + h) - G(r=R - h)) / (2 * h)}
    for nom, x in (("delta", a["delta"]), ("vega", a["vega_seaux"].sum(axis=1)), ("rho", a["rho"])):
        assert abs(x.mean() - ref[nom]) < 4 * x.std(ddof=1) / np.sqrt(x.size), nom


def test_vega_par_seaux_contre_bump_and_revalue_et_cout_beaucoup_plus_faible():
    d = 16
    Z = np.random.default_rng(3).standard_normal((20_000, d))
    sv = np.full(d, SIG)
    a = aad.aad_vectorise("asiatique", S0, K, T, R, sv, Z)
    b = aad.bump_seaux("asiatique", S0, K, T, R, sv, Z)
    assert np.max(np.abs(a["vega_seaux"].mean(axis=0) - b.mean(axis=0))) < 1e-3 * np.abs(b.mean(axis=0)).max()
    assert a["vega_seaux"].shape == (20_000, d)
    t_aad = aad.chronometrer(lambda: aad.aad_vectorise("asiatique", S0, K, T, R, sv, Z), 3)
    t_bump = aad.chronometrer(lambda: aad.bump_seaux("asiatique", S0, K, T, R, sv, Z), 1)
    assert t_aad < t_bump / 5                                    # 2 d + 1 = 33 réévaluations contre ~2 passes


def test_bump_crn_retrouve_les_grecques_moyennes():
    Z = np.random.default_rng(4).standard_normal((50_000, 6))
    a = aad.aad_boucle("asiatique", S0, K, T, R, SIG, Z)
    bp = aad.bump("asiatique", S0, K, T, R, SIG, Z)
    for cle, ref in (("delta", a["delta"]), ("rho", a["rho"]), ("dK", a["dK"])):
        assert abs(ref.mean() - bp[cle].mean()) < 1e-3 * (1 + abs(ref.mean()))
    assert abs(a["vega_seaux"].sum(axis=1).mean() - bp["vega"].mean()) < 1e-2
