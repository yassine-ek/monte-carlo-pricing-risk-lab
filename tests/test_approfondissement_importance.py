"""Approfondissement Q5 : importance sampling optimal (second moment, entropie croisée, grandes déviations)."""
import numpy as np
from scipy.special import ndtr, log_ndtr
from scipy.stats import binom

from mclab import importance as im
from mclab.analytique import bs_prix

S0, K, T, R, SIG = 100.0, 180.0, 1.0, 0.02, 0.2
F = im.payoff_call(S0, K, T, R, SIG)
ZK = im.z_strike(S0, K, T, R, SIG)
KW = dict(a=ZK, b=20.0)


def test_esperance_par_quadrature_est_le_prix_black_scholes():
    assert abs(im.esperance(F, **KW) - bs_prix(S0, K, T, R, SIG)) < 1e-12


def test_second_moment_convexe_et_minimum_unique():
    mus = np.linspace(0.0, 7.0, 71)
    logm = np.log([im.second_moment(F, m, **KW) for m in mus])
    assert np.all(np.diff(logm, 2) > -1e-9)                          # ln M convexe
    mu_star = im.mu_optimal(F, bracket=(ZK - 1.0, ZK + 2.0), **KW)
    assert 3.0 < mu_star < 4.0 and mu_star > ZK                      # le minimum est à DROITE du strike (payoff croissant)
    m_star = im.second_moment(F, mu_star, **KW)
    assert m_star <= min(im.second_moment(F, mu_star + h, **KW) for h in (-0.05, 0.05))
    assert m_star < im.second_moment(F, ZK, **KW) < im.second_moment(F, 0.0, **KW)


def test_variance_empirique_egale_second_moment_moins_carre():
    I = im.esperance(F, **KW)
    for mu in (2.0, ZK, 3.4):
        th = im.variance_is(F, mu, I, **KW)
        est, var = im.estimer_is(F, mu, 400_000, graine=7)
        assert abs(var / th - 1) < 0.1
        assert abs(est - I) < 4 * np.sqrt(var / 400_000)


def test_entropie_croisee_iterative_converge_vers_le_point_fixe_exact():
    mu_ce = im.mu_entropie_croisee_exact(F, **KW)
    traj = im.entropie_croisee(F, n_iter=8, n=20_000, graine=1)
    assert abs(traj[-1] - mu_ce) < 0.1
    assert abs(mu_ce - im.mu_optimal(F, bracket=(ZK - 1.0, ZK + 2.0), **KW)) < 0.3      # CE et second moment sont voisins
    I = im.esperance(F, **KW)
    # la dérive CE rend la variance aussi petite (à 5 % près) que la dérive optimale
    assert im.variance_is(F, traj[-1], I, **KW) < 1.05 * im.variance_is(F, im.mu_optimal(F, bracket=(ZK - 1.0, ZK + 2.0), **KW), I, **KW)


def test_second_moment_evenement_rare_forme_fermee():
    a = 3.0
    indicatrice = lambda z: (np.asarray(z) > a).astype(float)
    for mu in (0.0, 2.0, 3.0):
        quad = im.second_moment(indicatrice, mu, a=a, b=25.0)
        assert abs(quad / (np.exp(mu**2) * ndtr(-(a + mu))) - 1) < 1e-8


def test_efficacite_logarithmique_tend_vers_un():
    r = im.efficacite_log_evenement([2, 4, 6, 10, 20, 40])
    rap = [x["rapport_log"] for x in r]
    assert all(np.diff(rap) > 0) and rap[-1] > 0.99 and rap[0] < 0.9
    assert all(abs(x["mu_star"] - x["a"]) < 0.3 for x in r)           # mu* ~ a : on translate jusqu'au seuil
    assert r[-1]["erreur_relative_is"] < 10 < r[-1]["erreur_relative_naive"]
    assert abs(r[0]["log_P"] - log_ndtr(-2.0)) < 1e-12


def test_portefeuille_de_credit_probabilite_et_derive_negative():
    fc = im.perte_portefeuille_credit(100, 0.01, 0.2, 10)
    P = im.esperance(fc)
    rng = np.random.default_rng(0)
    y = rng.standard_normal(2_000_000)
    p_cond = ndtr((-2.3263478740408408 - np.sqrt(0.2) * y) / np.sqrt(0.8))
    ref = binom.sf(9, 100, p_cond)
    assert abs(P - ref.mean()) < 4 * ref.std() / np.sqrt(ref.size)
    mu_star = im.mu_optimal(fc, bracket=(-5.0, 0.0))
    assert -4.5 < mu_star < -1.5                                     # on tire des conjonctures NÉGATIVES (facteur Y bas)
    assert im.variance_is(fc, mu_star, P) < im.variance_is(fc, 0.0, P) / 20
