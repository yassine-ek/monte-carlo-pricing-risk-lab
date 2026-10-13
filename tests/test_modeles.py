"""Chaque modèle de simulation est validé contre son prix de référence, et la graine fixe tout."""
import time

import numpy as np
import pytest

from mclab.analytique import bs_prix, heston_prix, merton_prix
from mclab.modeles import gbm_exact, heston_euler, heston_qe, merton_exact
from mclab.utils import ParamsHeston, ParamsMerton, payoff_actualise, resumer

H = ParamsHeston()
M = ParamsMerton()
K, T = 100.0, 1.0


def prix_call(S, r):
    return resumer(payoff_actualise("call_europeen", S, K, r, T), time.perf_counter())


def test_forme_origine_et_reproductibilite():
    S = gbm_exact(100.0, 0.02, 0.2, T, 7, 50, graine=3)
    assert S.shape == (50, 8) and np.all(S[:, 0] == 100.0)
    assert np.array_equal(S, gbm_exact(100.0, 0.02, 0.2, T, 7, 50, graine=3))
    assert not np.array_equal(S, gbm_exact(100.0, 0.02, 0.2, T, 7, 50, graine=4))
    for f in (lambda g: heston_qe(H, T, 5, 40, graine=g), lambda g: heston_euler(H, T, 5, 40, graine=g),
              lambda g: merton_exact(M, T, 5, 40, graine=g)):
        assert np.array_equal(f(1), f(1))


def test_tirages_fournis_et_antithetique():
    Z = np.random.default_rng(0).standard_normal((20, 6))
    assert np.array_equal(gbm_exact(100.0, 0.02, 0.2, T, 6, 20, Z=Z), gbm_exact(100.0, 0.02, 0.2, T, 6, 20, Z=Z))
    S = gbm_exact(100.0, 0.02, 0.2, T, 6, 40, Z=Z, antithetique=True)
    assert S.shape == (40, 7)
    drift = (0.02 - 0.5 * 0.2**2) * T  # chemin + miroir : les log-rendements centrés s'annulent
    assert np.allclose(np.log(S[:20, -1] / 100.0) + np.log(S[20:, -1] / 100.0), 2 * drift)
    with pytest.raises(ValueError):
        gbm_exact(100.0, 0.02, 0.2, T, 6, 40, Z=Z)  # mauvaise forme sans antithétique
    with pytest.raises(ValueError):
        gbm_exact(100.0, 0.02, 0.2, T, 6, 41, antithetique=True)


def test_gbm_exact_contre_bs():
    ref = bs_prix(100.0, K, T, 0.02, 0.2)
    res = prix_call(gbm_exact(100.0, 0.02, 0.2, T, 4, 200_000, graine=11), 0.02)
    assert abs(res.z(ref)) < 3


def test_heston_qe_contre_semi_ferme():
    ref = heston_prix(H, K, T)
    res = prix_call(heston_qe(H, T, 20, 100_000, graine=12), H.r)
    assert abs(res.z(ref)) < 3


def test_heston_euler_biais_de_discretisation_documente():
    # Euler full-truncation a un biais en O(dt) (de l'ordre de 0,02 à 50 pas pour ces paramètres) :
    # on tolère 3 erreurs-types + 0,05, soit ~0,6 % du prix.
    ref = heston_prix(H, K, T)
    res = prix_call(heston_euler(H, T, 50, 100_000, graine=13), H.r)
    assert abs(res.prix - ref) < 3 * res.erreur_type + 0.05


def test_heston_qe_positif_et_martingale():
    S = heston_qe(H, T, 10, 100_000, graine=14)
    assert np.all(S > 0)
    assert abs(S[:, -1].mean() - H.S0 * np.exp(H.r * T)) < 3 * S[:, -1].std() / np.sqrt(len(S))


def test_merton_exact_contre_serie():
    ref = merton_prix(M, K, T)
    res = prix_call(merton_exact(M, T, 4, 200_000, graine=15), M.r)
    assert abs(res.z(ref)) < 3
