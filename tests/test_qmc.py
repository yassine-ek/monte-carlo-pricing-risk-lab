"""QMC : le pont brownien est correct, l'estimateur randomisé est calibré, la pente de convergence est meilleure que MC."""
import numpy as np

from mclab.analytique import asiatique_geo_prix
from mclab.qmc import (etude_convergence, normales_sobol, pente_loglog, pont_brownien, prix_gbm_qmc)

A = dict(S0=100.0, K=100.0, T=1.0, r=0.02, sigma=0.2)


def test_pont_brownien_a_la_bonne_covariance():
    Z = np.random.default_rng(0).standard_normal((200_000, 8))
    W = pont_brownien(Z, 1.0)
    t = np.arange(9) / 8
    assert np.allclose(W.var(axis=0)[1:], t[1:], atol=0.01)  # Var(W_t) = t
    assert abs(np.cov(W[:, 2], W[:, 6])[0, 1] - 0.25) < 0.01  # Cov(W_s, W_t) = min(s, t)
    assert np.all(W[:, 0] == 0)


def test_sobol_brouille_est_reproductible_et_centre():
    Z = normales_sobol(10, 4, graine=1)
    assert Z.shape == (1024, 4) and np.array_equal(Z, normales_sobol(10, 4, graine=1))
    assert np.all(np.abs(Z.mean(axis=0)) < 0.05) and np.all(np.abs(Z.std(axis=0) - 1) < 0.05)


def test_qmc_randomise_sans_biais_et_plus_precis_que_mc():
    ref = asiatique_geo_prix(100.0, 100.0, 1.0, 0.02, 0.2, 16)
    res = prix_gbm_qmc("call_asiatique_geo", n_pas=16, m=10, k_brouillages=16, graine=5, **A)
    assert abs(res.z(ref)) < 3
    # à 16 384 chemins, l'erreur-type MC vaudrait environ 0,1 ; le QMC randomisé fait beaucoup mieux
    assert res.erreur_type < 0.02
    assert res.n_chemins == 16 * 2**10


def test_pente_de_convergence_qmc_meilleure_que_mc():
    etude = etude_convergence(ms=range(5, 12), repetitions=30, graine=0)
    pentes = etude["pentes"]
    assert -0.65 < pentes["mc"] < -0.35        # MC : ~ -0,5
    assert pentes["qmc_pont"] < -0.75           # QMC avec pont : proche de -1
    assert pentes["qmc_pont"] < pentes["qmc_incrementale"] < pentes["mc"]
    assert etude["rmse"]["qmc_pont"][-1] < etude["rmse"]["mc"][-1] / 10


def test_pente_loglog_exacte_sur_une_loi_de_puissance():
    N = 2.0 ** np.arange(4, 12)
    assert abs(pente_loglog(N, 3.0 * N**-0.5) + 0.5) < 1e-12
