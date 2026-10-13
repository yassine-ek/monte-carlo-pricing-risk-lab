"""Approfondissement Q3 : constructions du brownien, ANOVA / Sobol, dimension effective et RMSE du QMC."""
import numpy as np

from mclab import anova
from mclab.qmc import incrementer_normales, pont_brownien


def test_toutes_les_constructions_reproduisent_la_covariance_du_brownien():
    d = 16
    i = np.arange(1, d + 1)
    cov = (1.0 / d) * np.minimum.outer(i, i)
    for c in anova.CONSTRUCTIONS:
        A = anova.matrice_construction(d, 1.0, c)
        assert np.allclose(A @ A.T, cov, atol=1e-12), c
    # le pont de la matrice == le pont de qmc.py
    Z = np.random.default_rng(0).standard_normal((50, d))
    assert np.allclose(Z @ anova.matrice_construction(d, 1.0, "pont").T, pont_brownien(Z, 1.0)[:, 1:], atol=1e-12)


def test_acp_ordonne_la_variance_et_incrementer_normales_acp_est_coherent():
    A = anova.matrice_construction(16, 1.0, "acp")
    var_colonnes = (A**2).sum(axis=0)
    assert np.all(np.diff(var_colonnes) <= 1e-12)             # valeurs propres décroissantes
    assert var_colonnes[0] > 0.8 * var_colonnes.sum()          # un seul mode porte plus de 80 % de la variance du brownien
    Z = np.random.default_rng(1).standard_normal((100_000, 8))
    inc = incrementer_normales(Z, 1.0, "acp")
    assert abs(inc.std() - 1.0) < 0.01 and np.abs(np.corrcoef(inc.T) - np.eye(8)).max() < 0.02   # incréments N(0,1) indépendants


def test_indices_de_sobol_fermes_contre_la_formule_exacte():
    a = np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.05, 0.05])
    f, exact = anova.exponentielle_lineaire(a)
    res = anova.variance_expliquee(f, 8, [1, 2, 4, 8], n=262_144, graine=1)
    for k, (s, se) in res.items():
        assert abs(s - exact(k)) < 4 * se + 0.005, k
    assert res[8][0] == 1.0


def test_indices_premier_ordre_hermite_contre_la_formule_exacte():
    a = np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.05, 0.05])
    f, _ = anova.exponentielle_lineaire(a)
    S, se = anova.indices_premier_ordre(f, 8, n=262_144, graine=2)
    exact = (np.exp(a**2) - 1) / (np.exp((a**2).sum()) - 1)
    assert np.all(np.abs(S - exact) < 4 * se + 0.01)
    assert S.sum() < 1.0 + 0.02                     # la partie additive ne dépasse pas la variance totale


def test_variance_concentree_dans_les_premieres_coordonnees_acp_pont_incrementale():
    d, ks = 16, [1, 2, 4]
    res = {}
    for c in anova.CONSTRUCTIONS:
        A = anova.matrice_construction(d, 1.0, c)
        res[c] = anova.variance_expliquee(lambda Z, A=A: anova.payoff_asiatique(Z, A), d, ks, n=65_536, graine=3)
    for k in ks:
        assert res["acp"][k][0] > res["pont"][k][0] > res["incrementale"][k][0], k
    assert res["acp"][1][0] > 0.95 and res["pont"][1][0] > 0.65 and res["incrementale"][1][0] < 0.2
    assert anova.dimension_troncature({k: v for k, v in res["acp"].items()}, 0.95) == 1


def test_rmse_qmc_ordre_acp_pont_incrementale_mc():
    d, ms = 16, [8, 10]
    r = {c: anova.rmse_qmc_construction(c, d, ms, repetitions=20, graine=5) for c in anova.CONSTRUCTIONS}
    mc = anova.rmse_mc(d, ms, repetitions=40, graine=5)
    for j in range(len(ms)):
        assert r["acp"]["rmse"][j] < r["pont"]["rmse"][j] < r["incrementale"]["rmse"][j] < mc["rmse"][j]
    assert -0.65 < mc["pente"] < -0.3                # MC ~ -0,5
    assert r["pont"]["pente"] < mc["pente"] - 0.2    # le QMC va plus vite que le MC
