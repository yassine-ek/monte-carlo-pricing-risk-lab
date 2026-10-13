"""Approfondissement Q1/Q8 : ordres de convergence d'Euler et de Milstein, biais faible de Heston, condition de Feller."""
import numpy as np

from mclab import schemas
from mclab.analytique import heston_prix
from mclab.utils import ParamsHeston

S0, MU, SIGMA, T, K = 100.0, 0.02, 0.2, 1.0, 100.0


def test_pas_de_milstein_gbm_formule_a_la_main():
    dW = np.array([[0.3], [-0.2]])
    dt = T
    attendu = S0 * (1 + MU * dt + SIGMA * dW[:, 0] + 0.5 * SIGMA**2 * (dW[:, 0] ** 2 - dt))
    assert np.allclose(schemas.gbm_terminal("milstein", S0, MU, SIGMA, T, dW), attendu, atol=1e-12)
    assert np.allclose(schemas.gbm_terminal("euler", S0, MU, SIGMA, T, dW), S0 * (1 + MU * dt + SIGMA * dW[:, 0]), atol=1e-12)
    assert np.allclose(schemas.gbm_terminal("exact", S0, MU, SIGMA, T, dW), S0 * np.exp((MU - 0.5 * SIGMA**2) * T + SIGMA * dW[:, 0]))


def test_ordres_forts_gbm_dans_les_intervalles_theoriques():
    ks = [2, 3, 4, 5, 6]
    dts = [T / 2**k for k in ks]
    E = schemas.erreurs_gbm(S0, MU, SIGMA, T, K, ks, 30_000, graine=3)
    fort_euler = schemas.pente_loglog_ic(dts, E["euler"]["fort"], graine=1)
    mse_milstein = schemas.pente_loglog_ic(dts, E["milstein"]["fort_l2"], graine=2)
    assert 0.44 < fort_euler["pente"] < 0.56                       # ordre fort 1/2
    assert fort_euler["ic95"][0] <= fort_euler["pente"] <= fort_euler["ic95"][1]
    assert 0.9 < mse_milstein["pente"] / 2 < 1.1                   # ordre fort 1 (L2 : MSE ~ dt^2)
    assert np.all(E["milstein"]["fort"].mean(0) < E["euler"]["fort"].mean(0) / 3)   # Milstein bien plus proche de la trajectoire


def test_ordre_faible_sans_bruit_vaut_un_pour_les_deux_schemas():
    ks = np.arange(2, 9)
    dts = T / 2.0**ks
    exact = schemas.moment2_gbm_exact(S0, MU, SIGMA, T)
    for s in ("euler", "milstein"):
        err = [abs(schemas.moment2_gbm_schema(s, S0, MU, SIGMA, T, 2**k) - exact) for k in ks]
        assert abs(np.polyfit(np.log(dts), np.log(err), 1)[0] - 1.0) < 0.03
    # Milstein est strictement meilleur à dt fixe (même ordre, constante plus petite)
    assert abs(schemas.moment2_gbm_schema("milstein", S0, MU, SIGMA, T, 16) - exact) < abs(schemas.moment2_gbm_schema("euler", S0, MU, SIGMA, T, 16) - exact)


def test_moment2_gbm_schema_coincide_avec_la_simulation():
    rng = np.random.default_rng(0)
    dW = np.sqrt(T / 4) * rng.standard_normal((400_000, 4))
    for s in ("euler", "milstein"):
        x = schemas.gbm_terminal(s, S0, MU, SIGMA, T, dW) ** 2
        th = schemas.moment2_gbm_schema(s, S0, MU, SIGMA, T, 4)
        assert abs(x.mean() - th) < 4 * x.std() / np.sqrt(x.size)


def test_ajustement_et_pente_ic_retrouvent_une_loi_de_puissance_bruitee():
    rng = np.random.default_rng(1)
    dts = 2.0 ** -np.arange(2, 8)
    se = 0.02 * 0.5 * dts**0.7
    err = 0.5 * dts**0.7 + se * rng.standard_normal(dts.size)
    aj = schemas.ajuster_loi_puissance(dts, err, se)
    assert aj["ic95"][0] < 0.7 < aj["ic95"][1]
    v = 0.5 * dts**0.7 * (1 + 0.1 * rng.standard_normal((5000, dts.size)))
    r = schemas.pente_loglog_ic(dts, np.abs(v), graine=2)
    assert r["ic95"][0] < 0.7 < r["ic95"][1]


def test_cir_milstein_bat_euler_quand_feller_est_respecte():
    ks = [1, 2, 3, 4, 5]
    dts = [T / 2**k for k in ks]
    err = schemas.erreurs_forte_cir(0.04, 1.5, 0.04, 0.2, T, ks, 9, 6_000, graine=4)        # F = 3
    pe = schemas.pente_loglog_ic(dts, err["euler"], graine=1)["pente"]
    pm = schemas.pente_loglog_ic(dts, err["milstein"], graine=1)["pente"]
    assert pm > pe + 0.2 and pm > 0.85
    assert np.all(err["milstein"].mean(0)[2:] < err["euler"].mean(0)[2:])


def test_heston_euler_biaise_qe_non():
    p = ParamsHeston()
    exact = heston_prix(p, K, T)
    euler = schemas.biais_faible_heston(p, K, T, "euler_ft", 8, 2_000_000, 5, exact)
    qe = schemas.biais_faible_heston(p, K, T, "qe", 8, 2_000_000, 5, exact)
    assert euler["biais"] > 20 * euler["se"] > 0       # Euler, 8 pas : +0,2 environ
    assert abs(qe["biais"]) < 4 * qe["se"]             # QE compatible avec 0 à l'erreur-type près
    fin = schemas.biais_faible_heston(p, K, T, "euler_ft", 32, 2_000_000, 6, exact)
    assert fin["biais"] < euler["biais"] / 3           # le biais décroît fortement avec dt


def test_fraction_tronquee_decroit_avec_dt_et_avec_feller():
    kappa, theta = 1.5, 0.04
    xi_f = lambda F: np.sqrt(2 * kappa * theta / F)
    fr = {(F, n): schemas.fraction_pas_tronques(kappa, theta, xi_f(F), T, n, 60_000, graine=n) for F in (0.5, 2.0) for n in (16, 64)}
    assert fr[(0.5, 64)] < fr[(0.5, 16)] and fr[(2.0, 64)] < fr[(2.0, 16)]       # décroît quand dt diminue
    assert fr[(0.5, 16)] > 10 * fr[(2.0, 16)]                                      # Feller violé : troncature massive
    pente = lambda F: np.log(fr[(F, 64)] / fr[(F, 16)]) / np.log(1 / 4)             # pente log-log en dt
    assert 0.2 < pente(0.5) < 0.7 and 1.2 < pente(2.0) < 2.4                       # ~ F, au régime pré-asymptotique près
