"""Tests du volet risque : formules fermees, backtests, absence de fuite temporelle."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, special, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # racine du projet (import mclab)
from mclab import risque as rk  # noqa: E402


# --------------------------------------------------------------------------- #
# Formules fermees
# --------------------------------------------------------------------------- #
def _sigma_connue(n=4, sigma_p=0.004, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    S = A @ A.T
    w = np.array([0.4, 0.3, 0.2, 0.1])[:n]
    return S * sigma_p ** 2 / (w @ S @ w), w          # w' S w = sigma_p^2 exactement


def test_var_es_normale_formule_fermee():
    sigma_p = 0.004
    assert rk.var_normale(sigma_p, 1e7) == pytest.approx(1e7 * sigma_p * 2.3263478740408408, rel=1e-12)
    es_exact = 1e7 * sigma_p * 2.337802792201415          # phi(z_97.5%) / 2,5 %
    assert rk.es_normale(sigma_p, 1e7) == pytest.approx(es_exact, rel=1e-12)
    # l'ES est bien la moyenne de la fonction quantile sur [97,5 % ; 100 %]
    moy_quantiles = integrate.quad(stats.norm.ppf, 0.975, 1 - 1e-13)[0] / 0.025
    assert rk.es_normale(1.0, 1.0) == pytest.approx(moy_quantiles, rel=1e-6)


def test_prevision_normale_sur_donnees_simulees():
    S, w = _sigma_connue()
    X = np.random.default_rng(1).multivariate_normal(np.zeros(4), S, size=400_000)
    sigma_p = np.sqrt(w @ S @ w)
    p = rk.prevision_normale(X, w, 1e7, lam=1.0)        # lam = 1 : poids egaux
    assert p.var99 == pytest.approx(rk.var_normale(sigma_p, 1e7), rel=0.01)
    assert p.es975 == pytest.approx(rk.es_normale(sigma_p, 1e7), rel=0.01)
    # la simulation historique converge vers la meme valeur (rendements simples ~ log a 0,4 %)
    h = rk.prevision_hs(X, w, 1e7)
    assert h.var99 == pytest.approx(p.var99, rel=0.03)
    assert h.es975 == pytest.approx(p.es975, rel=0.03)


def test_cov_ewma_recursion():
    R = np.random.default_rng(2).normal(size=(60, 3)) * 0.01
    lam = 0.94
    S = np.zeros((3, 3))
    poids = 0.0
    for r in R:                                        # recursion RiskMetrics avec normalisation
        S = lam * S + np.outer(r, r)
        poids = lam * poids + 1.0
    assert rk.cov_ewma(R, lam) == pytest.approx(S / poids, rel=1e-10)


def test_student_multivarie_covariance_et_nu():
    S, _ = _sigma_connue()
    X = rk.simuler_t_multivarie(S, 6.0, 600_000, np.random.default_rng(3))
    assert np.cov(X.T) == pytest.approx(S, rel=0.06)
    z = stats.t.rvs(5.0, size=20_000, random_state=4) * np.sqrt(3.0 / 5.0)   # variance 1, nu = 5
    assert rk.nu_student_mle(z) == pytest.approx(5.0, abs=0.6)


def test_marge_copule_interpolee_proche_exacte():
    x = stats.t.rvs(4.0, size=50_000, random_state=5)
    nu = 6.0
    exact = special.stdtrit(nu, special.stdtr(4.0, x)) * np.sqrt((nu - 2) / nu)
    ok = np.abs(x) < 100
    assert np.abs(rk.transformee_marge(x, 4.0, nu)[ok] - exact[ok]).max() < 1e-4   # mesuré : 4,6e-5 (1e-3 était 20x trop lâche)


def test_stats_echantillon_normale():
    L = np.random.default_rng(6).standard_normal(2_000_000)
    p = rk.stats_echantillon(L)
    assert p.var99 == pytest.approx(2.3263, abs=0.01)
    assert p.var975 == pytest.approx(1.9600, abs=0.01)
    assert p.es975 == pytest.approx(2.3378, abs=0.01)


# --------------------------------------------------------------------------- #
# Backtests
# --------------------------------------------------------------------------- #
def _kupiec_sur_modele(sigma_modele, reps=400, T=2500, seed=7):
    """P-values de Kupiec quand la vraie vol est 1 % et que le modele annonce sigma_modele."""
    r = np.random.default_rng(seed).normal(0.0, 0.01, size=(reps, T))
    var = rk.var_normale(sigma_modele, 1.0)
    x = (-r > var).sum(axis=1)
    return np.array([rk.kupiec_pof(xi, T)[1] for xi in x])


def test_kupiec_valeurs_exactes():
    lr, p = rk.kupiec_pof(25, 2500)                    # taux observe = taux attendu
    assert lr == pytest.approx(0.0, abs=1e-9) and p == pytest.approx(1.0)
    lr0, _ = rk.kupiec_pof(0, 250)                     # LR = -2 T ln(1-p) quand x = 0
    assert lr0 == pytest.approx(-2 * 250 * np.log(0.99), rel=1e-12)


def test_kupiec_accepte_le_vrai_modele():
    p = _kupiec_sur_modele(0.01)
    assert np.mean(p < 0.05) < 0.12                    # taille du test proche de 5 %
    assert np.mean(p < 0.05) > 0.01
    assert np.median(p) > 0.2


def test_kupiec_rejette_un_modele_faux():
    p = _kupiec_sur_modele(0.0075)                     # vol sous-estimee de 25 % : ~4 % d'exceptions
    assert np.all(p < 1e-4)


def test_christoffersen_independance():
    rng = np.random.default_rng(8)
    iid = (rng.random(2500) < 0.01).astype(int)
    assert rk.christoffersen(iid)["p_ind"] > 0.01
    groupes = np.zeros(2500, int)                      # 25 exceptions mais en 5 paquets de 5 jours
    for d in (200, 700, 1200, 1700, 2200):
        groupes[d:d + 5] = 1
    c = rk.christoffersen(groupes)
    assert c["p_ind"] < 0.01 and c["p_cc"] < 0.01
    assert rk.kupiec_pof(groupes.sum(), 2500)[1] > 0.9   # Kupiec seul ne voit pas le probleme


def test_feux_bale():
    assert [rk.zone_bale_compte(k) for k in (0, 4, 5, 9, 10, 30)] == \
        ["vert", "vert", "orange", "orange", "rouge", "rouge"]
    exc = np.zeros(1000, int)
    exc[800:807] = 1
    f = rk.feux_bale(exc)
    assert f["max_exceptions_250j"] == 7 and f["pire_zone"] == "orange"
    assert f["zone_derniere_fenetre"] == "orange"
    assert f["part_vert"] + f["part_orange"] + f["part_rouge"] == pytest.approx(1.0)


def _z2_sur_modele(vol_reelle, T=1000, seed=9):
    """Modele annonce N(0,1) ; pertes reelles N(0, vol_reelle)."""
    K = 200
    q = np.tile(stats.norm.ppf(rk.niveaux_queue(K)), (T, 1))
    var975 = np.full(T, stats.norm.ppf(0.975))
    es = np.full(T, q[0].mean())
    L = np.random.default_rng(seed).normal(0.0, vol_reelle, T)
    return rk.z2_acerbi_szekely(L, var975, es, q, n_sim=2000)


def test_es_z2_vrai_modele_et_modele_faux():
    ok = _z2_sur_modele(1.0)
    assert ok["p_value"] > 0.05 and abs(ok["z2"]) < 3 * ok["z2_simule_sd"]
    faux = _z2_sur_modele(1.3)                          # ES sous-estime
    assert faux["p_value"] < 0.01 and faux["z2"] > 0.2


# --------------------------------------------------------------------------- #
# Absence de fuite temporelle
# --------------------------------------------------------------------------- #
def test_aucune_fuite_temporelle():
    """Changer les rendements a partir du jour t* ne change aucune prevision jusqu'en t*."""
    rng = np.random.default_rng(10)
    n, T, fen, t_star = 3, 330, 250, 275
    R = rng.standard_t(5, size=(T, n)) * 0.01
    w = np.array([0.5, 0.3, 0.2])
    R2 = R.copy()
    R2[t_star:] = rng.normal(0.0, 0.2, size=(T - t_star, n))     # futur (et jour t*) falsifies
    R2[t_star] = -0.3                                            # krach au jour t*
    kw = dict(fenetre=fen, n_sim=800, refit=10, debut=fen)
    a = rk.backtest_roulant(R, w, fin=t_star + 1, **kw)
    b = rk.backtest_roulant(R2, w, fin=t_star + 1, **kw)
    assert a["jours"][-1] == t_star
    for m in rk.METHODES:
        for champ in ("var99", "var975", "es975", "queue"):
            assert np.array_equal(a["methodes"][m][champ], b["methodes"][m][champ]), (m, champ)
            assert np.all(np.isfinite(a["methodes"][m][champ]))
    # controle positif : la prevision du jour t*+1, elle, voit le rendement falsifie de t*
    a2 = rk.backtest_roulant(R, w, fin=t_star + 2, **kw)
    b2 = rk.backtest_roulant(R2, w, fin=t_star + 2, **kw)
    for m in rk.METHODES:
        assert a2["methodes"][m]["var99"][-1] != b2["methodes"][m]["var99"][-1], m


def test_prevision_ne_depend_que_de_la_fenetre():
    R = np.random.default_rng(11).normal(0, 0.01, size=(400, 3))
    w = np.array([0.5, 0.3, 0.2])
    full = rk.backtest_roulant(R, w, fenetre=250, n_sim=500, debut=300, fin=301, methodes=("hs", "normale", "mc_t"))
    R_tronque = R.copy()
    R_tronque[:50] = 99.0                                # avant la fenetre [50, 300[ : sans effet
    bis = rk.backtest_roulant(R_tronque, w, fenetre=250, n_sim=500, debut=300, fin=301, methodes=("hs", "normale", "mc_t"))
    for m in ("hs", "normale", "mc_t"):
        assert np.array_equal(full["methodes"][m]["var99"], bis["methodes"][m]["var99"])


# --------------------------------------------------------------------------- #
# Donnees
# --------------------------------------------------------------------------- #
def test_actif_mort_refuse():
    for a in ("WTIUSD", "GRXEUR", "XAUEUR"):
        with pytest.raises(ValueError):
            rk.cloture_quotidienne(a)


def test_cloture_quotidienne_jours_de_trading(tmp_path):
    # vendredi 17h-> dimanche 16h : bar de dimanche isolee (stub) et lundi complet
    idx = pd.date_range("2024-01-05 00:00", "2024-01-05 16:00", freq="h")          # vendredi
    idx = idx.append(pd.DatetimeIndex(["2024-01-07 16:00"]))                         # stub dimanche
    idx = idx.append(pd.date_range("2024-01-07 17:00", "2024-01-08 16:00", freq="h"))  # lundi (trading day)
    px = np.arange(1.0, len(idx) + 1.0)
    pd.DataFrame({"datetime": idx, "close": px}).to_csv(tmp_path / "TESTAA_1h.csv", index=False)
    s = rk.cloture_quotidienne("TESTAA", tmp_path)
    assert list(s.index.strftime("%Y-%m-%d")) == ["2024-01-05", "2024-01-08"]
    assert s.iloc[0] == px[16]                                                       # barre 16h du vendredi
    assert s.iloc[1] == px[-1]                                                       # barre 16h du lundi
