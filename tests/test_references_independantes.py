"""Durcissement (revue 2026-10-13) : chaque formule est comparée à une méthode DIFFÉRENTE et exacte (~1e-8).

Constat de la revue : les tests historiques comparent surtout à du Monte Carlo (erreur-type ~1e-2) avec les paramètres
par défaut (v0 = theta pour Heston). Des mutations comme v0 <-> theta (Heston), 0,5 -> 0,4 sur la convexité des sauts de
Merton, 2 -> 1,8 dans le pont brownien, le ddl du test conditionnel de Christoffersen, ou l'évaluation de LSM sur les
chemins d'apprentissage passaient toute la suite. Ici : Riccati (solve_ivp), COS, quadrature, covariance explicite.
"""
import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp
from scipy.stats import chi2, norm

from mclab import risque as rk
from mclab.americaine import lsm_put_americain
from mclab.analytique import (_fonction_caracteristique_heston, asiatique_geo_prix, down_and_out_call,
                              heston_prix, merton_prix)
from mclab.barriere import down_and_out_mc
from mclab.utils import ParamsHeston, ParamsMerton

# v0 != theta, q != 0, rho != 0 : aucune symétrie ne masque une inversion de paramètres
HP = ParamsHeston(S0=100.0, r=0.03, q=0.01, v0=0.02, kappa=2.0, theta=0.06, xi=0.4, rho=-0.6)


def cf_riccati(u, T, p):
    """E[exp(i u ln S_T)] par intégration numérique du système de Riccati de Heston (sans « little trap »)."""
    iu = 1j * u

    def f(t, y):
        D = y[0]
        return [0.5 * p.xi**2 * D * D + (p.rho * p.xi * iu - p.kappa) * D - 0.5 * (u * u + iu), p.kappa * p.theta * D]

    y = solve_ivp(f, [0, T], [0j, 0j], method="DOP853", rtol=1e-11, atol=1e-13).y[:, -1]
    return np.exp(iu * (np.log(p.S0) + (p.r - p.q) * T) + y[1] + y[0] * p.v0)


def cos_call(phi_ln_s, S0, K, T, r, a=-4.0, b=4.0, N=300):
    """Call par la méthode COS (Fang-Oosterlee) ; phi_ln_s(u) = E[exp(i u ln S_T)] vectorisée en u."""
    k = np.arange(N)
    w = k * np.pi / (b - a)
    phi_y = np.exp(-1j * w * np.log(K)) * phi_ln_s(w)
    chi = (np.cos(w * (b - a)) * np.exp(b) - np.cos(w * (0 - a)) + w * np.sin(w * (b - a)) * np.exp(b)
           - w * np.sin(w * (0 - a))) / (1 + w**2)
    psi = np.r_[b, (np.sin(w[1:] * (b - a)) - np.sin(w[1:] * (0 - a))) / w[1:]]
    termes = np.real(phi_y * np.exp(-1j * w * a)) * 2.0 / (b - a) * K * (chi - psi)
    termes[0] *= 0.5
    return np.exp(-r * T) * termes.sum()


@pytest.mark.parametrize("T", [0.5, 3.0])
def test_heston_fonction_caracteristique_contre_riccati(T):
    for u in (0.7, 6.0, 25.0):
        assert _fonction_caracteristique_heston(u, T, HP) == pytest.approx(cf_riccati(u, T, HP), abs=1e-9)


def test_heston_prix_contre_cos():
    T = 1.0
    phi = lambda w: np.array([cf_riccati(x, T, HP) for x in w])
    for K in (90.0, 110.0):
        assert heston_prix(HP, K, T) == pytest.approx(cos_call(phi, HP.S0, K, T, HP.r), abs=1e-6)


def test_merton_contre_cos():
    p = ParamsMerton(r=0.03, q=0.01, sigma=0.25, lam=1.5, mu_j=-0.08, delta_j=0.2)
    T, k = 1.0, np.exp(-0.08 + 0.5 * 0.2**2) - 1

    def phi(u):
        return np.exp(1j * u * (np.log(p.S0) + (p.r - p.q - 0.5 * p.sigma**2 - p.lam * k) * T) - 0.5 * p.sigma**2 * u**2 * T
                      + p.lam * T * (np.exp(1j * u * p.mu_j - 0.5 * p.delta_j**2 * u**2) - 1))

    for K in (90.0, 110.0):
        assert merton_prix(p, K, T) == pytest.approx(cos_call(phi, p.S0, K, T, p.r, a=-5, b=5, N=512), abs=1e-8)


def test_asiatique_geometrique_contre_covariance_explicite():
    S0, K, T, r, s, q, n = 100.0, 95.0, 1.5, 0.03, 0.25, 0.01, 7
    t = T * np.arange(1, n + 1) / n
    m = np.log(S0) + (r - q - 0.5 * s**2) * t.mean()
    v = (s**2 * np.minimum.outer(t, t)).sum() / n**2
    d1 = (m + v - np.log(K)) / np.sqrt(v)
    attendu = np.exp(-r * T) * (np.exp(m + 0.5 * v) * norm.cdf(d1) - K * norm.cdf(d1 - np.sqrt(v)))
    assert asiatique_geo_prix(S0, K, T, r, s, n, q) == pytest.approx(attendu, abs=1e-12)


def test_barriere_continue_contre_quadrature_de_la_densite_conjointe():
    S0, K, H, T, r, s, q = 100.0, 100.0, 90.0, 1.0, 0.02, 0.2, 0.03
    nu, b, sq = r - q - 0.5 * s**2, np.log(H / S0), s * np.sqrt(T)
    dens = lambda x: (norm.pdf((x - nu * T) / sq) - np.exp(2 * nu * b / s**2) * norm.pdf((x - 2 * b - nu * T) / sq)) / sq
    ref = np.exp(-r * T) * quad(lambda x: (S0 * np.exp(x) - K) * dens(x), np.log(K / S0), nu * T + 14 * sq,
                                epsabs=1e-13, epsrel=1e-13)[0]
    assert down_and_out_call(S0, K, H, T, r, s, q) == pytest.approx(ref, abs=1e-9)


def test_pont_brownien_sans_biais_meme_a_2_dates():
    """Le pont est exact pour le GBM quel que soit m : à m = 2 l'erreur vaudrait ~0,2 si le facteur 2 était faux."""
    P = dict(S0=100.0, K=100.0, H=90.0, T=1.0, r=0.02, sigma=0.2)
    res = down_and_out_mc(m=2, n_chemins=500_000, graine=3, methode="pont", **P)
    assert abs(res.z(down_and_out_call(**P))) < 4


def test_christoffersen_cc_est_un_khi2_a_2_ddl():
    e = np.zeros(1000, int)
    e[[100, 101, 400, 700]] = 1
    c = rk.christoffersen(e)
    assert c["p_cc"] == pytest.approx(chi2.sf(c["lr_cc"], 2), rel=1e-12)
    assert c["p_ind"] == pytest.approx(chi2.sf(c["lr_ind"], 1), rel=1e-12)


def test_z2_valeur_exacte_sur_un_exemple_a_la_main():
    """Z2 = sum L_t 1{L_t > VaR_t} / (T p ES_t) - 1 : (3 + 5) / (3 * 0,025 * 4) - 1 = 25,6667."""
    L, var, es = np.array([3.0, 1.0, 5.0]), np.full(3, 2.0), np.full(3, 4.0)
    q = np.tile(np.linspace(2.0, 6.0, 200), (3, 1))
    assert rk.z2_acerbi_szekely(L, var, es, q, n_sim=100)["z2"] == pytest.approx(8 / (3 * 0.025 * 4) - 1, rel=1e-12)


def test_lsm_evaluation_independante_est_sous_l_apprentissage():
    """Avec peu de chemins d'apprentissage le biais de prévoyance est net : évaluer sur ces chemins gonflerait le prix."""
    l = lsm_put_americain(100.0, 100.0, 1.0, 0.06, 0.2, n_pas=50, n_apprentissage=2000, n_evaluation=200_000, graine=0)
    assert l.prix_apprentissage - l.resultat.prix > 3 * l.resultat.erreur_type
