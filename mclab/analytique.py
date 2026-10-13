"""Prix de référence : formules fermées, semi-fermées (Heston) et arbre binomial (américain).

Tout ce qui est estimé par Monte Carlo ailleurs dans le package est comparé à ces fonctions.
"""
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm, poisson

from mclab.utils import ParamsHeston, ParamsMerton


# ---------------------------------------------------------------- Black-Scholes
def _d1_d2(S0, K, T, r, sigma, q):
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bs_prix(S0, K, T, r, sigma, q=0.0, type_option="call"):
    """Prix Black-Scholes-Merton (dividende continu q) d'un call ou put européen."""
    d1, d2 = _d1_d2(S0, K, T, r, sigma, q)
    if type_option == "call":
        return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)


def bs_grecques(S0, K, T, r, sigma, q=0.0, type_option="call"):
    """Delta, gamma, vega (par unité de vol), theta (par an, dC/dt) et rho (par unité de taux)."""
    d1, d2 = _d1_d2(S0, K, T, r, sigma, q)
    eq, er, racine_t = np.exp(-q * T), np.exp(-r * T), np.sqrt(T)
    gamma = eq * norm.pdf(d1) / (S0 * sigma * racine_t)
    vega = S0 * eq * norm.pdf(d1) * racine_t
    terme_vol = -S0 * eq * norm.pdf(d1) * sigma / (2 * racine_t)
    if type_option == "call":
        delta = eq * norm.cdf(d1)
        theta = terme_vol - r * K * er * norm.cdf(d2) + q * S0 * eq * norm.cdf(d1)
        rho = K * T * er * norm.cdf(d2)
    else:
        delta = eq * (norm.cdf(d1) - 1.0)
        theta = terme_vol + r * K * er * norm.cdf(-d2) - q * S0 * eq * norm.cdf(-d1)
        rho = -K * T * er * norm.cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# ---------------------------------------------------------------- Option digitale
def digitale_prix(S0, K, T, r, sigma, q=0.0, type_option="call"):
    """Digitale « cash-or-nothing » qui paie 1 si S_T > K (call) ou S_T < K (put)."""
    _, d2 = _d1_d2(S0, K, T, r, sigma, q)
    signe = 1.0 if type_option == "call" else -1.0
    return np.exp(-r * T) * norm.cdf(signe * d2)


def digitale_delta(S0, K, T, r, sigma, q=0.0, type_option="call"):
    _, d2 = _d1_d2(S0, K, T, r, sigma, q)
    signe = 1.0 if type_option == "call" else -1.0
    return signe * np.exp(-r * T) * norm.pdf(d2) / (S0 * sigma * np.sqrt(T))


def digitale_vega(S0, K, T, r, sigma, q=0.0):
    """Vega de la digitale call : -e^{-rT} n(d2) d1 / sigma."""
    d1, d2 = _d1_d2(S0, K, T, r, sigma, q)
    return -np.exp(-r * T) * norm.pdf(d2) * d1 / sigma


# ---------------------------------------------------------------- Heston
def _fonction_caracteristique_heston(u, T, p: ParamsHeston):
    """E[exp(i u ln S_T)] sous Q — formulation « little trap » d'Albrecher et al. (2007).

    On utilise exp(-dT) (jamais exp(+dT)) : le logarithme complexe reste alors sur sa
    branche principale et l'intégrale ne saute plus pour les grandes maturités.
    """
    iu = 1j * u
    a = p.kappa - p.rho * p.xi * iu
    d = np.sqrt(a**2 + p.xi**2 * (iu + u**2))
    g = (a - d) / (a + d)
    e = np.exp(-d * T)
    c = (p.kappa * p.theta / p.xi**2) * ((a - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    dd = ((a - d) / p.xi**2) * (1.0 - e) / (1.0 - g * e)
    return np.exp(iu * (np.log(p.S0) + (p.r - p.q) * T) + c + dd * p.v0)


def heston_prix(p: ParamsHeston, K, T, type_option="call"):
    """Call/put européen Heston par inversion de Fourier : C = S0 e^{-qT} P1 - K e^{-rT} P2."""
    ln_k = np.log(K)
    phi_moins_i = _fonction_caracteristique_heston(-1j, T, p)  # = S0 e^{(r-q)T}

    def integrande_p1(u):
        phi = _fonction_caracteristique_heston(u - 1j, T, p) / phi_moins_i
        return (np.exp(-1j * u * ln_k) * phi / (1j * u)).real

    def integrande_p2(u):
        return (np.exp(-1j * u * ln_k) * _fonction_caracteristique_heston(u, T, p) / (1j * u)).real

    p1 = 0.5 + quad(integrande_p1, 0, np.inf, limit=400)[0] / np.pi
    p2 = 0.5 + quad(integrande_p2, 0, np.inf, limit=400)[0] / np.pi
    call = p.S0 * np.exp(-p.q * T) * p1 - K * np.exp(-p.r * T) * p2
    if type_option == "call":
        return call
    return call - p.S0 * np.exp(-p.q * T) + K * np.exp(-p.r * T)  # parité call-put


# ---------------------------------------------------------------- Merton
def merton_prix(p: ParamsMerton, K, T, type_option="call", n_sauts_max=80):
    """Série de Merton : mélange poissonnien de formules de Black conditionnelles au nombre de sauts."""
    esperance_j = np.exp(p.mu_j + 0.5 * p.delta_j**2) - 1.0  # E[J], sert au compensateur
    n = np.arange(n_sauts_max + 1)
    poids = poisson.pmf(n, p.lam * T)
    forward = p.S0 * np.exp((p.r - p.q - p.lam * esperance_j) * T + n * (p.mu_j + 0.5 * p.delta_j**2))
    variance_totale = p.sigma**2 * T + n * p.delta_j**2
    ecart = np.sqrt(variance_totale)
    d1 = (np.log(forward / K) + 0.5 * variance_totale) / ecart
    call_cond = np.exp(-p.r * T) * (forward * norm.cdf(d1) - K * norm.cdf(d1 - ecart))
    call = float(np.sum(poids * call_cond))
    if type_option == "call":
        return call
    return call - p.S0 * np.exp(-p.q * T) + K * np.exp(-p.r * T)


# ---------------------------------------------------------------- Asiatique géométrique
def asiatique_geo_prix(S0, K, T, r, sigma, n_dates, q=0.0, type_option="call"):
    """Asiatique géométrique à n_dates fixings équidistants t_i = iT/n (i=1..n).

    ln G est gaussien : on applique Black avec le forward exp(m + v/2).
    """
    n = n_dates
    m = np.log(S0) + (r - q - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
    v = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
    ecart = np.sqrt(v)
    d1 = (m + v - np.log(K)) / ecart
    d2 = d1 - ecart
    call = np.exp(-r * T) * (np.exp(m + 0.5 * v) * norm.cdf(d1) - K * norm.cdf(d2))
    if type_option == "call":
        return call
    return np.exp(-r * T) * (K * norm.cdf(-d2) - np.exp(m + 0.5 * v) * norm.cdf(-d1))


# ---------------------------------------------------------------- Barrière continue
def down_and_out_call(S0, K, H, T, r, sigma, q=0.0):
    """Call down-and-out, barrière continue H <= K (Merton 1973 / Reiner-Rubinstein 1991).

    C_do = C_BS - C_di avec C_di = S0 e^{-qT} (H/S0)^{2l} N(y) - K e^{-rT} (H/S0)^{2l-2} N(y - s sqrt(T)).
    """
    if H > K:
        raise ValueError("formule codée pour H <= K")
    if S0 <= H:
        return 0.0
    s_t = sigma * np.sqrt(T)
    lam = (r - q + 0.5 * sigma**2) / sigma**2
    y = np.log(H**2 / (S0 * K)) / s_t + lam * s_t
    call_di = (S0 * np.exp(-q * T) * (H / S0) ** (2 * lam) * norm.cdf(y)
               - K * np.exp(-r * T) * (H / S0) ** (2 * lam - 2) * norm.cdf(y - s_t))
    return bs_prix(S0, K, T, r, sigma, q, "call") - call_di


# ---------------------------------------------------------------- Américain (arbre CRR)
def crr_americain(S0, K, T, r, sigma, q=0.0, N=2000, type_option="put", retourner_frontiere=False,
                  pas_exercice=1):
    """Arbre binomial Cox-Ross-Rubinstein avec exercice anticipé à chaque nœud.

    Si retourner_frontiere : renvoie aussi, pour chaque pas i, le plus haut S où l'exercice est optimal.
    pas_exercice=k : exercice permis seulement tous les k pas (option bermudéenne, N/k dates), pour
    comparer à Longstaff-Schwartz sur exactement les mêmes dates d'exercice.
    """
    dt = T / N
    x = sigma * np.sqrt(dt)
    actu = np.exp(-r * dt)
    p_haut = (np.exp((r - q) * dt) - np.exp(-x)) / (np.exp(x) - np.exp(-x))
    signe = 1.0 if type_option == "call" else -1.0

    def spot(i):  # S aux i+1 nœuds du pas i (j = nombre de hausses)
        return S0 * np.exp(x * (2.0 * np.arange(i + 1) - i))

    valeur = np.maximum(signe * (spot(N) - K), 0.0)
    frontiere = np.full(N + 1, np.nan)
    for i in range(N - 1, -1, -1):
        s = spot(i)
        continuation = actu * (p_haut * valeur[1:] + (1.0 - p_haut) * valeur[:-1])
        intrinseque = np.maximum(signe * (s - K), 0.0) * (i % pas_exercice == 0)
        if retourner_frontiere:
            exercice = (intrinseque > continuation) & (intrinseque > 0)
            if exercice.any():
                frontiere[i] = s[exercice].max() if type_option == "put" else s[exercice].min()
        valeur = np.maximum(continuation, intrinseque)
    if retourner_frontiere:
        return float(valeur[0]), frontiere
    return float(valeur[0])
