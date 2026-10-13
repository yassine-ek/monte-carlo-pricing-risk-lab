"""Simulation de trajectoires : GBM exact, Heston (Euler full-truncation et QE), Merton exact.

Convention commune : chaque fonction renvoie S de forme (n_chemins, n_pas+1), S[:, 0] = S0.
  - Z (optionnel) : tirages normaux fournis par l'appelant (utile pour le QMC).
      GBM / Merton : forme (n, n_pas) ;  Heston : forme (n, n_pas, 2) avec
      Z[..., 0] = moteur de la variance, Z[..., 1] = bruit indépendant de l'actif.
  - antithetique=True : le résultat est rangé [chemins ; chemins miroirs] (n_chemins pair) et Z,
      s'il est fourni, ne contient que la première moitié (n_chemins // 2 lignes).
"""
import numpy as np
from scipy.special import ndtr

from mclab.utils import ParamsHeston, ParamsMerton


def _normales(rng, n_chemins, dims, Z, antithetique):
    """Tirages normaux de forme (n_chemins, *dims), avec miroir -Z si antithétique."""
    n_tirees = n_chemins // 2 if antithetique else n_chemins
    if antithetique and n_chemins % 2:
        raise ValueError("antithetique=True demande un n_chemins pair")
    if Z is None:
        Z = rng.standard_normal((n_tirees, *dims))
    elif Z.shape != (n_tirees, *dims):
        raise ValueError(f"Z doit avoir la forme {(n_tirees, *dims)}, reçu {Z.shape}")
    return np.concatenate([Z, -Z]) if antithetique else Z


def _avec_origine(log_increments, S0):
    """Cumule les incréments de ln S et ajoute la colonne t=0."""
    log_s = np.cumsum(log_increments, axis=1)
    return S0 * np.exp(np.concatenate([np.zeros((log_s.shape[0], 1)), log_s], axis=1))


# ---------------------------------------------------------------- GBM
def gbm_exact(S0, r, sigma, T, n_pas, n_chemins, q=0.0, graine=None, Z=None, antithetique=False):
    """Schéma exact : ln S_{t+dt} = ln S_t + (r-q-sigma^2/2) dt + sigma sqrt(dt) Z (aucun biais de discrétisation)."""
    rng = np.random.default_rng(graine)
    dt = T / n_pas
    Z = _normales(rng, n_chemins, (n_pas,), Z, antithetique)
    return _avec_origine((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, S0)


# ---------------------------------------------------------------- Heston : Euler full truncation
def heston_euler(p: ParamsHeston, T, n_pas, n_chemins, graine=None, Z=None, antithetique=False):
    """Euler « full truncation » (Lord et al. 2010) : v est tronqué à 0 partout où il sert (drift et diffusion).

    ln S est discrétisé en log (S reste positif). Biais de discrétisation en O(dt).
    """
    rng = np.random.default_rng(graine)
    dt = T / n_pas
    Z = _normales(rng, n_chemins, (n_pas, 2), Z, antithetique)
    zv = Z[..., 0]
    zs = p.rho * zv + np.sqrt(1.0 - p.rho**2) * Z[..., 1]
    v = np.full(n_chemins, p.v0)
    incr = np.empty((n_chemins, n_pas))
    for j in range(n_pas):
        v_plus = np.maximum(v, 0.0)
        incr[:, j] = (p.r - p.q - 0.5 * v_plus) * dt + np.sqrt(v_plus * dt) * zs[:, j]
        v = v + p.kappa * (p.theta - v_plus) * dt + p.xi * np.sqrt(v_plus * dt) * zv[:, j]
    return _avec_origine(incr, p.S0)


# ---------------------------------------------------------------- Heston : schéma QE (Andersen 2008)
def _variance_qe(v, zv, p: ParamsHeston, dt, psi_c):
    """Un pas de la variance par Quadratic-Exponential.

    On approche v_{t+dt} | v_t (loi de type khi-deux non centrée) par :
      - psi <= psi_c : a (b + Z)^2   (quadratique, bien adapté quand v est loin de 0)
      - psi >  psi_c : masse en 0 + queue exponentielle (bien adapté quand v est proche de 0)
    où psi = variance conditionnelle / moyenne conditionnelle au carré.
    """
    e = np.exp(-p.kappa * dt)
    m = p.theta + (v - p.theta) * e
    s2 = (v * p.xi**2 * e / p.kappa * (1.0 - e)
          + p.theta * p.xi**2 / (2.0 * p.kappa) * (1.0 - e) ** 2)
    psi = s2 / m**2
    # branche quadratique (psi borné pour ne jamais produire de NaN dans la branche inutilisée)
    inv = 2.0 / np.minimum(psi, psi_c)
    b2 = inv - 1.0 + np.sqrt(inv * (inv - 1.0))
    v_quad = m / (1.0 + b2) * (np.sqrt(b2) + zv) ** 2
    # branche exponentielle (la normale fournie est convertie en uniforme)
    psi_e = np.maximum(psi, psi_c)
    p_zero = (psi_e - 1.0) / (psi_e + 1.0)
    beta = (1.0 - p_zero) / m
    u = np.minimum(ndtr(zv), 1.0 - 1e-16)
    v_exp = np.where(u <= p_zero, 0.0, np.log((1.0 - p_zero) / (1.0 - u)) / beta)
    return np.where(psi <= psi_c, v_quad, v_exp)


def _coefficients_qe(p: ParamsHeston, dt):
    """K0..K4 de l'équation (33) d'Andersen avec gamma1 = gamma2 = 1/2 (discrétisation centrée)."""
    c = p.kappa * p.rho / p.xi - 0.5
    k0 = -p.rho * p.kappa * p.theta * dt / p.xi
    k1 = 0.5 * dt * c - p.rho / p.xi
    k2 = 0.5 * dt * c + p.rho / p.xi
    k3 = 0.5 * dt * (1.0 - p.rho**2)
    return k0, k1, k2, k3, k3  # K4 = K3 quand gamma1 = gamma2


def pas_heston_qe(log_s, v, zv, zs, p: ParamsHeston, dt, psi_c=1.5):
    """Un pas QE complet : renvoie (nouveau ln S, nouvelle variance)."""
    k0, k1, k2, k3, k4 = _coefficients_qe(p, dt)
    v_new = _variance_qe(v, zv, p, dt, psi_c)
    log_s_new = (log_s + (p.r - p.q) * dt + k0 + k1 * v + k2 * v_new
                 + np.sqrt(k3 * v + k4 * v_new) * zs)
    return log_s_new, v_new


def heston_qe(p: ParamsHeston, T, n_pas, n_chemins, graine=None, Z=None, antithetique=False, psi_c=1.5):
    """Schéma QE d'Andersen (2008) : précis même avec peu de pas, variance toujours >= 0."""
    rng = np.random.default_rng(graine)
    dt = T / n_pas
    Z = _normales(rng, n_chemins, (n_pas, 2), Z, antithetique)
    log_s = np.full(n_chemins, np.log(p.S0))
    v = np.full(n_chemins, p.v0)
    chemins = np.empty((n_chemins, n_pas + 1))
    chemins[:, 0] = p.S0
    for j in range(n_pas):
        log_s, v = pas_heston_qe(log_s, v, Z[:, j, 0], Z[:, j, 1], p, dt, psi_c)
        chemins[:, j + 1] = np.exp(log_s)
    return chemins


# ---------------------------------------------------------------- Merton
def merton_exact(p: ParamsMerton, T, n_pas, n_chemins, graine=None, Z=None, antithetique=False):
    """Merton exact : à chaque pas, N ~ Poisson(lam dt) sauts, dont la somme est N(N mu_j, N delta_j^2).

    En mode antithétique on miroite les normales (diffusion et sauts) et on garde les mêmes N.
    """
    rng = np.random.default_rng(graine)
    dt = T / n_pas
    Z = _normales(rng, n_chemins, (n_pas,), Z, antithetique)
    n_demi = n_chemins // 2 if antithetique else n_chemins
    nb_sauts = rng.poisson(p.lam * dt, size=(n_demi, n_pas))
    z_sauts = rng.standard_normal((n_demi, n_pas))
    if antithetique:
        nb_sauts = np.concatenate([nb_sauts, nb_sauts])
        z_sauts = np.concatenate([z_sauts, -z_sauts])
    esperance_j = np.exp(p.mu_j + 0.5 * p.delta_j**2) - 1.0
    somme_sauts = nb_sauts * p.mu_j + p.delta_j * np.sqrt(nb_sauts) * z_sauts
    drift = (p.r - p.q - 0.5 * p.sigma**2 - p.lam * esperance_j) * dt
    return _avec_origine(drift + p.sigma * np.sqrt(dt) * Z + somme_sauts, p.S0)
