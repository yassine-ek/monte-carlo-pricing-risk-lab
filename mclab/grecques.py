"""Grecques par Monte Carlo : pathwise, rapport de vraisemblance (LR) et bump-and-revalue.

Trois façons de dériver E[payoff] :
  - pathwise : on dérive le payoff le long du chemin (exige un payoff dérivable ; faux pour une digitale) ;
  - LR (score) : on dérive la densité, payoff * d ln f / d theta (marche pour tout payoff, variance plus haute) ;
  - bump : différence finie avec les MÊMES nombres aléatoires (CRN) pour que le bruit se compense.
"""
import time

import numpy as np

from mclab.analytique import bs_grecques, digitale_delta, digitale_vega
from mclab.utils import resumer

PAYOFFS = ("call", "digitale")


def _terminal(S0, T, r, sigma, q, Z):
    return S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def _valeur(payoff, S0, K, T, r, sigma, q, Z):
    """Payoff actualisé pour chaque tirage Z."""
    s_t = _terminal(S0, T, r, sigma, q, Z)
    brut = np.maximum(s_t - K, 0.0) if payoff == "call" else (s_t > K).astype(float)
    return np.exp(-r * T) * brut


def _pathwise(grecque, payoff, S0, K, T, r, sigma, q, Z):
    s_t = _terminal(S0, T, r, sigma, q, Z)
    if payoff == "digitale":  # d(1{S_T>K})/d theta = 0 presque partout : l'estimateur voit « zéro »
        return np.zeros_like(Z)
    dans_la_monnaie = (s_t > K) * np.exp(-r * T)
    if grecque == "delta":
        return dans_la_monnaie * s_t / S0
    if grecque == "vega":  # dS_T/dsigma = S_T (sqrt(T) Z - sigma T)
        return dans_la_monnaie * s_t * (np.sqrt(T) * Z - sigma * T)
    raise ValueError("pathwise : delta et vega seulement (le gamma du call est un Dirac)")


def _lr(grecque, payoff, S0, K, T, r, sigma, q, Z):
    valeur = _valeur(payoff, S0, K, T, r, sigma, q, Z)
    racine_t = np.sqrt(T)
    if grecque == "delta":
        score = Z / (S0 * sigma * racine_t)
    elif grecque == "gamma":
        score = (Z**2 - 1.0) / (S0**2 * sigma**2 * T) - Z / (S0**2 * sigma * racine_t)
    elif grecque == "vega":
        score = (Z**2 - 1.0) / sigma - racine_t * Z
    else:
        raise ValueError(f"grecque inconnue : {grecque}")
    return valeur * score


def _bump(grecque, payoff, S0, K, T, r, sigma, q, Z, h):
    def v(s0=S0, vol=sigma):
        return _valeur(payoff, s0, K, T, r, vol, q, Z)

    if grecque == "delta":
        h = 0.01 * S0 if h is None else h
        return (v(S0 + h) - v(S0 - h)) / (2 * h)
    if grecque == "gamma":
        h = 0.01 * S0 if h is None else h
        return (v(S0 + h) - 2 * v() + v(S0 - h)) / h**2
    if grecque == "vega":
        h = 0.01 if h is None else h
        return (v(vol=sigma + h) - v(vol=sigma - h)) / (2 * h)
    raise ValueError(f"grecque inconnue : {grecque}")


def grecque_mc(grecque, methode, payoff, S0, K, T, r, sigma, q=0.0, n_chemins=200_000, graine=None, h=None):
    """Estime une grecque ('delta', 'gamma', 'vega') par 'pathwise', 'lr' ou 'bump' ; renvoie un ResultatMC."""
    if payoff not in PAYOFFS:
        raise ValueError(f"payoff : {PAYOFFS}")
    t0 = time.perf_counter()
    Z = np.random.default_rng(graine).standard_normal(n_chemins)
    args = (grecque, payoff, S0, K, T, r, sigma, q, Z)
    if methode == "pathwise":
        echantillon = _pathwise(*args)
    elif methode == "lr":
        echantillon = _lr(*args)
    elif methode == "bump":
        echantillon = _bump(*args, h)
    else:
        raise ValueError("methode : 'pathwise', 'lr' ou 'bump'")
    return resumer(echantillon, t0)


def grecque_exacte(grecque, payoff, S0, K, T, r, sigma, q=0.0):
    """Valeur analytique de référence (Black-Scholes ou digitale)."""
    if payoff == "call":
        return float(bs_grecques(S0, K, T, r, sigma, q)[grecque])
    if grecque == "delta":
        return float(digitale_delta(S0, K, T, r, sigma, q))
    if grecque == "vega":
        return float(digitale_vega(S0, K, T, r, sigma, q))
    raise ValueError("digitale : delta et vega seulement")


COMBINAISONS = {  # (payoff, grecque) -> méthodes applicables
    ("call", "delta"): ("pathwise", "lr", "bump"),
    ("call", "gamma"): ("lr", "bump"),
    ("call", "vega"): ("pathwise", "lr", "bump"),
    ("digitale", "delta"): ("pathwise", "lr", "bump"),
    ("digitale", "vega"): ("pathwise", "lr", "bump"),
}


def toutes_les_grecques(S0=100.0, K=100.0, T=1.0, r=0.02, sigma=0.2, q=0.0, n_chemins=200_000, graine=0):
    """Tableau {(payoff, grecque, methode): (ResultatMC, valeur exacte)} pour toutes les combinaisons."""
    sortie = {}
    for (payoff, grecque), methodes in COMBINAISONS.items():
        exacte = grecque_exacte(grecque, payoff, S0, K, T, r, sigma, q)
        for methode in methodes:
            sortie[(payoff, grecque, methode)] = (
                grecque_mc(grecque, methode, payoff, S0, K, T, r, sigma, q, n_chemins, graine), exacte)
    return sortie
