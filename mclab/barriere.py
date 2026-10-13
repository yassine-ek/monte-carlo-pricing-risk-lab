"""Option à barrière : le piège de la surveillance discrète.

La formule fermée suppose une barrière surveillée en continu. En simulant sur m dates seulement,
un chemin peut traverser H entre deux dates sans que la grille le voie : le MC « naïf » sur-estime
donc le prix d'un down-and-out (biais en O(1/sqrt(m))). Deux corrections :
  - pont brownien : on pondère chaque chemin par la probabilité de ne PAS avoir touché H entre
    deux dates consécutives, conditionnellement aux deux extrémités (biais en O(1/m) ou mieux) ;
  - décalage de Broadie-Glasserman-Kou : on rapproche H du spot, H -> H exp(+0,5826 sigma sqrt(dt)).
"""
import time

import numpy as np

from mclab.analytique import down_and_out_call
from mclab.modeles import gbm_exact
from mclab.utils import resumer

BETA_BGK = 0.5826  # -zeta(1/2)/sqrt(2 pi)


def down_and_out_mc(S0, K, H, T, r, sigma, m, n_chemins, q=0.0, graine=None, methode="naif"):
    """Call down-and-out surveillé sur m dates. methode : 'naif', 'bgk' ou 'pont'."""
    t0 = time.perf_counter()
    S = gbm_exact(S0, r, sigma, T, m, n_chemins, q, graine)
    dt = T / m
    payoff = np.exp(-r * T) * np.maximum(S[:, -1] - K, 0.0)
    if methode == "naif":
        vivant = (S[:, 1:] > H).all(axis=1)
        return resumer(payoff * vivant, t0)
    if methode == "bgk":
        h_decale = H * np.exp(BETA_BGK * sigma * np.sqrt(dt))  # vers le spot : compense les franchissements manqués
        return resumer(payoff * (S[:, 1:] > h_decale).all(axis=1), t0)
    if methode == "pont":
        a, b = np.log(S[:, :-1] / H), np.log(S[:, 1:] / H)
        # P(pas de franchissement entre deux dates | extrémités au-dessus de H) = 1 - exp(-2ab/(sigma^2 dt))
        p_survie = np.where((a > 0) & (b > 0), 1.0 - np.exp(-2.0 * a * b / (sigma**2 * dt)), 0.0)
        return resumer(payoff * p_survie.prod(axis=1), t0)
    raise ValueError("methode : 'naif', 'bgk' ou 'pont'")


def etude_biais(S0=100.0, K=100.0, H=90.0, T=1.0, r=0.02, sigma=0.2, ms=(5, 10, 25, 50, 100, 250),
                n_chemins=200_000, graine=0):
    """Erreur de chaque méthode (prix MC - formule continue) selon le nombre de dates m."""
    reference = down_and_out_call(S0, K, H, T, r, sigma)
    sortie = {"reference": reference, "ms": list(ms)}
    for methode in ("naif", "bgk", "pont"):
        res = [down_and_out_mc(S0, K, H, T, r, sigma, m, n_chemins, graine=graine + i, methode=methode)
               for i, m in enumerate(ms)]
        sortie[methode] = {"erreur": [x.prix - reference for x in res],
                           "erreur_type": [x.erreur_type for x in res]}
    return sortie
