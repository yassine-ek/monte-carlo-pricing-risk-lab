"""Réduction de variance : antithétique, variable de contrôle, importance sampling.

Tous les estimateurs renvoient un ResultatMC ; l'erreur-type est toujours calculée sur des
unités i.i.d. (paires antithétiques moyennées, ou payoffs corrigés par le contrôle).
"""
import time

import numpy as np

from mclab.analytique import asiatique_geo_prix
from mclab.modeles import gbm_exact
from mclab.utils import payoff_actualise, resumer, resumer_paires


def mc_naif(payoff, S0, K, T, r, sigma, n_pas, n_chemins, q=0.0, graine=None):
    """Monte Carlo standard (référence de comparaison)."""
    t0 = time.perf_counter()
    S = gbm_exact(S0, r, sigma, T, n_pas, n_chemins, q, graine)
    return resumer(payoff_actualise(payoff, S, K, r, T), t0)


def mc_antithetique(payoff, S0, K, T, r, sigma, n_pas, n_chemins, q=0.0, graine=None):
    """Chaque tirage Z est accompagné de -Z ; on moyenne les deux payoffs avant de calculer l'erreur-type."""
    t0 = time.perf_counter()
    S = gbm_exact(S0, r, sigma, T, n_pas, n_chemins, q, graine, antithetique=True)
    return resumer_paires(payoff_actualise(payoff, S, K, r, T), t0)


def _controle(payoff, S, S0, K, T, r, sigma, n_pas, q):
    """(valeur du contrôle par chemin, son espérance exacte)."""
    if payoff == "call_europeen":  # S_T actualisé : E = S0 e^{-qT}
        return np.exp(-r * T) * S[:, -1], S0 * np.exp(-q * T)
    if payoff == "call_asiatique_arith":  # asiatique géométrique : corrélation ~0,99+
        return (payoff_actualise("call_asiatique_geo", S, K, r, T),
                asiatique_geo_prix(S0, K, T, r, sigma, n_pas, q))
    raise ValueError(f"pas de contrôle prévu pour {payoff}")


def mc_controle(payoff, S0, K, T, r, sigma, n_pas, n_chemins, q=0.0, graine=None, n_pilote=20000):
    """Variable de contrôle : estime E[Y - beta (X - E X)], avec beta = Cov(Y, X) / Var(X).

    beta est estimé sur un échantillon pilote INDÉPENDANT de l'échantillon principal : sinon
    l'estimateur serait légèrement biaisé (beta dépendrait des mêmes chemins que la moyenne).
    Le temps du pilote est inclus dans temps_s.
    """
    t0 = time.perf_counter()
    graine_pilote, graine_main = np.random.SeedSequence(graine).spawn(2)
    S_p = gbm_exact(S0, r, sigma, T, n_pas, n_pilote, q, graine_pilote)
    x_p, _ = _controle(payoff, S_p, S0, K, T, r, sigma, n_pas, q)
    y_p = payoff_actualise(payoff, S_p, K, r, T)
    beta = np.cov(y_p, x_p)[0, 1] / np.var(x_p, ddof=1)

    S = gbm_exact(S0, r, sigma, T, n_pas, n_chemins, q, graine_main)
    x, esperance_x = _controle(payoff, S, S0, K, T, r, sigma, n_pas, q)
    y = payoff_actualise(payoff, S, K, r, T)
    return resumer(y - beta * (x - esperance_x), t0)


def mu_optimal_call(S0, K, T, r, sigma, q=0.0):
    """Translation de la normale qui centre S_T sur le strike (là où le payoff devient non nul)."""
    return max(0.0, (np.log(K / S0) - (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)))


def mc_importance_call(S0, K, T, r, sigma, n_chemins, q=0.0, graine=None, mu=None):
    """Importance sampling par translation de drift pour un call très hors de la monnaie.

    On tire Y ~ N(mu, 1) au lieu de N(0, 1) : beaucoup plus de chemins finissent dans la monnaie.
    Poids de vraisemblance : phi(Y) / phi_mu(Y) = exp(-mu Y + mu^2 / 2) (estimateur sans biais).
    """
    t0 = time.perf_counter()
    mu = mu_optimal_call(S0, K, T, r, sigma, q) if mu is None else mu
    y = np.random.default_rng(graine).standard_normal(n_chemins) + mu
    s_t = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * y)
    poids = np.exp(-mu * y + 0.5 * mu**2)
    return resumer(np.exp(-r * T) * np.maximum(s_t - K, 0.0) * poids, t0)


def facteur_variance(naif, vr):
    """Rapport des variances PAR CHEMIN (indépendant du temps de calcul)."""
    return (naif.erreur_type**2 * naif.n_chemins) / (vr.erreur_type**2 * vr.n_chemins)


def facteur_efficacite(naif, vr):
    """(var_naïf · t_naïf) / (var_vr · t_vr) : gain réel à temps de calcul égal.

    var·t = SE² · temps_total ne dépend pas de n (SE² ~ 1/n, temps ~ n), donc on peut comparer
    deux estimateurs de tailles différentes.
    """
    return (naif.erreur_type**2 * naif.temps_s) / (vr.erreur_type**2 * vr.temps_s)
