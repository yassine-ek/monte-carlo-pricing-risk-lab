"""Quasi-Monte Carlo : Sobol brouillé, pont brownien, QMC randomisé, étude de convergence.

Pourquoi randomiser ? Une suite de Sobol déterministe n'a pas d'erreur-type. En la brouillant
K fois de façon indépendante, on obtient K estimations i.i.d. non biaisées : leur dispersion
donne une erreur-type honnête, tout en gardant la vitesse ~ 1/N du QMC.
"""
import time
from collections import deque

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from mclab.analytique import asiatique_geo_prix
from mclab.modeles import gbm_exact
from mclab.utils import payoff_actualise, resumer


def normales_sobol(m, d, graine):
    """2^m points de Sobol brouillés (LMS + décalage numérique, scipy) en dimension d, transformés en N(0,1) par la normale inverse."""
    u = qmc.Sobol(d=d, scramble=True, seed=np.random.default_rng(graine)).random_base2(m)
    return ndtri(np.clip(u, 1e-16, 1.0 - 1e-16))


def _plan_pont(d):
    """Ordre de remplissage du pont : (indice m, borne gauche l, borne droite r), du grossier au fin."""
    plan, file = [], deque([(0, d)])
    while file:
        l, r = file.popleft()
        if r - l > 1:
            m = (l + r) // 2
            plan.append((m, l, r))
            file.extend([(l, m), (m, r)])
    return plan


def pont_brownien(Z, T):
    """Construit W_{t_0..t_d} à partir de Z (n, d) : Z[:,0] fixe W_T, Z[:,1] le milieu, etc.

    Les premières coordonnées du point Sobol (les mieux réparties) portent donc la plus grande
    part de la variance du chemin — c'est ce qui rend le QMC efficace sur les payoffs path-dependent.
    """
    n, d = Z.shape
    dt = T / d
    W = np.zeros((n, d + 1))
    W[:, d] = np.sqrt(T) * Z[:, 0]
    for k, (m, l, r) in enumerate(_plan_pont(d), start=1):
        moyenne = ((r - m) * W[:, l] + (m - l) * W[:, r]) / (r - l)
        ecart_type = np.sqrt(dt * (m - l) * (r - m) / (r - l))
        W[:, m] = moyenne + ecart_type * Z[:, k]
    return W


def incrementer_normales(Z, T, construction):
    """Normales d'incréments à passer à gbm_exact : telles quelles, ou réordonnées par le pont."""
    if construction == "incrementale":
        return Z
    if construction == "pont":
        dt = T / Z.shape[1]
        return np.diff(pont_brownien(Z, T), axis=1) / np.sqrt(dt)
    if construction == "acp":  # décomposition en vecteurs propres de la covariance (voir anova.py)
        from mclab.anova import matrice_construction
        d = Z.shape[1]
        W = np.concatenate([np.zeros((Z.shape[0], 1)), Z @ matrice_construction(d, T, "acp").T], axis=1)
        return np.diff(W, axis=1) / np.sqrt(T / d)
    raise ValueError("construction : 'incrementale', 'pont' ou 'acp'")


def _estimation_qmc(payoff, S0, K, T, r, sigma, n_pas, m, q, seed, construction):
    """Une estimation QMC : moyenne du payoff actualisé sur UN jeu brouillé de 2^m points."""
    Z = incrementer_normales(normales_sobol(m, n_pas, seed), T, construction)
    S = gbm_exact(S0, r, sigma, T, n_pas, Z.shape[0], q, Z=Z)
    return payoff_actualise(payoff, S, K, r, T).mean()


def prix_gbm_qmc(payoff, S0, K, T, r, sigma, n_pas, m, k_brouillages=16, q=0.0, graine=0,
                 construction="pont"):
    """QMC randomisé : K brouillages indépendants de 2^m points -> moyenne, erreur-type, IC95 (Student)."""
    t0 = time.perf_counter()
    estimations = [_estimation_qmc(payoff, S0, K, T, r, sigma, n_pas, m, q, seed, construction)
                   for seed in np.random.SeedSequence(graine).spawn(k_brouillages)]
    return resumer(estimations, t0, n_chemins=k_brouillages * 2**m, student_ic=True)


def pente_loglog(N, erreurs):
    """Pente de log(erreur) en fonction de log(N) (MC attendu ~ -0,5 ; QMC ~ -1)."""
    return float(np.polyfit(np.log(N), np.log(erreurs), 1)[0])


def etude_convergence(S0=100.0, K=100.0, T=1.0, r=0.02, sigma=0.2, n_pas=16, ms=range(5, 14),
                      repetitions=30, graine=0, payoff="call_asiatique_geo"):
    """RMSE vs N pour MC, QMC incrémental et QMC pont, sur l'asiatique géométrique (référence exacte).

    Pour chaque N=2^m, on répète l'estimation `repetitions` fois indépendamment et on calcule
    RMSE = sqrt(moyenne((estimation - référence)^2)).
    """
    reference = asiatique_geo_prix(S0, K, T, r, sigma, n_pas)
    N = np.array([2**m for m in ms])
    rmse = {"mc": [], "qmc_incrementale": [], "qmc_pont": []}
    racine = np.random.SeedSequence(graine)
    for m in ms:
        graines = racine.spawn(3)
        mc = [payoff_actualise(payoff, gbm_exact(S0, r, sigma, T, n_pas, 2**m, graine=s), K, r, T).mean()
              for s in graines[0].spawn(repetitions)]
        rmse["mc"].append(np.sqrt(np.mean((np.array(mc) - reference) ** 2)))
        for nom, seed in (("incrementale", graines[1]), ("pont", graines[2])):
            qm = [_estimation_qmc(payoff, S0, K, T, r, sigma, n_pas, m, 0.0, s, nom)
                  for s in seed.spawn(repetitions)]
            rmse["qmc_" + nom].append(np.sqrt(np.mean((np.array(qm) - reference) ** 2)))
    rmse = {nom: np.array(v) for nom, v in rmse.items()}
    pentes = {nom: pente_loglog(N, v) for nom, v in rmse.items()}
    return {"N": N, "rmse": rmse, "pentes": pentes, "reference": reference}
