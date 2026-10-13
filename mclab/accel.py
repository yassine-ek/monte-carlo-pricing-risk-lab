"""Accélération numba de la simulation Heston QE + benchmark numpy vs numba.

Le noyau fusionne la boucle sur les pas et sur les chemins : aucun tableau (n, n_pas) n'est
alloué, un chemin vit dans quelques registres. Le parallélisme se fait par chemin (prange).

Générateur aléatoire : numba n'offre pas default_rng. Chaque chemin utilise donc son propre flux
SplitMix64 (graine = f(graine, indice du chemin)) : le résultat dépend uniquement de la graine,
JAMAIS du nombre de threads. Ce n'est pas PCG64 : l'équivalence avec la version numpy est
statistique (mêmes prix à l'erreur-type près), pas bit à bit.
"""
import time

import numpy as np
from numba import njit, prange

from mclab.modeles import heston_qe, pas_heston_qe
from mclab.utils import ParamsHeston, resumer

_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_MULT1 = np.uint64(0xBF58476D1CE4E5B9)
_MULT2 = np.uint64(0x94D049BB133111EB)
_MULT_CHEMIN = np.uint64(0xD1B54A32D192ED03)


@njit(cache=True)
def _uniforme(etat):
    """SplitMix64 : renvoie (nouvel état, uniforme dans ]0, 1[ sur 53 bits)."""
    etat = etat + _GAMMA
    z = etat
    z = (z ^ (z >> np.uint64(30))) * _MULT1
    z = (z ^ (z >> np.uint64(27))) * _MULT2
    z = z ^ (z >> np.uint64(31))
    return etat, (float(z >> np.uint64(11)) + 0.5) * (1.0 / 9007199254740992.0)


@njit(parallel=True, cache=True)
def _noyau_heston_qe(S0, r, q, v0, kappa, theta, xi, rho, T, n_pas, n_chemins, graine, psi_c, sortie):
    dt = T / n_pas
    e = np.exp(-kappa * dt)
    c = kappa * rho / xi - 0.5
    k0 = -rho * kappa * theta * dt / xi
    k1 = 0.5 * dt * c - rho / xi
    k2 = 0.5 * dt * c + rho / xi
    k3 = 0.5 * dt * (1.0 - rho**2)  # K4 = K3 (gamma1 = gamma2 = 1/2)
    for i in prange(n_chemins):
        etat = np.uint64(graine) ^ (np.uint64(i) * _MULT_CHEMIN)
        log_s = np.log(S0)
        v = v0
        for _ in range(n_pas):
            etat, u1 = _uniforme(etat)
            etat, u2 = _uniforme(etat)
            etat, u3 = _uniforme(etat)
            rayon = np.sqrt(-2.0 * np.log(u1))
            zv = rayon * np.cos(2.0 * np.pi * u2)   # Box-Muller : deux normales indépendantes
            zs = rayon * np.sin(2.0 * np.pi * u2)
            m = theta + (v - theta) * e
            s2 = v * xi**2 * e / kappa * (1.0 - e) + theta * xi**2 / (2.0 * kappa) * (1.0 - e) ** 2
            psi = s2 / m**2
            if psi <= psi_c:
                inv = 2.0 / psi
                b2 = inv - 1.0 + np.sqrt(inv * (inv - 1.0))
                v_new = m / (1.0 + b2) * (np.sqrt(b2) + zv) ** 2
            else:
                p0 = (psi - 1.0) / (psi + 1.0)
                v_new = 0.0 if u3 <= p0 else np.log((1.0 - p0) / (1.0 - u3)) * m / (1.0 - p0)
            log_s += (r - q) * dt + k0 + k1 * v + k2 * v_new + np.sqrt(k3 * v + k3 * v_new) * zs
            v = v_new
        sortie[i] = np.exp(log_s)


def heston_qe_numba(p: ParamsHeston, T, n_pas, n_chemins, graine=0, psi_c=1.5):
    """S_T pour n_chemins trajectoires Heston QE (seulement la valeur terminale est conservée)."""
    sortie = np.empty(n_chemins)
    _noyau_heston_qe(p.S0, p.r, p.q, p.v0, p.kappa, p.theta, p.xi, p.rho, T, n_pas, n_chemins,
                     graine, psi_c, sortie)
    return sortie


def prix_call_heston_numba(p: ParamsHeston, K, T, n_pas, n_chemins, graine=0):
    t0 = time.perf_counter()
    s_t = heston_qe_numba(p, T, n_pas, n_chemins, graine)
    return resumer(np.exp(-p.r * T) * np.maximum(s_t - K, 0.0), t0)


def _numpy_terminal_seul(p: ParamsHeston, T, n_pas, n_chemins, graine):
    """Même algorithme que heston_qe mais sans stocker les trajectoires (comparaison équitable avec numba)."""
    rng = np.random.default_rng(graine)
    dt = T / n_pas
    log_s, v = np.full(n_chemins, np.log(p.S0)), np.full(n_chemins, p.v0)
    for _ in range(n_pas):
        zv, zs = rng.standard_normal(n_chemins), rng.standard_normal(n_chemins)
        log_s, v = pas_heston_qe(log_s, v, zv, zs, p, dt)
    return np.exp(log_s)


def _chrono(fonction, repetitions=3):
    """Meilleur temps sur `repetitions` exécutions (limite le bruit du système)."""
    meilleur = np.inf
    for _ in range(repetitions):
        t = time.perf_counter()
        fonction()
        meilleur = min(meilleur, time.perf_counter() - t)
    return meilleur


def benchmark(p: ParamsHeston = ParamsHeston(), T=1.0, n_pas=50, n_chemins=200_000, rep_numpy=2, rep_numba=7):
    """Temps et débit (chemins/s) : numpy trajectoires complètes, numpy terminal seul, numba parallèle."""
    heston_qe_numba(p, T, n_pas, 100)  # échauffement : exclut la compilation JIT du chronométrage
    temps = {
        "numpy_trajectoires_completes": _chrono(lambda: heston_qe(p, T, n_pas, n_chemins, graine=0), rep_numpy),
        "numpy_terminal_seul": _chrono(lambda: _numpy_terminal_seul(p, T, n_pas, n_chemins, 0), rep_numpy),
        "numba_parallele": _chrono(lambda: heston_qe_numba(p, T, n_pas, n_chemins, 0), rep_numba),
    }
    sortie = {nom: {"temps_s": t, "chemins_par_s": n_chemins / t} for nom, t in temps.items()}
    sortie["acceleration_vs_numpy_terminal"] = temps["numpy_terminal_seul"] / temps["numba_parallele"]
    sortie["acceleration_vs_numpy_complet"] = temps["numpy_trajectoires_completes"] / temps["numba_parallele"]
    sortie["n_chemins"], sortie["n_pas"] = n_chemins, n_pas
    return sortie
