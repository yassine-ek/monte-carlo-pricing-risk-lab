"""Put américain par Longstaff-Schwartz (2001), avec évaluation hors échantillon.

Idée : à chaque date, on compare le gain immédiat (K - S) à la valeur de continuation, qu'on ne
connaît pas mais qu'on RÉGRESSE sur les chemins dans la monnaie (polynômes de Laguerre de S/K).

Deux jeux de chemins INDÉPENDANTS :
  1. apprentissage : on estime les coefficients de régression, donc la règle d'exercice ;
  2. évaluation : on applique cette règle figée à de nouveaux chemins.
La règle est sous-optimale pour la vraie loi, donc l'estimateur est biaisé BAS (jamais haut) ;
évaluer sur les chemins d'apprentissage donnerait un biais de prévoyance (la règle « connaît »
ces chemins) qui pousse le prix vers le haut.

Pour encadrer le prix par le HAUT on ajoute une borne duale (Rogers 2002, Haugh-Kogan 2004) :
tout martingale M (M_0 = 0) donne  prix_bermudéen <= E[ max_j ( D_j gain_j - M_j ) ].
On construit M avec la fonction de valeur approchée issue de la régression LSM.
"""
import time
from dataclasses import dataclass

import numpy as np

from mclab.modeles import gbm_exact
from mclab.utils import resumer


def base_laguerre(x, degre=3):
    """Matrice (len(x), degre+1) des polynômes de Laguerre L_0..L_degre (récurrence de Bonnet)."""
    colonnes = [np.ones_like(x), 1.0 - x]
    for k in range(1, degre):
        colonnes.append(((2 * k + 1 - x) * colonnes[k] - k * colonnes[k - 1]) / (k + 1))
    return np.column_stack(colonnes[: degre + 1])


@dataclass
class ResultatLSM:
    resultat: object           # ResultatMC hors échantillon (estimateur biaisé bas)
    prix_apprentissage: float  # même règle évaluée sur les chemins d'apprentissage (biaisé haut)
    coefs: list                # coefs[j] : régression à la date t_j (j = 1..n_pas-1)
    dates: np.ndarray          # t_0..t_{n_pas}
    frontiere: np.ndarray      # S*(t_j) : sous cette valeur, exercer est optimal (NaN si aucune)
    valeur_approchee: tuple    # (coefs dans la monnaie, coefs tous chemins) de degré degre_valeur
    parametres: dict           # S0, K, T, r, sigma, q, n_pas, degre_valeur (pour la borne duale)


def _gain(S, K, signe):
    return np.maximum(signe * (K - S), 0.0)


def _apprendre(S, K, r, dt, degre, signe, degre_valeur=7):
    """Récurrence arrière : renvoie (coefs par date, valeur actualisée en t=0 par chemin, valeur approchée).

    valeur_approchee = (cv_monnaie, cv_tous) : régressions de degré degre_valeur du flux de la politique,
    sur les chemins dans la monnaie puis sur tous les chemins (pour la zone hors de la monnaie).
    Elles ne servent qu'à la borne duale : elles ne modifient pas la règle d'exercice.
    """
    n_pas = S.shape[1] - 1
    coefs = [None] * n_pas
    cv_monnaie, cv_tous = [None] * n_pas, [None] * n_pas
    flux = _gain(S[:, n_pas], K, signe)  # flux de la politique courante, exprimé en valeur à la date j
    for j in range(n_pas - 1, 0, -1):
        flux = flux * np.exp(-r * dt)
        gain = _gain(S[:, j], K, signe)
        dans_la_monnaie = gain > 0
        if dans_la_monnaie.sum() <= max(degre, degre_valeur) + 1:
            continue
        cv_tous[j] = np.linalg.lstsq(base_laguerre(S[:, j] / K, degre_valeur), flux, rcond=None)[0]
        cv_monnaie[j] = np.linalg.lstsq(base_laguerre(S[dans_la_monnaie, j] / K, degre_valeur),
                                        flux[dans_la_monnaie], rcond=None)[0]
        base = base_laguerre(S[dans_la_monnaie, j] / K, degre)
        coefs[j] = np.linalg.lstsq(base, flux[dans_la_monnaie], rcond=None)[0]
        continuation = base @ coefs[j]
        exercer = gain[dans_la_monnaie] > continuation
        indices = np.flatnonzero(dans_la_monnaie)[exercer]
        flux[indices] = gain[indices]
    return coefs, flux * np.exp(-r * dt), (cv_monnaie, cv_tous)


def _evaluer(S, K, r, dt, coefs, degre, signe):
    """Applique la règle d'exercice figée (coefs) à de nouveaux chemins ; renvoie les payoffs actualisés."""
    n, n_pas = S.shape[0], S.shape[1] - 1
    actif = np.ones(n, dtype=bool)
    valeur = np.zeros(n)
    for j in range(1, n_pas + 1):
        gain = _gain(S[:, j], K, signe)
        candidats = actif & (gain > 0)
        if j < n_pas and coefs[j] is not None:
            idx = np.flatnonzero(candidats)
            continuation = base_laguerre(S[idx, j] / K, degre) @ coefs[j]
            idx = idx[gain[idx] > continuation]
        elif j < n_pas:
            idx = np.array([], dtype=int)
        else:
            idx = np.flatnonzero(candidats)  # à l'échéance on exerce si dans la monnaie
        valeur[idx] = gain[idx] * np.exp(-r * j * dt)
        actif[idx] = False
    return valeur


def _frontiere(coefs, K, degre, signe):
    """S* par date : plus grand S de la grille où gain > continuation (put). NaN si l'exercice n'est jamais optimal."""
    grille = np.linspace(0.3 * K, K, 2000)
    base = base_laguerre(grille / K, degre)
    sortie = np.full(len(coefs), np.nan)
    for j, c in enumerate(coefs):
        if c is None:
            continue
        ecart = (K - grille) - base @ c
        positifs = np.flatnonzero(ecart > 0)
        if positifs.size:
            sortie[j] = grille[positifs.max()]
    return sortie


def lsm_put_americain(S0, K, T, r, sigma, q=0.0, n_pas=50, n_apprentissage=50_000, n_evaluation=100_000,
                      degre=3, graine=0, degre_valeur=7):
    """Put bermudéen à n_pas dates (limite américaine quand n_pas grandit) par Longstaff-Schwartz."""
    t0 = time.perf_counter()
    dt = T / n_pas
    graine_app, graine_eval = np.random.SeedSequence(graine).spawn(2)
    S_app = gbm_exact(S0, r, sigma, T, n_pas, n_apprentissage, q, graine_app, antithetique=True)
    coefs, valeurs_app, valeur_approchee = _apprendre(S_app, K, r, dt, degre, 1.0, degre_valeur)
    S_eval = gbm_exact(S0, r, sigma, T, n_pas, n_evaluation, q, graine_eval, antithetique=True)
    valeurs = _evaluer(S_eval, K, r, dt, coefs, degre, signe=1.0)
    plancher = max(K - S0, 0.0)  # la valeur en t=0 est au moins le gain immédiat
    res = resumer_hors_echantillon(valeurs, t0, plancher)
    parametres = dict(S0=S0, K=K, T=T, r=r, sigma=sigma, q=q, n_pas=n_pas, degre_valeur=degre_valeur)
    return ResultatLSM(res, max(float(valeurs_app.mean()), plancher), coefs,
                       np.linspace(0.0, T, n_pas + 1), _frontiere(coefs, K, degre, 1.0),
                       valeur_approchee, parametres)


def _valeur_approchee(S, j, lsm, K):
    """Q_j(S) = max(gain, continuation approchée), continuation >= 0 ; régression 'tous chemins' si S >= K."""
    gain = np.maximum(K - S, 0.0)
    if j == lsm.parametres["n_pas"] or lsm.valeur_approchee[0][j] is None:
        return gain
    base = base_laguerre(S / K, lsm.parametres["degre_valeur"])
    continuation = np.where(S < K, base @ lsm.valeur_approchee[0][j], base @ lsm.valeur_approchee[1][j])
    return np.maximum(gain, np.maximum(continuation, 0.0))


def borne_haute_duale(lsm, n_exterieurs=3000, graine=1, n_noeuds=161):
    """Borne HAUTE du prix bermudéen (dualité de Rogers / Haugh-Kogan).

    Martingale : M_j = somme_k D_k (Q_k(S_k) - E[Q_k(S_k) | S_{k-1}]) avec Q = valeur approchée LSM.
    L'espérance conditionnelle est calculée par quadrature sur la loi normale du pas (le pas GBM est
    exactement lognormal) : aucun bruit de sous-simulation. Borne = E[max_j (D_j gain_j - M_j)].
    Elle est valide quelle que soit la qualité de Q ; Q plus fidèle => borne plus serrée.
    """
    t0 = time.perf_counter()
    p = lsm.parametres
    n_pas, dt, K = p["n_pas"], p["T"] / p["n_pas"], p["K"]
    S = gbm_exact(p["S0"], p["r"], p["sigma"], p["T"], n_pas, n_exterieurs, p["q"], graine)
    x = np.linspace(-6.0, 6.0, n_noeuds)
    poids = np.exp(-0.5 * x**2)
    poids /= poids.sum()
    drift, vol = (p["r"] - p["q"] - 0.5 * p["sigma"] ** 2) * dt, p["sigma"] * np.sqrt(dt)
    martingale = np.zeros(n_exterieurs)
    maximum = np.full(n_exterieurs, max(K - p["S0"], 0.0))
    for j in range(1, n_pas + 1):
        suivants = S[:, j - 1][:, None] * np.exp(drift + vol * x[None, :])
        esperance = (_valeur_approchee(suivants.ravel(), j, lsm, K).reshape(suivants.shape) * poids).sum(axis=1)
        actu = np.exp(-p["r"] * j * dt)
        martingale += actu * (_valeur_approchee(S[:, j], j, lsm, K) - esperance)
        maximum = np.maximum(maximum, actu * np.maximum(K - S[:, j], 0.0) - martingale)
    return resumer(maximum, t0)


def resumer_hors_echantillon(valeurs, t0, plancher):
    """Résumé en paires antithétiques (chemins rangés [moitié ; miroirs]) avec plancher sur la valeur."""
    moitie = valeurs.size // 2
    paires = 0.5 * (valeurs[:moitie] + valeurs[moitie:])
    res = resumer(paires, t0, n_chemins=valeurs.size)
    if res.prix < plancher:  # exercice immédiat meilleur : la valeur du contrat est le plancher
        res.prix = res.ic_bas = res.ic_haut = plancher
    return res
