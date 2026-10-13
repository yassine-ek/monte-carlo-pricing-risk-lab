"""Différentiation algorithmique adjointe (AAD, mode inverse) codée à la main pour un GBM discret.

Principe (Griewank, « cheap gradient principle »). Un programme qui calcule un prix P(theta_1..theta_m) en F opérations
élémentaires donne TOUTES ses dérivées dP/dtheta_i en environ 3 à 4 F opérations, quel que soit m, si on remonte le
graphe de calcul à l'envers en propageant les « adjoints » (bar x = dP / dx) :
    avant  : a_j = (r - q - sigma_j^2/2) dt + sigma_j sqrt(dt) Z_j ,  S_j = S_{j-1} exp(a_j) ,  P = e^{-rT} g(S_1..S_d)
    arrière: bar S_j = dP/dS_j direct + bar S_{j+1} (S_{j+1} / S_j) ,  bar a_j = bar S_j S_j ,
             bar sigma_j = bar a_j (-sigma_j dt + sqrt(dt) Z_j) ,  bar r = sum_j bar a_j dt - T P ,  bar S_0 = bar S_1 S_1 / S_0
Le bump-and-revalue, lui, coûte 2 m réévaluations. Ici m = d + 3 (delta, rho, strike et UN vega par date de fixing : vega
« par seaux » d'une vol à terme), donc l'écart de coût est le facteur m / 2 qu'on ne peut pas éviter autrement.
Les formules pathwise (dérivées à la main, en avant) servent de contre-vérification EXACTE sur les mêmes chemins.
"""
import time

import numpy as np

PAYOFFS = ("europeen", "asiatique", "asiatique_geo")


def _avant(S0, r, q, sigma, T, Z):
    """Passe avant : (S_1..S_d) de forme (n, d), vols par date (d,), pas dt."""
    n, d = Z.shape
    dt = T / d
    sig = np.broadcast_to(np.asarray(sigma, float), (d,))
    a = (r - q - 0.5 * sig**2) * dt + sig * np.sqrt(dt) * Z
    return S0 * np.exp(np.cumsum(a, axis=1)), sig, dt


def _payoff_et_directes(payoff, S, K, r, T):
    """(payoff actualisé (n,), indicatrice (n,), dérivées directes disc * dg/dS_j (n, d))."""
    n, d = S.shape
    disc = np.exp(-r * T)
    if payoff == "europeen":
        brut = S[:, -1] - K
        direct = np.zeros_like(S)
        ind = brut > 0
        direct[:, -1] = disc * ind
    elif payoff == "asiatique":
        brut = S.mean(axis=1) - K
        ind = brut > 0
        direct = (disc * ind / d)[:, None] * np.ones_like(S)
    elif payoff == "asiatique_geo":
        G = np.exp(np.log(S).mean(axis=1))
        brut = G - K
        ind = brut > 0
        direct = (disc * ind * G / d)[:, None] / S          # dG/dS_j = G / (d S_j)
    else:
        raise ValueError(f"payoff : {PAYOFFS}")
    return disc * np.maximum(brut, 0.0), ind, direct


def valeur(payoff, S0, K, T, r, sigma, Z, q=0.0):
    """Payoff actualisé par chemin, passe avant SEULE (sans dérivées directes) : prix, bump-and-revalue, chronométrage.

    Version volontairement minimale : c'est la référence de coût « 1 » du principe de Griewank, elle ne doit pas payer
    le calcul des adjoints que seule l'AAD utilise.
    """
    S, _, _ = _avant(S0, r, q, sigma, T, Z)
    if payoff == "europeen":
        brut = S[:, -1] - K
    elif payoff == "asiatique":
        brut = S.mean(axis=1) - K
    elif payoff == "asiatique_geo":
        brut = np.exp(np.log(S).mean(axis=1)) - K
    else:
        raise ValueError(f"payoff : {PAYOFFS}")
    return np.exp(-r * T) * np.maximum(brut, 0.0)


def aad_boucle(payoff, S0, K, T, r, sigma, Z, q=0.0):
    """AAD en mode inverse, boucle explicite sur les dates (version pédagogique, la plus proche du cours).

    Renvoie {'prix','delta','rho','dK' : (n,), 'vega_seaux' : (n, d)} — dérivées PAR CHEMIN (leur moyenne estime la grecque).
    Les tableaux sont transposés en (d, n) pour que chaque date soit un bloc contigu en mémoire : sans cela, le coût par
    date croît avec d (accès par colonne d'un tableau rangé par lignes) et masque le coût en O(operations) du principe.
    """
    S, sig, dt = _avant(S0, r, q, sigma, T, Z)
    n, d = S.shape
    prix, ind, direct = _payoff_et_directes(payoff, S, K, r, T)
    disc = np.exp(-r * T)
    St, Dt, Zt = np.ascontiguousarray(S.T), np.ascontiguousarray(direct.T), np.ascontiguousarray(Z.T)
    vega = np.empty((d, n))
    r_bar = -T * prix                       # dérivée explicite par le facteur d'actualisation
    acc = np.zeros(n)                       # bar S_j (adjoint du spot à la date j)
    for j in range(d - 1, -1, -1):          # j = d-1 .. 0 : dates t_{j+1}, ..., t_1
        acc = Dt[j] + (acc * St[j + 1] / St[j] if j < d - 1 else 0.0)
        a_bar = acc * St[j]                                        # d S_j / d a_j = S_j
        vega[j] = a_bar * (-sig[j] * dt + np.sqrt(dt) * Zt[j])
        r_bar = r_bar + a_bar * dt
    delta = acc * St[0] / S0                # bar S_0 = bar S_1 * S_1 / S_0
    return {"prix": prix, "delta": delta, "rho": r_bar, "dK": -disc * ind, "vega_seaux": vega.T}


def aad_vectorise(payoff, S0, K, T, r, sigma, Z, q=0.0):
    """Même calcul, balayage inverse VECTORISÉ : bar S_j S_j = somme_{i>=j} direct_i S_i (somme cumulée inverse).

    Le produit des rapports S_{l+1}/S_l se télescope en S_i / S_j : l'adjoint d'une exponentielle cumulée est une somme
    cumulée dans l'autre sens. Mêmes nombres que `aad_boucle` à l'arrondi près, mais sans boucle Python.
    """
    S, sig, dt = _avant(S0, r, q, sigma, T, Z)
    prix, ind, direct = _payoff_et_directes(payoff, S, K, r, T)
    disc = np.exp(-r * T)
    a_bar = np.cumsum((direct * S)[:, ::-1], axis=1)[:, ::-1]      # bar a_j
    vega = a_bar * (-sig * dt + np.sqrt(dt) * Z)
    return {"prix": prix, "delta": a_bar[:, 0] / S0, "rho": -T * prix + dt * a_bar.sum(axis=1),
            "dK": -disc * ind, "vega_seaux": vega}


def pathwise(payoff, S0, K, T, r, sigma, Z, q=0.0):
    """Dérivées pathwise écrites en AVANT à la main (vol CONSTANTE sigma scalaire) : référence exacte pour l'AAD.

    d S_j / d S_0 = S_j / S_0 ;  d S_j / d sigma = S_j (W_j - sigma t_j) ;  d S_j / d r = S_j t_j.
    """
    n, d = Z.shape
    S, _, dt = _avant(S0, r, q, float(sigma), T, Z)
    prix, ind, direct = _payoff_et_directes(payoff, S, K, r, T)
    t = T * np.arange(1, d + 1) / d
    W = np.sqrt(dt) * np.cumsum(Z, axis=1)
    return {"prix": prix,
            "delta": np.sum(direct * S / S0, axis=1),
            "vega": np.sum(direct * S * (W - sigma * t), axis=1),
            "rho": -T * prix + np.sum(direct * S * t, axis=1),
            "dK": -np.exp(-r * T) * ind}


def bump(payoff, S0, K, T, r, sigma, Z, q=0.0, h_rel=1e-5):
    """Différences finies centrées, MÊMES Z (CRN) : delta, vega (vol constante), rho, dK. 8 réévaluations + 1."""
    def v(**kw):
        a = dict(S0=S0, K=K, T=T, r=r, sigma=sigma)
        a.update(kw)
        return valeur(payoff, a["S0"], a["K"], a["T"], a["r"], a["sigma"], Z, q)

    h0, hs, hr, hk = h_rel * S0, h_rel, h_rel, h_rel * K
    return {"delta": (v(S0=S0 + h0) - v(S0=S0 - h0)) / (2 * h0),
            "vega": (v(sigma=sigma + hs) - v(sigma=sigma - hs)) / (2 * hs),
            "rho": (v(r=r + hr) - v(r=r - hr)) / (2 * hr),
            "dK": (v(K=K + hk) - v(K=K - hk)) / (2 * hk)}


def bump_seaux(payoff, S0, K, T, r, sigma_vec, Z, q=0.0, h=1e-5):
    """Vega par seaux par bump-and-revalue : 2 d réévaluations (une bosse sur la vol de CHAQUE date)."""
    sig = np.asarray(sigma_vec, float).copy()
    sortie = np.empty((Z.shape[0], sig.size))
    for j in range(sig.size):
        hp, hm = sig.copy(), sig.copy()
        hp[j] += h
        hm[j] -= h
        sortie[:, j] = (valeur(payoff, S0, K, T, r, hp, Z, q) - valeur(payoff, S0, K, T, r, hm, Z, q)) / (2 * h)
    return sortie


def chronometrer(f, repetitions=5):
    """Meilleur temps mural (s) sur `repetitions` exécutions : le minimum écarte les interruptions du système.

    (`time.process_time` aurait l'avantage d'ignorer les autres programmes, mais sa résolution est de 15,6 ms sous Windows.)
    """
    meilleur = np.inf
    for _ in range(repetitions):
        t = time.perf_counter()
        f()
        meilleur = min(meilleur, time.perf_counter() - t)
    return meilleur
