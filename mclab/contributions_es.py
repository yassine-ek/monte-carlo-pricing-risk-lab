"""Contributions à l'Expected Shortfall par allocation d'Euler.

Théorème d'Euler. Si F est positivement homogène de degré 1 (F(lambda w) = lambda F(w)) et dérivable, alors
        F(w) = sum_i w_i dF/dw_i .
La perte d'un portefeuille de poids w est L(w) = -N sum_i w_i r_i (r_i : rendement simple de la ligne i), linéaire donc
homogène de degré 1 en w ; l'ES (moyenne de L au-delà de la VaR) l'est aussi. Tasche (1999) / Artzner et al. :
        dES/dw_i = E[ -N r_i | L >= VaR ]   donc   contribution_i = w_i dES/dw_i = E[ perte de la ligne i | L >= VaR ].
La somme des contributions reproduit l'ES EXACTEMENT : c'est la seule décomposition additive cohérente avec la
diversification (une ligne qui perd peu quand le portefeuille perd beaucoup a une contribution faible, voire négative).

Estimateur : l'ES est la moyenne des k = n (1 - alpha) pertes les plus fortes (frontière fractionnaire pondérée).
Avec ces poids de scénario omega_s, ES = sum_s omega_s L_s et la contribution de la ligne i est sum_s omega_s l_{s,i}.
"""
import numpy as np


def poids_queue(L, alpha=0.975):
    """Poids omega_s (somme 1) portés par les n (1 - alpha) pertes les plus fortes ; ES = sum omega_s L_s."""
    n = L.size
    k = round(n * (1.0 - alpha), 9)                          # 1000 * (1 - 0,975) = 25,000000000000022 sinon : un 26e poids ~1e-16
    m = int(np.floor(k))
    ordre = np.argsort(-L)
    omega = np.zeros(n)
    omega[ordre[:m]] = 1.0 / k
    if m < n:
        omega[ordre[m]] = (k - m) / k
    return omega


def contributions_es(X, w, notionnel=1.0, alpha=0.975):
    """Contributions d'Euler à l'ES pour des scénarios de rendements LOG X (n, m) et des poids w (m,).

    Renvoie {'es', 'var', 'contributions' (m,), 'parts' (m,) (somme 1), 'omega'} ; perte de ligne i = -N w_i (e^{X_i} - 1).
    """
    pertes_lignes = -notionnel * np.expm1(X) * w            # (n, m) : perte de chaque ligne dans chaque scénario
    L = pertes_lignes.sum(axis=1)
    omega = poids_queue(L, alpha)
    es = float(omega @ L)
    contrib = omega @ pertes_lignes
    return {"es": es, "var": float(np.sort(L)[int(np.ceil(alpha * L.size)) - 1]), "contributions": contrib,
            "parts": contrib / es, "omega": omega, "pertes": L}


def es_homogene(X, w, notionnel=1.0, alpha=0.975):
    """ES seul (sans la décomposition) : sert aux différences finies du test d'homogénéité / d'Euler."""
    L = -notionnel * (np.expm1(X) @ w)
    return float(poids_queue(L, alpha) @ L)


def derivees_numeriques(X, w, notionnel=1.0, alpha=0.975, h_rel=1e-6):
    """w_i dES/dw_i par différences finies centrées (scénarios fixes) : doit retrouver les contributions d'Euler."""
    sortie = np.empty(w.size)
    for i in range(w.size):
        wp, wm = w.copy(), w.copy()
        h = h_rel * w[i]
        wp[i] += h
        wm[i] -= h
        sortie[i] = w[i] * (es_homogene(X, wp, notionnel, alpha) - es_homogene(X, wm, notionnel, alpha)) / (2 * h)
    return sortie
