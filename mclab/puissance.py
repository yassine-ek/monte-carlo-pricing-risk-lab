"""Puissance des backtests de VaR : Kupiec (couverture) et Christoffersen (indépendance), par Monte Carlo.

Question : un modèle qui dépasse sa VaR 99 % deux fois trop souvent (2 % au lieu de 1 %) sera-t-il détecté ?
Un test a une TAILLE (probabilité de rejeter à tort un bon modèle, visée 5 %) et une PUISSANCE (probabilité de rejeter
un mauvais modèle). Kupiec compte les exceptions x sur T jours : sous H0 x ~ Binomial(T, p0), LR_pof ~ chi2(1).
La statistique est discrète et x est RARE (T p0 = 2,5 exceptions par an) : la loi chi2 est mauvaise à T petit, la taille
réelle diffère de 5 % et la puissance est faible. On calcule donc la puissance de deux façons : par énumération
EXACTE de la loi binomiale, et par simulation (qui doit retrouver l'exacte à l'erreur-type près).
Christoffersen teste l'indépendance des exceptions (chaîne de Markov d'ordre 1) : il voit les exceptions GROUPÉES que
Kupiec, qui ne regarde que le nombre, ne peut pas voir.
"""
import numpy as np
from scipy import special, stats

from mclab.risque import ALPHA_VAR

P0 = 1.0 - ALPHA_VAR            # taux d'exceptions nominal : 1 %
CRIT_1 = float(stats.chi2.ppf(0.95, 1))
CRIT_2 = float(stats.chi2.ppf(0.95, 2))


def lr_pof(x, T, p=P0):
    """LR de Kupiec, vectorisé en x : identique à risque.kupiec_pof (qui traite un seul couple)."""
    x = np.asarray(x, float)
    ll0 = special.xlogy(T - x, 1 - p) + special.xlogy(x, p)
    ll1 = special.xlogy(T - x, 1 - x / T) + special.xlogy(x, x / T)
    return np.maximum(-2.0 * (ll0 - ll1), 0.0)


def rejet_kupiec_exact(T, p_vrai, p0=P0, crit=CRIT_1):
    """P(rejeter H0: taux = p0 | taux vrai = p_vrai) par énumération exacte de x ~ Binomial(T, p_vrai)."""
    x = np.arange(T + 1)
    return float(stats.binom.pmf(x, T, p_vrai)[lr_pof(x, T, p0) > crit].sum())


def rejet_kupiec_mc(T, p_vrai, n_sim, graine=0, p0=P0, crit=CRIT_1):
    """Même probabilité par simulation : (estimation, erreur-type binomiale)."""
    x = np.random.default_rng(graine).binomial(T, p_vrai, n_sim)
    rej = float(np.mean(lr_pof(x, T, p0) > crit))
    return rej, float(np.sqrt(rej * (1 - rej) / n_sim))


def courbe_puissance_kupiec(T_valeurs, p_vrai, p0=P0):
    """Puissance exacte pour chaque horizon T (en jours)."""
    return np.array([rejet_kupiec_exact(int(T), p_vrai, p0) for T in T_valeurs])


def horizon_puissance(T_valeurs, puissance, cible=0.8):
    """Plus petit T à partir duquel la puissance reste >= cible (la courbe est en dents de scie : discrétisation)."""
    puissance = np.asarray(puissance)
    ok = puissance >= cible
    for i in range(len(ok)):
        if ok[i:].all():
            return int(np.asarray(T_valeurs)[i])
    return None


def taille_echantillon_normale(p0, p1, alpha=0.05, puissance=0.8):
    """Formule de dimensionnement (test z bilatéral sur une proportion) : T = ((z_a s0 + z_b s1) / (p1 - p0))^2."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(puissance)
    return float(((za * np.sqrt(p0 * (1 - p0)) + zb * np.sqrt(p1 * (1 - p1))) / (p1 - p0)) ** 2)


def proba_zone_bale(p_vrai, T=250):
    """Probabilité qu'un modèle de taux vrai p_vrai tombe en zone verte / orange / rouge de Bâle sur T jours."""
    cdf = stats.binom.cdf
    return {"vert": float(cdf(4, T, p_vrai)), "orange": float(cdf(9, T, p_vrai) - cdf(4, T, p_vrai)),
            "rouge": float(1.0 - cdf(9, T, p_vrai))}


# ---------------------------------------------------------------- exceptions groupées et Christoffersen
def pi01_pour_taux(p, pi11):
    """P(exception demain | pas d'exception aujourd'hui) qui donne le taux STATIONNAIRE p avec persistance pi11."""
    return p * (1.0 - pi11) / (1.0 - p)


def simuler_exceptions(T, p, pi11, n_sim, rng):
    """Séries d'exceptions (n_sim, T) d'une chaîne de Markov de taux stationnaire p ; pi11 = p donne des tirages i.i.d."""
    pi01 = pi01_pour_taux(p, pi11)
    u = rng.random((n_sim, T))
    e = np.zeros((n_sim, T), bool)
    e[:, 0] = u[:, 0] < p
    for t in range(1, T):
        e[:, t] = u[:, t] < np.where(e[:, t - 1], pi11, pi01)
    return e


def christoffersen_lot(E, p=P0):
    """(LR_pof, LR_ind, LR_cc) pour chaque ligne de E (n_sim, T) ; mêmes formules que risque.christoffersen."""
    E = np.asarray(E, int)
    a, b = E[:, :-1], E[:, 1:]
    n00 = np.sum((a == 0) & (b == 0), axis=1)
    n01 = np.sum((a == 0) & (b == 1), axis=1)
    n10 = np.sum((a == 1) & (b == 0), axis=1)
    n11 = np.sum((a == 1) & (b == 1), axis=1)
    pi01 = n01 / np.maximum(n00 + n01, 1)
    pi11 = n11 / np.maximum(n10 + n11, 1)
    pi = (n01 + n11) / np.maximum(n00 + n01 + n10 + n11, 1)
    ll0 = special.xlogy(n00 + n10, 1 - pi) + special.xlogy(n01 + n11, pi)
    ll1 = (special.xlogy(n00, 1 - pi01) + special.xlogy(n01, pi01)
           + special.xlogy(n10, 1 - pi11) + special.xlogy(n11, pi11))
    lr_ind = np.maximum(-2.0 * (ll0 - ll1), 0.0)
    pof = lr_pof(E.sum(axis=1), E.shape[1], p)
    return pof, lr_ind, pof + lr_ind


def puissance_tests(T, p, pi11, n_sim, graine=0, p0=P0):
    """Taux de rejet à 5 % de Kupiec, Christoffersen-indépendance et Christoffersen-cc pour une série (p, pi11) vraie."""
    E = simuler_exceptions(T, p, pi11, n_sim, np.random.default_rng(graine))
    pof, ind, cc = christoffersen_lot(E, p0)
    f = lambda v, c: float(np.mean(v > c))
    taux = {"kupiec": f(pof, CRIT_1), "ind": f(ind, CRIT_1), "cc": f(cc, CRIT_2)}
    taux["se_max"] = float(np.sqrt(0.25 / n_sim))
    taux["exceptions_moyennes"] = float(E.sum(axis=1).mean())
    return taux
