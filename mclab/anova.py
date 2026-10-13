"""Dimension effective et constructions du brownien : pourquoi le pont brownien (et l'ACP) aident le QMC.

Idée centrale. Une suite de Sobol est beaucoup mieux répartie dans ses PREMIÈRES coordonnées que dans les
dernières. Le prix d'une option est une fonction f(Z_1..Z_d) de d normales ; la construction du brownien
(Z -> W) décide quelles coordonnées portent la variance :
    incrémentale : Z_j = j-ième incrément                  -> variance répartie sur toutes les coordonnées
    pont         : Z_1 = W_T, Z_2 = milieu, ...            -> les gros mouvements viennent en premier
    ACP          : Z_j multiplie le j-ième vecteur propre  -> maximise la variance expliquée par les k premières
Décomposition ANOVA (Hoeffding/Sobol) : f = sum_u f_u, Var f = sum_u sigma_u^2. La variance « expliquée par les
k premières coordonnées » est l'indice de Sobol FERMÉ de {1..k}, soit Var(E[f | Z_1..Z_k]) / Var f.
On l'estime par « pick-freeze » : Cov(f(Z), f(Z_1..Z_k, Z'_{k+1..d})) avec Z' une copie indépendante.
Lien avec Koksma-Hlawka : |I_N - I| <= V_HK(f) D*_N ; la discrépance D*_N d'une projection de dimension s est
~ (log N)^s / N, bien meilleure pour s petit : si f est (presque) fonction de peu de coordonnées de tête, l'erreur
suit la bonne projection (dimension effective de troncature faible) et non la mauvaise discrépance en dimension d.
"""
import math

import numpy as np

from mclab.qmc import normales_sobol, pente_loglog
from mclab.utils import Z95  # noqa: F401

CONSTRUCTIONS = ("incrementale", "pont", "acp")


def matrice_construction(d, T, construction):
    """Matrice A (d x d) telle que (W_{t_1}, ..., W_{t_d}) = A Z, avec t_i = i T / d. Toujours A A^T = Cov(W)."""
    dt = T / d
    if construction == "incrementale":
        return np.sqrt(dt) * np.tril(np.ones((d, d)))
    if construction == "pont":
        from mclab.qmc import pont_brownien
        return pont_brownien(np.eye(d), T)[:, 1:].T   # la construction est linéaire : on l'applique à la base canonique
    if construction == "acp":
        i = np.arange(1, d + 1)
        cov = dt * np.minimum.outer(i, i)
        val, vec = np.linalg.eigh(cov)
        ordre = np.argsort(val)[::-1]            # valeurs propres décroissantes : Z_1 porte le plus de variance
        return vec[:, ordre] * np.sqrt(val[ordre])
    raise ValueError(f"construction : {CONSTRUCTIONS}")


def payoff_asiatique(Z, A, S0=100.0, K=100.0, T=1.0, r=0.02, sigma=0.2, geometrique=False):
    """Payoff actualisé d'un call asiatique pour des normales Z (n, d), brownien W = Z A^T, GBM exact."""
    d = Z.shape[1]
    t = T * np.arange(1, d + 1) / d
    log_s = np.log(S0) + (r - 0.5 * sigma**2) * t + sigma * (Z @ A.T)
    moyenne = np.exp(log_s.mean(axis=1)) if geometrique else np.exp(log_s).mean(axis=1)
    return np.exp(-r * T) * np.maximum(moyenne - K, 0.0)


def _indice_jmn(f0, fm, n_lots=16):
    """Indice de Sobol (estimateur de Janon-Monod-Nodet, moyenne poolée) et son erreur-type par lots."""
    def indice(a, b):
        m = 0.5 * (a.mean() + b.mean())
        return (np.mean(a * b) - m**2) / (0.5 * np.mean(a**2 + b**2) - m**2)

    lots = [indice(a, b) for a, b in zip(np.array_split(f0, n_lots), np.array_split(fm, n_lots))]
    return float(indice(f0, fm)), float(np.std(lots, ddof=1) / np.sqrt(n_lots))


def variance_expliquee(f, d, ks, n=262_144, graine=0):
    """Indice fermé de {1..k} pour chaque k de `ks` (part de variance expliquée par les k premières coordonnées).

    f : fonction (n, d) -> (n,). Renvoie {k: (indice, erreur-type)}. k = d donne 1 exactement.
    """
    rng = np.random.default_rng(graine)
    Z, Zp = rng.standard_normal((n, d)), rng.standard_normal((n, d))
    f0 = f(Z)
    sortie = {}
    for k in ks:
        sortie[int(k)] = (1.0, 0.0) if k >= d else _indice_jmn(f0, f(np.concatenate([Z[:, :k], Zp[:, k:]], axis=1)))
    return sortie


def indices_premier_ordre(f, d, n=524_288, graine=0, degre=8, n_lots=8):
    """Indices de Sobol d'ordre 1 S_j = Var(E[f | Z_j]) / Var f, j = 1..d ; leur somme = part ADDITIVE de f.

    Méthode : projection sur les polynômes d'Hermite (probabilistes) He_k, orthogonaux pour la loi N(0,1) :
    E[f | Z_j] = mu + somme_k c_{jk} He_k(Z_j) / k! avec c_{jk} = E[f He_k(Z_j)], d'où
    Var(E[f | Z_j]) = somme_{k>=1} c_{jk}^2 / k!. Un SEUL jeu d'évaluations de f sert à toutes les coordonnées, et chaque
    coefficient est une simple moyenne : la somme des S_j est bien plus stable qu'avec d estimateurs pick-freeze
    corrélés (dont les erreurs, partagées par f(Z), s'additionnent). Troncature à `degre` : sous-estime très légèrement.
    Erreur-type par lots indépendants (n_lots).
    """
    rng = np.random.default_rng(graine)
    Z = rng.standard_normal((n, d))
    f0 = f(Z)
    fact = [float(math.factorial(k)) for k in range(degre + 1)]
    lots = np.array_split(np.arange(n), n_lots)
    # c[l, j, k-1] = moyenne sur le lot l de f He_k(Z_j) ; on n'a besoin que de ces moyennes (par lot, puis globales)
    c = np.empty((n_lots, d, degre))
    for j in range(d):
        z = Z[:, j]
        he_prec, he = np.ones(n), z
        for k in range(1, degre + 1):
            produit = f0 * he
            for l, idx in enumerate(lots):
                c[l, j, k - 1] = produit[idx].mean()
            he_prec, he = he, z * he - k * he_prec       # He_{k+1} = z He_k - k He_{k-1}
    poids = 1.0 / np.array(fact[1:])

    def indices(c_moy, var):
        return (c_moy**2 * poids).sum(axis=1) / var

    S = indices(c.mean(axis=0), f0.var())
    par_lot = np.array([indices(c[l], f0[idx].var()) for l, idx in enumerate(lots)])
    return S, par_lot.std(axis=0, ddof=1) / np.sqrt(n_lots)


def dimension_troncature(indices_fermes, seuil=0.99):
    """Plus petit k tel que l'indice fermé de {1..k} atteint `seuil` (dimension effective de troncature, Caflisch et al.)."""
    for k, (s, _) in sorted(indices_fermes.items()):
        if s >= seuil:
            return k
    return None


def exponentielle_lineaire(a):
    """f(Z) = exp(a . Z) et son indice fermé EXACT de {1..k} : (e^{|a_{1:k}|^2} - 1) / (e^{|a|^2} - 1). Sert de test."""
    a = np.asarray(a, float)

    def f(Z):
        return np.exp(Z @ a)

    def exact(k):
        return (np.exp(np.sum(a[:k] ** 2)) - 1.0) / (np.exp(np.sum(a**2)) - 1.0)

    return f, exact


def rmse_qmc_construction(construction, d, ms, repetitions=32, graine=0, S0=100.0, K=100.0, T=1.0, r=0.02,
                          sigma=0.2, geometrique=False):
    """RMSE du QMC randomisé vs N = 2^m pour une construction ; RMSE = écart-type sur `repetitions` brouillages.

    L'estimateur à brouillage est SANS biais : son erreur quadratique moyenne est donc exactement sa variance,
    estimée sur les répétitions (aucune valeur de référence n'est nécessaire).
    """
    A = matrice_construction(d, T, construction)
    rmse = []
    for m in ms:
        seeds = np.random.SeedSequence([graine, d, m]).spawn(repetitions)
        est = [payoff_asiatique(normales_sobol(m, d, s), A, S0, K, T, r, sigma, geometrique).mean() for s in seeds]
        rmse.append(float(np.std(est, ddof=1)))
    N = 2.0 ** np.asarray(list(ms))
    return {"N": N.tolist(), "rmse": rmse, "pente": pente_loglog(N, rmse)}


def rmse_mc(d, ms, repetitions=200, graine=0, **kw):
    """RMSE du MC ordinaire (référence ~ sigma_f / sqrt(N)), même estimation par répétitions."""
    A = matrice_construction(d, kw.get("T", 1.0), "incrementale")
    rmse = []
    for m in ms:
        rng = np.random.default_rng(np.random.SeedSequence([graine, d, m, 99]))
        est = [payoff_asiatique(rng.standard_normal((2**m, d)), A, **kw).mean() for _ in range(repetitions)]
        rmse.append(float(np.std(est, ddof=1)))
    N = 2.0 ** np.asarray(list(ms))
    return {"N": N.tolist(), "rmse": rmse, "pente": pente_loglog(N, rmse)}
