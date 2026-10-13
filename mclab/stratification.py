"""Stratification (allocation proportionnelle et de Neyman) et hypercube latin (LHS).

Stratification. On découpe (0,1) en H strates de probabilité p_h et on tire n_h points uniformes dans la strate h :
    I_strat = sum_h p_h * moyenne_h(f),     Var(I_strat) = sum_h p_h^2 sigma_h^2 / n_h.
  - allocation PROPORTIONNELLE (n_h = n p_h) : Var = (1/n) sum_h p_h sigma_h^2  <=  sigma^2 / n   (toujours),
    car la loi de la variance totale donne sigma^2 = sum_h p_h sigma_h^2 + sum_h p_h (mu_h - mu)^2 ;
  - allocation de NEYMAN (n_h ~ p_h sigma_h)  : Var = (1/n) (sum_h p_h sigma_h)^2  <=  allocation proportionnelle
    (Cauchy-Schwarz) — mais elle exige les sigma_h, estimés ici sur un échantillon pilote.
Hypercube latin. Chaque coordonnée est stratifiée en n tranches (une seule observation par tranche), les tranches
étant permutées indépendamment d'une coordonnée à l'autre. Théorème (Stein 1987) : la variance asymptotique est
(1/n) (sigma^2 - sum_j sigma_j^2) : le LHS élimine la partie ADDITIVE de f (sigma_j^2 = Var(E[f | U_j])) et rien d'autre.
"""
import numpy as np
from scipy import integrate
from scipy.special import ndtri
from scipy.stats import norm


# ---------------------------------------------------------------- stratification de U_1 (1 variable)
def tirer_strates(effectifs, rng):
    """u uniformes : effectifs[h] points dans la strate [h/H, (h+1)/H), renvoyés avec l'indice de strate."""
    effectifs = np.asarray(effectifs, int)
    H = effectifs.size
    h = np.repeat(np.arange(H), effectifs)
    return (h + rng.random(h.size)) / H, h


def allocation_neyman(sigmas, n, minimum=2):
    """n_h proportionnel à sigma_h (strates équiprobables), somme exactement n, au moins `minimum` par strate."""
    sigmas = np.asarray(sigmas, float)
    H = sigmas.size
    poids = sigmas / sigmas.sum() if sigmas.sum() > 0 else np.full(H, 1.0 / H)
    n_h = np.maximum(minimum, np.floor(poids * n)).astype(int)
    while n_h.sum() > n:                       # retire aux plus gros d'abord
        n_h[np.argmax(n_h)] -= 1
    while n_h.sum() < n:                       # ajoute là où le manque relatif est le plus fort
        n_h[np.argmax(poids * n - n_h)] += 1
    return n_h


def estimer_stratifie(f_de_u, H, n, allocation, rng, n_pilote=0, sigmas=None):
    """Estimateur stratifié de E[f(U)] avec H strates équiprobables.

    f_de_u(u, rng) -> valeurs (n,) ; rng sert aux autres sources d'aléa (ex. incréments du pont brownien).
    allocation : 'proportionnelle', 'neyman' (pilote de n_pilote points, répartis uniformément, puis allocation
    optimale sur les n - n_pilote points restants ; le pilote est INDÉPENDANT de l'estimateur final) ou 'neyman_oracle'
    (allocation optimale avec les vrais `sigmas` fournis : borne théorique, inaccessible en pratique).
    Renvoie (estimation, variance estimée de l'estimateur).
    """
    if allocation == "proportionnelle":
        n_h = np.full(H, n // H)
        n_h[: n - n_h.sum()] += 1
    elif allocation == "neyman":
        pil = np.full(H, max(2, n_pilote // H))
        u, h = tirer_strates(pil, rng)
        y = f_de_u(u, rng)
        sig = np.array([y[h == k].std(ddof=1) for k in range(H)])
        n_h = allocation_neyman(sig, n - pil.sum())
    elif allocation == "neyman_oracle":
        n_h = allocation_neyman(sigmas, n)
    else:
        raise ValueError("allocation : 'proportionnelle', 'neyman' ou 'neyman_oracle'")
    u, h = tirer_strates(n_h, rng)
    y = f_de_u(u, rng)
    moy = np.array([y[h == k].mean() for k in range(H)])
    var = np.array([y[h == k].var(ddof=1) for k in range(H)])
    return float(moy.mean()), float(np.sum(var / n_h) / H**2)


def moments_strates_numeriques(f_de_z, H):
    """(mu_h, sigma_h) EXACTS (quadrature) de f(Z), Z ~ N(0,1), sur H strates équiprobables en Phi(Z)."""
    bornes = np.clip(ndtri(np.linspace(0.0, 1.0, H + 1)), -12.0, 12.0)   # |z| > 12 : densité < 1e-32, négligeable
    mu, sig = np.empty(H), np.empty(H)
    for h in range(H):
        a, b = bornes[h], bornes[h + 1]
        m1 = integrate.quad(lambda z: f_de_z(z) * norm.pdf(z), a, b, limit=400, epsabs=1e-13, epsrel=1e-12)[0] * H
        m2 = integrate.quad(lambda z: f_de_z(z) ** 2 * norm.pdf(z), a, b, limit=400, epsabs=1e-13, epsrel=1e-12)[0] * H
        mu[h], sig[h] = m1, np.sqrt(max(m2 - m1**2, 0.0))
    return mu, sig


def moments_strates_empiriques(f_de_u, H, n, rng):
    """(mu_h, sigma_h) estimés sur n points répartis uniformément entre les strates (pilote de grande taille)."""
    u, h = tirer_strates(np.full(H, n // H), rng)
    y = f_de_u(u, rng)
    return (np.array([y[h == k].mean() for k in range(H)]), np.array([y[h == k].std(ddof=1) for k in range(H)]))


def variances_theoriques(mu, sig):
    """(sigma^2, sigma^2_prop, sigma^2_neyman) : variances PAR POINT des trois estimateurs (strates équiprobables)."""
    total = float(np.mean(sig**2) + np.mean((mu - mu.mean()) ** 2))
    return {"naif": total, "proportionnelle": float(np.mean(sig**2)), "neyman": float(np.mean(sig) ** 2)}


# ---------------------------------------------------------------- hypercube latin
def hypercube_latin(n, d, rng):
    """n points de (0,1)^d : chaque coordonnée a exactement un point par tranche [i/n, (i+1)/n)."""
    perm = np.argsort(rng.random((n, d)), axis=0)       # une permutation aléatoire par colonne
    return (perm + rng.random((n, d))) / n


def estimation_lhs(f_de_Z, n, d, rng):
    """Moyenne de f sur un LHS de n points, transformé en normales par Phi^{-1}."""
    return float(np.mean(f_de_Z(ndtri(hypercube_latin(n, d, rng)))))


def variance_estimateur(generateur, repetitions, graine=0):
    """Variance empirique d'un estimateur sur des répétitions indépendantes : generateur(rng) -> estimation."""
    rng = np.random.default_rng(graine)
    est = np.array([generateur(rng) for _ in range(repetitions)])
    return float(est.var(ddof=1)), float(est.mean())
