"""Schémas de discrétisation et ordres de convergence : Euler, Milstein (GBM, variance CIR de Heston).

Deux notions d'erreur, à ne pas confondre :
  - erreur FORTE   e_fort(dt) = E|X_T^dt - X_T|          : proximité des TRAJECTOIRES (même brownien) ;
  - erreur FAIBLE  e_faible(dt) = |E f(X_T^dt) - E f(X_T)| : proximité des LOIS (ce qui compte pour un prix).
Théorème (Kloeden-Platen) : Euler a un ordre fort 1/2 et faible 1 ; Milstein un ordre fort 1 et faible 1
(pour des coefficients réguliers). Milstein ajoute au pas d'Euler le terme 1/2 b b' (dW^2 - dt).

Astuce de mesure : pour estimer l'erreur faible SANS bruit parasite, on moyenne la DIFFÉRENCE
f(X^dt) - f(X) calculée sur les MÊMES browniens (nombres aléatoires communs) : sa variance est celle
de l'erreur forte (petite), pas celle du payoff.
"""
import numpy as np
from numba import njit, prange
from scipy.optimize import curve_fit

from mclab.accel import _GAMMA, _MULT1, _MULT2, _MULT_CHEMIN, _uniforme  # noqa: F401  (générateur SplitMix64)
from mclab.utils import ParamsHeston, resumer


# ---------------------------------------------------------------- utilitaires communs
def blocs(dW, n_pas):
    """Incréments d'un schéma à n_pas pas obtenus en SOMMANT les incréments fins (même brownien)."""
    n, m = dW.shape
    return dW.reshape(n, n_pas, m // n_pas).sum(axis=2)


def pente_loglog_ic(dts, valeurs_par_chemin, graine=0, n_blocs=100, n_boot=2000):
    """Pente de log(erreur) vs log(dt) et IC95 par bootstrap de BLOCS de chemins.

    valeurs_par_chemin : (n, K) — pour chaque chemin, la quantité dont la MOYENNE est l'erreur à chaque dt
    (|X^dt - X| pour l'erreur forte, f(X^dt) - f(X) pour l'erreur faible). Les erreurs aux différents dt
    sont corrélées (mêmes chemins) : on rééchantillonne donc des blocs de chemins entiers (toutes les
    colonnes ensemble) puis on refait la régression.
    """
    v = np.asarray(valeurs_par_chemin, float)
    n, K = v.shape
    n_blocs = min(n_blocs, n)
    moyennes_blocs = np.array([b.mean(axis=0) for b in np.array_split(v, n_blocs)])   # (n_blocs, K)
    x = np.log(np.asarray(dts, float))

    def pente(m):
        return np.polyfit(x, np.log(np.abs(m)), 1)[0]

    rng = np.random.default_rng(graine)
    idx = rng.integers(0, n_blocs, (n_boot, n_blocs))
    pentes = np.array([pente(moyennes_blocs[i].mean(axis=0)) for i in idx])
    erreur = v.mean(axis=0)
    se = np.sqrt(np.array([b.var(ddof=1) for b in v.T]) / n)
    return {"pente": float(pente(erreur)), "ic95": [float(np.percentile(pentes, 2.5)),
                                                    float(np.percentile(pentes, 97.5))],
            "erreurs": erreur.tolist(), "se": se.tolist()}


def ajuster_loi_puissance(dts, erreurs, se):
    """Ajuste erreur = c dt^p par moindres carrés pondérés (1/se) ; renvoie p, son erreur-type et son IC95.

    Utile quand l'erreur est mesurée avec bruit (parfois proche de 0) : un log-log ordinaire serait biaisé.
    """
    dts, erreurs, se = (np.asarray(a, float) for a in (dts, erreurs, se))
    p0 = np.polyfit(np.log(dts), np.log(np.maximum(np.abs(erreurs), 1e-12)), 1)
    (c, p), cov = curve_fit(lambda d, c, p: c * d**p, dts, erreurs, p0=[np.exp(p0[1]), p0[0]],
                            sigma=se, absolute_sigma=True, maxfev=20000)
    sp = float(np.sqrt(cov[1, 1]))
    return {"c": float(c), "p": float(p), "se_p": sp, "ic95": [float(p - 1.96 * sp), float(p + 1.96 * sp)]}


# ---------------------------------------------------------------- GBM : Euler, Milstein, solution exacte
def gbm_terminal(schema, S0, mu, sigma, T, dW):
    """S_T d'un schéma (dW : incréments du schéma, forme (n, n_pas)) ; schema in {'euler','milstein','exact'}.

    Euler    : S <- S (1 + mu dt + sigma dW)
    Milstein : S <- S (1 + mu dt + sigma dW + 1/2 sigma^2 (dW^2 - dt))     [b(S) = sigma S, b b' = sigma^2 S]
    exact    : S_T = S0 exp((mu - sigma^2/2) T + sigma W_T)               [W_T = somme des incréments]
    """
    if schema == "exact":
        return S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * dW.sum(axis=1))
    dt = T / dW.shape[1]
    facteur = 1.0 + mu * dt + sigma * dW
    if schema == "milstein":
        facteur = facteur + 0.5 * sigma**2 * (dW**2 - dt)
    elif schema != "euler":
        raise ValueError("schema : 'euler', 'milstein' ou 'exact'")
    return S0 * np.prod(facteur, axis=1)


def moment2_gbm_schema(schema, S0, mu, sigma, T, n_pas):
    """E[(S_T^dt)^2] EXACT (sans Monte Carlo) pour Euler/Milstein : le facteur par pas est un polynôme de Z.

    Euler : E[(m + s Z)^2] = m^2 + s^2 ; Milstein : E[(m + s Z + c (Z^2-1))^2] = m^2 + s^2 + 2 c^2
    avec m = 1 + mu dt, s = sigma sqrt(dt), c = sigma^2 dt / 2. Sert de test sans bruit de l'ordre faible.
    """
    dt = T / n_pas
    m, s, c = 1.0 + mu * dt, sigma * np.sqrt(dt), 0.5 * sigma**2 * dt
    par_pas = m**2 + s**2 + (2.0 * c**2 if schema == "milstein" else 0.0)
    return S0**2 * par_pas**n_pas


def moment2_gbm_exact(S0, mu, sigma, T):
    return S0**2 * np.exp((2.0 * mu + sigma**2) * T)


def erreurs_gbm(S0, mu, sigma, T, K, ks, n, graine=0, taille_lot=20_000):
    """Pour chaque schéma et chaque k (n_pas = 2^k), les erreurs PAR CHEMIN, mêmes browniens que la solution exacte.

    Renvoie {schema: {'fort': (n, len(ks)) = |S^dt - S|, 'fort_l2': idem au carré,
                       'faible_call': (n, len(ks)) = payoff(S^dt) - payoff(S),
                       'faible_call_cv': idem corrigé par la variable de contrôle S^dt - S (espérance exacte connue)}}.
    Les browniens sont tirés à la finesse 2^max(ks) puis agrégés par blocs pour les pas plus grossiers.
    """
    m = 2 ** max(ks)
    rng = np.random.default_rng(graine)
    sortie = {s: {"fort": [], "fort_l2": [], "faible_call": [], "diff": []} for s in ("euler", "milstein")}
    restant = n
    while restant > 0:
        b = min(taille_lot, restant)
        restant -= b
        dW = np.sqrt(T / m) * rng.standard_normal((b, m))
        exact = gbm_terminal("exact", S0, mu, sigma, T, dW)
        pay_exact = np.maximum(exact - K, 0.0)
        for schema in sortie:
            fort, faible, diff = [], [], []
            for k in ks:
                s_dt = gbm_terminal(schema, S0, mu, sigma, T, blocs(dW, 2**k))
                fort.append(np.abs(s_dt - exact))
                faible.append(np.maximum(s_dt - K, 0.0) - pay_exact)
                diff.append(s_dt - exact)
            fort = np.column_stack(fort)
            sortie[schema]["fort"].append(fort)
            sortie[schema]["fort_l2"].append(fort**2)
            sortie[schema]["faible_call"].append(np.column_stack(faible))
            sortie[schema]["diff"].append(np.column_stack(diff))
    sortie = {s: {c: np.concatenate(v) for c, v in d.items()} for s, d in sortie.items()}
    # variable de contrôle : X = S^dt - S a une espérance CONNUE (Euler et Milstein ont E[S^dt_T] = S0 (1 + mu dt)^N)
    for schema, d in sortie.items():
        m_x = np.array([S0 * (1.0 + mu * T / 2**k) ** (2**k) - S0 * np.exp(mu * T) for k in ks])
        X, Y = d["diff"], d["faible_call"]
        beta = ((X - X.mean(0)) * (Y - Y.mean(0))).mean(0) / X.var(0)
        d["faible_call_cv"] = Y - beta * (X - m_x)
    return sortie


# ---------------------------------------------------------------- CIR : Euler et Milstein « full truncation »
def cir_terminal(schema, v0, kappa, theta, xi, T, dW):
    """v_T^+ du schéma (full truncation : v^+ = max(v, 0) dans le drift ET la diffusion).

    Euler    : v <- v + kappa (theta - v^+) dt + xi sqrt(v^+) dW
    Milstein : + 1/4 xi^2 (dW^2 - dt)     [b(v) = xi sqrt(v), b b' = xi^2 / 2 => terme 1/2 b b' (dW^2 - dt)]
    Renvoie aussi la part des pas où v < 0 (truncation active), pour l'étude de la condition de Feller.
    """
    n, n_pas = dW.shape
    dt = T / n_pas
    v = np.broadcast_to(np.asarray(v0, float), (n,)).copy()      # v0 scalaire ou tableau (départ stationnaire)
    n_neg = 0
    for j in range(n_pas):
        vp = np.maximum(v, 0.0)
        n_neg += int(np.count_nonzero(v < 0.0))
        v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp) * dW[:, j]
        if schema == "milstein":
            v = v + 0.25 * xi**2 * (dW[:, j] ** 2 - dt)
        elif schema != "euler":
            raise ValueError("schema : 'euler' ou 'milstein'")
    return np.maximum(v, 0.0), n_neg / (n * n_pas)


def erreurs_forte_cir(v0, kappa, theta, xi, T, ks, k_ref, n, graine=0, taille_lot=10_000):
    """Erreur forte E|v_T^dt - v_T^ref| pour Euler et Milstein, référence = Milstein à 2^k_ref pas, MÊMES browniens.

    Pas de solution pathwise explicite pour le CIR : on prend un schéma très fin comme « vérité ». Cela
    sous-estime légèrement l'erreur grossière (la référence a elle-même une erreur) tant que dt >> dt_ref.
    """
    assert k_ref > max(ks)
    m = 2**k_ref
    rng = np.random.default_rng(graine)
    sortie = {s: [] for s in ("euler", "milstein")}
    restant = n
    while restant > 0:
        b = min(taille_lot, restant)
        restant -= b
        dW = np.sqrt(T / m) * rng.standard_normal((b, m))
        ref, _ = cir_terminal("milstein", v0, kappa, theta, xi, T, dW)
        for schema in sortie:
            cols = [np.abs(cir_terminal(schema, v0, kappa, theta, xi, T, blocs(dW, 2**k))[0] - ref)
                    for k in ks]
            sortie[schema].append(np.column_stack(cols))
    return {s: np.concatenate(v) for s, v in sortie.items()}


# ---------------------------------------------------------------- Heston : Euler full-truncation en numba
@njit(parallel=True, cache=True)
def _noyau_heston_euler_ft(S0, r, q, v0, kappa, theta, xi, rho, T, n_pas, n_chemins, graine, sortie):
    dt = T / n_pas
    racine = np.sqrt(dt)
    c = np.sqrt(1.0 - rho**2)
    for i in prange(n_chemins):
        etat = np.uint64(graine) ^ (np.uint64(i) * _MULT_CHEMIN)
        log_s = np.log(S0)
        v = v0
        for _ in range(n_pas):
            etat, u1 = _uniforme(etat)
            etat, u2 = _uniforme(etat)
            rayon = np.sqrt(-2.0 * np.log(u1))
            z1 = rayon * np.cos(2.0 * np.pi * u2)
            z2 = rayon * np.sin(2.0 * np.pi * u2)
            vp = max(v, 0.0)
            sv = np.sqrt(vp)
            log_s += (r - q - 0.5 * vp) * dt + sv * racine * (rho * z1 + c * z2)
            v = v + kappa * (theta - vp) * dt + xi * sv * racine * z1
        sortie[i] = np.exp(log_s)


def heston_euler_ft_numba(p: ParamsHeston, T, n_pas, n_chemins, graine=0):
    """S_T de n_chemins trajectoires Heston Euler full-truncation (même flux SplitMix64 que le QE numba)."""
    sortie = np.empty(n_chemins)
    _noyau_heston_euler_ft(p.S0, p.r, p.q, p.v0, p.kappa, p.theta, p.xi, p.rho, T, n_pas, n_chemins,
                           graine, sortie)
    return sortie


def biais_faible_heston(p: ParamsHeston, K, T, schema, n_pas, n_chemins, graine, prix_exact, lot=10_000_000):
    """Biais faible du prix d'un call Heston : (prix MC - prix semi-fermé) avec son erreur-type MC."""
    from mclab.accel import heston_qe_numba
    gen = heston_euler_ft_numba if schema == "euler_ft" else heston_qe_numba
    somme, somme2, restant, i = 0.0, 0.0, n_chemins, 0
    while restant > 0:
        b = min(lot, restant)
        restant -= b
        y = np.exp(-p.r * T) * np.maximum(gen(p, T, n_pas, b, graine + 7919 * i) - K, 0.0)
        somme, somme2, i = somme + y.sum(), somme2 + (y**2).sum(), i + 1
    moy = somme / n_chemins
    se = float(np.sqrt((somme2 / n_chemins - moy**2) / n_chemins))
    return {"prix": float(moy), "biais": float(moy - prix_exact), "se": se}


# ---------------------------------------------------------------- Condition de Feller
def fraction_pas_tronques(kappa, theta, xi, T, n_pas, n_chemins, graine=0):
    """Part des pas d'Euler full-truncation où v < 0 (la troncature agit), départ dans la loi STATIONNAIRE.

    La loi stationnaire du CIR est Gamma(forme = 2 kappa theta / xi^2, échelle = xi^2 / (2 kappa)) :
    démarrer de là évite un transitoire qui fausserait la fréquence de passage près de 0.
    """
    rng = np.random.default_rng(graine)
    forme = 2.0 * kappa * theta / xi**2
    v = rng.gamma(forme, xi**2 / (2.0 * kappa), n_chemins)
    dt = T / n_pas
    n_neg = 0
    for _ in range(n_pas):
        vp = np.maximum(v, 0.0)
        n_neg += int(np.count_nonzero(v < 0.0))
        v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * rng.standard_normal(n_chemins)
    return n_neg / (n_chemins * n_pas)
