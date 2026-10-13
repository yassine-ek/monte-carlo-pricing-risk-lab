"""Risque de marche : VaR 99 % / ES 97,5 % a 1 jour, cinq methodes, backtests reglementaires.

Tout est strictement causal : la prevision du jour t n'utilise que les rendements
des jours t-fenetre ... t-1 (jamais le rendement du jour t, jamais le futur).

Convention de signe : on travaille sur la PERTE L = -P&L (positive = perte).
    VaR_a  = quantile a du la perte      ES_a = esperance de la perte au-dela de VaR_a

Methodes (cle -> description) :
    hs        simulation historique sur la fenetre
    normale   parametrique normale, covariance EWMA RiskMetrics (lambda = 0,94)
    mc_t      Monte Carlo Student-t multivarie (covariance EWMA, ddl estime sur la fenetre)
    fhs       Filtered Historical Simulation (GARCH(1,1) par actif + bootstrap conjoint des residus)
    copule_t  Monte Carlo copule de Student + marges GARCH(1,1)-t
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from scipy.signal import lfilter

# --------------------------------------------------------------------------- #
# Parametres declares AVANT tout calcul
# --------------------------------------------------------------------------- #
ACTIFS_MORTS = ("WTIUSD", "ETXEUR", "XAUAUD", "XAUCHF", "XAUEUR", "XAUGBP", "UDXUSD", "GRXEUR")

# Poids fixes (rebalancement quotidien implicite), notionnel 10 M EUR, tous longs.
PORTEFEUILLE = {
    "EURUSD": 0.20,
    "USDJPY": 0.15,
    "GBPUSD": 0.15,
    "XAUUSD": 0.15,
    "BCOUSD": 0.10,
    "SPXUSD": 0.15,
    "NSXUSD": 0.10,
}
NOTIONNEL = 10_000_000.0

ALPHA_VAR = 0.99
ALPHA_ES = 0.975
FENETRE = 500
LAMBDA = 0.94
N_SIM = 10_000
REFIT = 20
K_QUEUE = 200          # nb de quantiles stockes dans la queue [97,5 % ; 100 %]
SEED = 20260101
GRILLE_NU_COPULE = (3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100)

METHODES = ("hs", "normale", "mc_t", "fhs", "copule_t")
LIBELLES = {
    "hs": "Simulation historique",
    "normale": "Normale EWMA",
    "mc_t": "MC Student-t",
    "fhs": "FHS (GARCH)",
    "copule_t": "Copule t + GARCH",
}


class Prevision(NamedTuple):
    var99: float
    var975: float
    es975: float
    queue: np.ndarray   # K_QUEUE quantiles de la perte, niveaux equirepartis dans (97,5 % ; 100 %)


# --------------------------------------------------------------------------- #
# Donnees
# --------------------------------------------------------------------------- #
def trouver_dossier_donnees() -> Path:
    """Dossier des CSV 1h (`<ACTIF>_1h.csv`) : variable d'environnement UNIVERS_DIR, sinon `data/processed/univers`.

    Le dépôt public ne contient AUCUNE donnée de marché : sans dossier, le volet risque et ses tests dépendants des
    données sont ignorés (voir tests/conftest.py). MCLAB_DATA_DIR reste accepté comme ancien nom de UNIVERS_DIR.
    """
    candidats = []
    for nom_variable in ("UNIVERS_DIR", "MCLAB_DATA_DIR"):
        if os.environ.get(nom_variable):
            candidats.append(Path(os.environ[nom_variable]))
    for parent in Path(__file__).resolve().parents:
        candidats.append(parent / "data" / "processed" / "univers")
    for c in candidats:
        if (c / "EURUSD_1h.csv").is_file():
            return c
    raise FileNotFoundError("Aucun dossier de donnees 1h trouve : definir la variable UNIVERS_DIR (dossier des <ACTIF>_1h.csv, colonnes "
                            "datetime,close ; voir « Obtenir les donnees » dans le README, source : https://www.histdata.com).")


def cloture_quotidienne(actif: str, dossier: Path | None = None, min_barres: int = 12,
                        decalage_h: int = 7) -> pd.Series:
    """Cloture quotidienne = derniere barre 1h de chaque jour de trading.

    Les horodatages sont en heure de New York : le jour de trading va de 17h a 17h.
    On decale de +7h avant de prendre la date, pour que la reouverture du dimanche
    soir tombe sur le lundi. Les jours a moins de `min_barres` barres (stubs de
    dimanche, jours feries) sont ecartes : le rendement les enjambe.
    """
    if actif in ACTIFS_MORTS:
        raise ValueError(f"{actif} est un actif mort : exclu de l'etude.")
    dossier = Path(dossier) if dossier is not None else trouver_dossier_donnees()
    fichier = dossier / f"{actif}_1h.csv"
    if not fichier.is_file():
        raise FileNotFoundError(f"{fichier} introuvable : le dossier UNIVERS_DIR doit contenir {actif}_1h.csv "
                                "(voir « Obtenir les donnees » dans le README).")
    df = pd.read_csv(fichier, usecols=["datetime", "close"], parse_dates=["datetime"])
    s = df.set_index("datetime")["close"].astype(float)
    s = s[s > 0].sort_index()
    jour = (s.index + pd.Timedelta(hours=decalage_h)).normalize()
    g = s.groupby(jour)
    cl, n = g.last(), g.size()
    return cl[n >= min_barres]


def rendements_log(portefeuille: dict | None = None, dossier: Path | None = None,
                   debut: str = "2010-01-01") -> pd.DataFrame:
    """Rendements log quotidiens sur les dates communes a tous les actifs du portefeuille."""
    portefeuille = PORTEFEUILLE if portefeuille is None else portefeuille
    cl = pd.concat({a: cloture_quotidienne(a, dossier) for a in portefeuille}, axis=1)
    cl = cl.loc[debut:].dropna()
    return np.log(cl).diff().dropna()


def pertes(R: np.ndarray, w: np.ndarray, notionnel: float = NOTIONNEL) -> np.ndarray:
    """Perte (en EUR) pour des rendements log R (reevaluation complete : exp(r)-1)."""
    return -notionnel * (np.expm1(R) @ w)


# --------------------------------------------------------------------------- #
# Statistiques de perte a partir d'un echantillon
# --------------------------------------------------------------------------- #
def niveaux_queue(k: int = K_QUEUE) -> np.ndarray:
    return ALPHA_ES + (1 - ALPHA_ES) * (np.arange(k) + 0.5) / k


def stats_echantillon(L: np.ndarray, k: int = K_QUEUE) -> Prevision:
    """VaR 99 %, VaR 97,5 %, ES 97,5 % et queue d'un echantillon de pertes.

    Quantile a interpolation lineaire (comme numpy). L'ES est la moyenne de la fonction
    quantile sur [97,5 % ; 100 %] (regle du point milieu, k points).
    """
    n = L.size
    m = min(n, int(np.ceil((1 - ALPHA_ES) * n)) + 3)
    top = np.sort(np.partition(L, n - m)[n - m:])

    def q(p):
        return np.interp(np.asarray(p) * (n - 1) - (n - m), np.arange(m), top)

    queue = q(niveaux_queue(k))
    return Prevision(float(q(ALPHA_VAR)), float(q(ALPHA_ES)), float(queue.mean()), queue)


# --------------------------------------------------------------------------- #
# (a) simulation historique
# --------------------------------------------------------------------------- #
def prevision_hs(Rwin: np.ndarray, w: np.ndarray, notionnel: float = NOTIONNEL) -> Prevision:
    return stats_echantillon(pertes(Rwin, w, notionnel))


# --------------------------------------------------------------------------- #
# (b) parametrique normale, EWMA RiskMetrics (moyenne nulle)
# --------------------------------------------------------------------------- #
def cov_ewma(Rwin: np.ndarray, lam: float = LAMBDA) -> np.ndarray:
    """Covariance EWMA a moyenne nulle, poids lam^(age) normalises sur la fenetre.

    Equivaut a la recursion S_t = lam S_{t-1} + (1-lam) r_{t-1} r_{t-1}' (sans amorce
    arbitraire). lam = 1 redonne la covariance a poids egaux.
    """
    T = len(Rwin)
    p = lam ** np.arange(T - 1, -1, -1)
    p = p / p.sum()
    return (Rwin * p[:, None]).T @ Rwin


def var_normale(sigma_p: float, notionnel: float = NOTIONNEL, alpha: float = ALPHA_VAR) -> float:
    """Formule fermee : VaR = N * z_alpha * sigma_p."""
    return notionnel * sigma_p * stats.norm.ppf(alpha)


def es_normale(sigma_p: float, notionnel: float = NOTIONNEL, alpha: float = ALPHA_ES) -> float:
    """Formule fermee : ES = N * sigma_p * phi(z_alpha) / (1 - alpha)."""
    return notionnel * sigma_p * stats.norm.pdf(stats.norm.ppf(alpha)) / (1 - alpha)


def prevision_normale(Rwin: np.ndarray, w: np.ndarray, notionnel: float = NOTIONNEL,
                      lam: float = LAMBDA, k: int = K_QUEUE) -> Prevision:
    sigma_p = float(np.sqrt(w @ cov_ewma(Rwin, lam) @ w))
    queue = notionnel * sigma_p * stats.norm.ppf(niveaux_queue(k))
    return Prevision(var_normale(sigma_p, notionnel), var_normale(sigma_p, notionnel, ALPHA_ES),
                     es_normale(sigma_p, notionnel), queue)


# --------------------------------------------------------------------------- #
# (c) Monte Carlo Student-t multivarie
# --------------------------------------------------------------------------- #
def nu_student_mle(z: np.ndarray, bornes: tuple[float, float] = (2.1, 100.0)) -> float:
    """MLE du ddl d'une Student de variance 1 (echelle sqrt((nu-2)/nu)) sur z."""
    z2 = z ** 2

    def nll(nu):
        ll = (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2)
              - 0.5 * np.log(np.pi * (nu - 2)) - (nu + 1) / 2 * np.log1p(z2 / (nu - 2)))
        return -ll.sum()

    return float(optimize.minimize_scalar(nll, bounds=bornes, method="bounded",
                                          options={"xatol": 1e-3}).x)


def nu_portefeuille(Rwin: np.ndarray, w: np.ndarray, lam: float = LAMBDA, burn: int = 50) -> float:
    """Ddl estime sur les rendements du portefeuille standardises par sa vol EWMA passee."""
    rp = Rwin @ w
    s2 = np.empty(rp.size)
    s2[0] = np.mean(rp[:burn] ** 2)
    s2[1:], _ = lfilter([1.0], [1.0, -lam], (1 - lam) * rp[:-1] ** 2, zi=[lam * s2[0]])
    return nu_student_mle((rp / np.sqrt(s2))[burn:])


def simuler_t_multivarie(sigma: np.ndarray, nu: float, n_sim: int, rng) -> np.ndarray:
    """Tirages Student-t multivariee de covariance `sigma` (variance finie, nu > 2)."""
    n = sigma.shape[0]
    chol = np.linalg.cholesky(sigma + 1e-14 * np.eye(n))
    z = rng.standard_normal((n_sim, n)) @ chol.T
    return z * np.sqrt((nu - 2) / rng.chisquare(nu, n_sim))[:, None]


def prevision_mc_t(Rwin: np.ndarray, w: np.ndarray, rng, n_sim: int = N_SIM,
                   notionnel: float = NOTIONNEL, lam: float = LAMBDA) -> Prevision:
    sigma = cov_ewma(Rwin, lam)
    nu = nu_portefeuille(Rwin, w, lam)
    X = simuler_t_multivarie(sigma, nu, n_sim, rng)
    return stats_echantillon(pertes(X, w, notionnel))


# --------------------------------------------------------------------------- #
# GARCH(1,1) par actif : ajustement (package arch) + filtre de variance maison
# --------------------------------------------------------------------------- #
PARAMS_DEFAUT = (0.02, 0.08, 0.90, 8.0)   # omega (en %^2, mis a l'echelle ensuite), alpha, beta, nu


def ajuster_garch(Rwin: np.ndarray, precedents: np.ndarray | None = None):
    """GARCH(1,1) a innovations Student, moyenne nulle, un par colonne de Rwin.

    Retourne (params (n,4) = omega, alpha, beta, nu ; nb d'echecs). Les rendements sont
    passes a `arch` en % (stabilite numerique). En cas d'echec (non-convergence, params
    invalides), on garde les parametres precedents de l'actif, sinon un jeu par defaut.
    """
    from arch import arch_model

    n = Rwin.shape[1]
    y = 100.0 * Rwin
    params = np.empty((n, 4))
    echecs = 0
    for i in range(n):
        ok = False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = arch_model(y[:, i], mean="Zero", vol="GARCH", p=1, q=1, dist="t",
                                 rescale=False).fit(disp="off", show_warning=False)
            p = res.params
            om, al, be, nu = (float(p["omega"]), float(p["alpha[1]"]),
                              float(p["beta[1]"]), float(p["nu"]))
            ok = (res.convergence_flag == 0 and om > 0 and al >= 0 and be >= 0
                  and al + be <= 1.0 + 1e-6 and np.isfinite(nu))   # alpha+beta = 1 (IGARCH) accepte
        except Exception:
            ok = False
        if ok:
            params[i] = (om, al, be, np.clip(nu, 2.5, 100.0))
        else:
            echecs += 1
            if precedents is not None:
                params[i] = precedents[i]
            else:
                v = float(np.var(y[:, i]))
                params[i] = (0.02 * v, PARAMS_DEFAUT[1], PARAMS_DEFAUT[2], PARAMS_DEFAUT[3])
    return params, echecs


def filtre_variance(y: np.ndarray, omega: float, alpha: float, beta: float,
                    n_init: int = 50) -> np.ndarray:
    """Variances conditionnelles s2[0..T] d'un GARCH(1,1) sur la serie y[0..T-1].

    s2[k] ne depend que de y[0..k-1] ; s2[T] est la variance prevue pour le jour suivant.
    Amorce : moyenne des n_init premiers y^2 de la fenetre.
    """
    init = float(np.mean(y[:n_init] ** 2))
    x = omega + alpha * y ** 2
    reste, _ = lfilter([1.0], [1.0, -beta], x, zi=[beta * init])
    return np.concatenate(([init], reste))


def etat_garch(Rwin: np.ndarray, params: np.ndarray):
    """(sigma_prevu (n,), residus standardises z (T,n)) avec params fixes, donnees de la fenetre."""
    T, n = Rwin.shape
    y = 100.0 * Rwin
    s2 = np.column_stack([filtre_variance(y[:, i], *params[i, :3]) for i in range(n)])
    return np.sqrt(s2[T]) / 100.0, y / np.sqrt(s2[:T])


# --------------------------------------------------------------------------- #
# (d) Filtered Historical Simulation
# --------------------------------------------------------------------------- #
def prevision_fhs(sigma_prevu: np.ndarray, z: np.ndarray, w: np.ndarray, rng,
                  n_sim: int = N_SIM, notionnel: float = NOTIONNEL) -> Prevision:
    """Bootstrap des residus standardises JOINTS (lignes entieres : la dependance est conservee),
    remis a l'echelle par la vol GARCH prevue pour le jour t."""
    X = z[rng.integers(0, z.shape[0], n_sim)] * sigma_prevu
    return stats_echantillon(pertes(X, w, notionnel))


# --------------------------------------------------------------------------- #
# (e) copule de Student + marges GARCH
# --------------------------------------------------------------------------- #
def _corr_psd(R: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh((R + R.T) / 2)
    R2 = (vecs * np.clip(vals, 1e-8, None)) @ vecs.T
    d = np.sqrt(np.diag(R2))
    return R2 / np.outer(d, d)


def correlation_copule(z: np.ndarray) -> np.ndarray:
    """Correlation de la copule t via le rho de Spearman : R = 2 sin(pi rho / 6)."""
    rangs = stats.rankdata(z, axis=0)
    return _corr_psd(2 * np.sin(np.pi / 6 * np.corrcoef(rangs, rowvar=False)))


def nu_copule_mle(z: np.ndarray, R: np.ndarray, grille=GRILLE_NU_COPULE) -> float:
    """Ddl de la copule t : maximum de la log-vraisemblance de la copule sur les
    pseudo-observations (rangs / (T+1)), sur une grille grossiere de ddl."""
    T, n = z.shape
    U = stats.rankdata(z, axis=0) / (T + 1)
    Rinv = np.linalg.inv(R)
    logdet = np.linalg.slogdet(R)[1]
    meilleur, nu_opt = -np.inf, grille[0]
    for nu in grille:
        x = special.stdtrit(nu, U)
        q = ((x @ Rinv) * x).sum(axis=1)
        ll_joint = (special.gammaln((nu + n) / 2) - special.gammaln(nu / 2)
                    - n / 2 * np.log(nu * np.pi) - 0.5 * logdet - (nu + n) / 2 * np.log1p(q / nu))
        ll_marg = (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2)
                   - 0.5 * np.log(nu * np.pi) - (nu + 1) / 2 * np.log1p(x ** 2 / nu)).sum(axis=1)
        ll = float(np.sum(ll_joint - ll_marg))
        if ll > meilleur:
            meilleur, nu_opt = ll, nu
    return float(nu_opt)


def estimer_copule_t(z: np.ndarray, grille=GRILLE_NU_COPULE):
    """(R, nu_copule) estimes sur les residus standardises z."""
    R = correlation_copule(z)
    return R, nu_copule_mle(z, R, grille)


# Grille (negative, dense pres de 0) pour interpoler x -> quantile de la marge ; stdtrit exact
# coute ~2,4 us par point, soit 0,2 s par jour pour 10 000 x 7 tirages.
_XG = np.sinh(np.linspace(-np.arcsinh(1e4), 0.0, 801))


def transformee_marge(x: np.ndarray, nu_c: float, nu_marge: float) -> np.ndarray:
    """Marge de variance 1 : F_marge^{-1}( F_t(nu_c)(x) ), par interpolation sur la grille _XG.

    Symetrique en x ; la grille couvre |x| <= 1e4 (au-dela : valeur de bord, effet negligeable).
    """
    u = np.clip(special.stdtr(nu_c, _XG), 1e-300, 0.5)
    h = special.stdtrit(nu_marge, u) * np.sqrt((nu_marge - 2) / nu_marge)
    h[-1] = 0.0                                    # x = 0 -> mediane exacte
    y = np.interp(-np.abs(x), _XG, h)
    return np.where(x < 0, y, -y)


def simuler_copule_t(R: np.ndarray, nu_c: float, nu_marges: np.ndarray, sigma_prevu: np.ndarray,
                     n_sim: int, rng) -> np.ndarray:
    """Rendements simules : dependance = copule t(R, nu_c), marges = Student(nu_i) de variance 1
    multipliee par la vol GARCH prevue."""
    n = R.shape[0]
    g = rng.standard_normal((n_sim, n)) @ np.linalg.cholesky(R).T
    x = g / np.sqrt(rng.chisquare(nu_c, n_sim) / nu_c)[:, None]
    e = np.column_stack([transformee_marge(x[:, i], nu_c, nu_marges[i]) for i in range(n)])
    return e * sigma_prevu


def prevision_copule(sigma_prevu: np.ndarray, z: np.ndarray, nu_marges: np.ndarray,
                     w: np.ndarray, rng, n_sim: int = N_SIM, notionnel: float = NOTIONNEL,
                     nu_c: float | None = None) -> Prevision:
    """R est reestimee a chaque appel ; nu_c (ddl de la copule) peut etre fourni (reestime
    seulement aux dates de refit pour economiser du calcul), sinon estime ici."""
    R = correlation_copule(z)
    if nu_c is None:
        nu_c = nu_copule_mle(z, R)
    X = simuler_copule_t(R, nu_c, nu_marges, sigma_prevu, n_sim, rng)
    return stats_echantillon(pertes(X, w, notionnel))


# --------------------------------------------------------------------------- #
# Moteur glissant causal
# --------------------------------------------------------------------------- #
def backtest_roulant(R: np.ndarray, w: np.ndarray, notionnel: float = NOTIONNEL,
                     fenetre: int = FENETRE, n_sim: int = N_SIM, refit: int = REFIT,
                     lam: float = LAMBDA, seed: int = SEED, debut: int | None = None,
                     fin: int | None = None, methodes=METHODES, k: int = K_QUEUE,
                     verbose: bool = False) -> dict:
    """Previsions de VaR/ES pour les jours t = debut ... fin-1 (indices de R).

    La prevision du jour t n'utilise que R[t-fenetre : t]. Le GARCH est reajuste tous les
    `refit` jours de prevision (sur la fenetre passee uniquement), de meme que le ddl de la
    copule t (sa matrice de correlation, elle, est reestimee chaque jour) ; la graine Monte Carlo
    est propre a chaque jour (default_rng([seed, t])) et partagee par les methodes MC.
    """
    R = np.asarray(R, float)
    w = np.asarray(w, float)
    debut = fenetre if debut is None else debut
    fin = len(R) if fin is None else fin
    if debut < fenetre:
        raise ValueError("debut doit etre >= fenetre")
    jours = np.arange(debut, fin)
    nj = jours.size
    sortie = {m: {"var99": np.full(nj, np.nan), "var975": np.full(nj, np.nan),
                  "es975": np.full(nj, np.nan), "queue": np.full((nj, k), np.nan)}
              for m in methodes}
    garch = ("fhs" in methodes) or ("copule_t" in methodes)
    params, nu_c, echecs_garch, n_refits = None, None, 0, 0

    def ranger(m, i, p: Prevision):
        s = sortie[m]
        s["var99"][i], s["var975"][i], s["es975"][i], s["queue"][i] = p.var99, p.var975, p.es975, p.queue

    for i, t in enumerate(jours):
        Rwin = R[t - fenetre:t]
        if garch:
            if i % refit == 0:
                params, e = ajuster_garch(Rwin, params)
                echecs_garch += e
                n_refits += 1
            sigma_prevu, z = etat_garch(Rwin, params)
            if "copule_t" in methodes and i % refit == 0:
                nu_c = nu_copule_mle(z, correlation_copule(z))
        if "hs" in methodes:
            ranger("hs", i, prevision_hs(Rwin, w, notionnel))
        if "normale" in methodes:
            ranger("normale", i, prevision_normale(Rwin, w, notionnel, lam, k))
        if "mc_t" in methodes:
            ranger("mc_t", i, prevision_mc_t(Rwin, w, np.random.default_rng([seed, int(t)]),
                                             n_sim, notionnel, lam))
        if "fhs" in methodes:
            ranger("fhs", i, prevision_fhs(sigma_prevu, z, w, np.random.default_rng([seed, int(t)]),
                                           n_sim, notionnel))
        if "copule_t" in methodes:
            ranger("copule_t", i, prevision_copule(sigma_prevu, z, params[:, 3], w,
                                                   np.random.default_rng([seed, int(t)]),
                                                   n_sim, notionnel, nu_c))
        if verbose and (i + 1) % 250 == 0:
            print(f"  {i + 1}/{nj} jours", flush=True)
    return {"jours": jours, "methodes": sortie, "echecs_garch": echecs_garch, "n_refits": n_refits}


# --------------------------------------------------------------------------- #
# Backtests
# --------------------------------------------------------------------------- #
def kupiec_pof(x: int, T: int, p: float = 1 - ALPHA_VAR):
    """Test de couverture inconditionnelle de Kupiec (POF). Retourne (LR, p-value), chi2(1)."""
    x, T = int(x), int(T)
    ll0 = special.xlogy(T - x, 1 - p) + special.xlogy(x, p)
    ll1 = special.xlogy(T - x, 1 - x / T) + special.xlogy(x, x / T)
    lr = max(-2.0 * (ll0 - ll1), 0.0)
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen(exc: np.ndarray, p: float = 1 - ALPHA_VAR) -> dict:
    """Independance (chaine de Markov d'ordre 1) et couverture conditionnelle de Christoffersen.

    LR_ind ~ chi2(1), LR_cc = LR_pof + LR_ind ~ chi2(2). Sans exception, l'independance
    n'est pas testable : LR_ind = 0.
    """
    e = np.asarray(exc, int)
    a, b = e[:-1], e[1:]
    n00 = int(np.sum((a == 0) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n11 = int(np.sum((a == 1) & (b == 1)))
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll0 = special.xlogy(n00 + n10, 1 - pi) + special.xlogy(n01 + n11, pi)
    ll1 = (special.xlogy(n00, 1 - pi01) + special.xlogy(n01, pi01)
           + special.xlogy(n10, 1 - pi11) + special.xlogy(n11, pi11))
    lr_ind = max(-2.0 * (ll0 - ll1), 0.0)
    lr_pof, _ = kupiec_pof(e.sum(), e.size, p)
    lr_cc = lr_pof + lr_ind
    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "lr_ind": float(lr_ind), "p_ind": float(stats.chi2.sf(lr_ind, 1)),
            "lr_cc": float(lr_cc), "p_cc": float(stats.chi2.sf(lr_cc, 2))}


def zone_bale_compte(n_exc: int) -> str:
    """Feux tricolores de Bale (VaR 99 %, 250 jours) : vert 0-4, orange 5-9, rouge >= 10."""
    return "vert" if n_exc <= 4 else ("orange" if n_exc <= 9 else "rouge")


def feux_bale(exc: np.ndarray, fenetre: int = 250) -> dict:
    """Zones de Bale sur toutes les fenetres glissantes de 250 jours."""
    c = np.convolve(np.asarray(exc, int), np.ones(fenetre, int), mode="valid")
    pire = int(c.max())
    return {"n_fenetres": int(c.size),
            "part_vert": float(np.mean(c <= 4)),
            "part_orange": float(np.mean((c >= 5) & (c <= 9))),
            "part_rouge": float(np.mean(c >= 10)),
            "max_exceptions_250j": pire,
            "pire_zone": zone_bale_compte(pire),
            "exceptions_derniere_fenetre": int(c[-1]),
            "zone_derniere_fenetre": zone_bale_compte(int(c[-1])),
            "serie": c}


def z2_acerbi_szekely(L: np.ndarray, var975: np.ndarray, es975: np.ndarray, queue: np.ndarray,
               n_sim: int = 5000, seed: int = 12345, p: float = 1 - ALPHA_ES) -> dict:
    """Test Z2 d'Acerbi-Szekely (2014) de l'ES a 97,5 %.

        Z2 = sum_t  L_t * 1{L_t > VaR_t} / (T * p * ES_t)  - 1        (E[Z2] = 0 sous H0)

    H1 : l'ES est sous-estime (Z2 > 0). P-value par simulation sous H0 : a chaque jour,
    la perte est tiree dans la loi predictive du modele, representee ici par sa queue
    (probabilite p de tomber dans la queue, puis tirage uniforme parmi les K quantiles
    stockes ; en dehors de la queue la contribution est nulle).
    """
    T, K = queue.shape
    exc = L > var975
    z2 = float(np.sum(L * exc / (T * p * es975)) - 1.0)
    rng = np.random.default_rng(seed)
    lignes = np.arange(T)[None, :]
    z2_sim = np.empty(n_sim)
    pas = 250
    for b in range(0, n_sim, pas):
        c = min(pas, n_sim - b)
        dans_queue = rng.random((c, T)) < p
        x = np.where(dans_queue, queue[lignes, rng.integers(0, K, (c, T))], 0.0)
        z2_sim[b:b + c] = (x / (T * p * es975)).sum(axis=1) - 1.0
    pval = float((1 + np.sum(z2_sim >= z2)) / (n_sim + 1))
    n_exc = int(exc.sum())
    ratio = float(L[exc].mean() / es975[exc].mean()) if n_exc else float("nan")
    return {"z2": z2, "p_value": pval, "z2_simule_sd": float(z2_sim.std()),
            "n_exceptions_975": n_exc, "taux_exceptions_975": float(n_exc / T),
            "ratio_perte_sur_es_apres_var975": ratio}


def backtest_var(L: np.ndarray, var99: np.ndarray, p: float = 1 - ALPHA_VAR) -> dict:
    """Kupiec + Christoffersen + feux de Bale pour une serie de pertes et de VaR 99 %."""
    exc = (L > var99).astype(int)
    T, x = exc.size, int(exc.sum())
    lr, pv = kupiec_pof(x, T, p)
    chri = christoffersen(exc, p)
    feux = feux_bale(exc)
    feux.pop("serie")
    return {"n_obs": int(T), "n_exceptions": x, "taux_exceptions": float(x / T),
            "taux_attendu": p, "kupiec_lr": lr, "kupiec_p": pv, "christoffersen": chri,
            "bale": feux, "exceptions": exc}


def stress_fenetre(dates, L, prev: dict, debut: str, fin: str) -> dict:
    """Perte realisee vs VaR/ES d'une methode sur une fenetre de stress."""
    masque = (dates >= pd.Timestamp(debut)) & (dates <= pd.Timestamp(fin))
    if not masque.any():
        return {"n_jours": 0}
    Ls, v, e = L[masque], prev["var99"][masque], prev["es975"][masque]
    i = int(np.argmax(Ls))
    return {"n_jours": int(masque.sum()),
            "n_exceptions_var99": int(np.sum(Ls > v)),
            "perte_max_realisee": float(Ls[i]),
            "date_perte_max": str(pd.Timestamp(dates[masque][i]).date()),
            "var99_ce_jour": float(v[i]), "es975_ce_jour": float(e[i]),
            "perte_max_sur_var99": float(Ls[i] / v[i]), "perte_max_sur_es975": float(Ls[i] / e[i]),
            "var99_moyenne": float(v.mean()), "es975_moyenne": float(e.mean()),
            "perte_moyenne_realisee": float(Ls.mean())}
