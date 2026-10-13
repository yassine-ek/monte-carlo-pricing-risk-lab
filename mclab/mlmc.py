"""Multilevel Monte Carlo de Giles (2008), schémas d'Euler et de Milstein, raffinement x2.

Idée : E[P_L] = E[P_0] + somme_{l=1..L} E[P_l - P_{l-1}]. Les différences P_l - P_{l-1}, calculées
avec les MÊMES incréments browniens (grossiers = sommes des fins), ont une variance V_l qui décroît
avec le niveau : on met beaucoup de chemins sur les niveaux grossiers (bon marché) et très peu
sur les niveaux fins (chers). Coût ~ eps^-2 au lieu de ~ eps^-3 pour le MC standard.
"""
import time
from dataclasses import dataclass, replace

import numpy as np

from mclab.utils import ParamsHeston, Z95, ResultatMC

TAILLE_LOT = 2_000_000  # éléments (chemins x pas fins) simulés d'un coup, pour borner la mémoire


@dataclass
class ResultatMLMC:
    resultat: ResultatMC
    niveaux: np.ndarray   # 0..L
    moy_Y: np.ndarray     # E_l  = moyenne de P_l - P_{l-1}
    var_Y: np.ndarray     # V_l  = variance de P_l - P_{l-1}
    moy_P: np.ndarray     # moyenne de P_l (niveau fin seul)
    var_P: np.ndarray     # variance de P_l
    n_l: np.ndarray       # nombre de chemins par niveau
    cout: float           # sum n_l 2^l : pas de temps fins simulés
    cout_mc_standard: float  # coût d'un MC standard de même précision (estimation à la Giles)


# ---------------------------------------------------------------- échantillonneurs par niveau
def _payoff(s, K, r, T):
    return np.exp(-r * T) * np.maximum(s - K, 0.0)


def echantillonneur_gbm_euler(S0, K, T, r, sigma, q=0.0):
    """Renvoie f(l, n, rng) -> (P_fin, P_grossier) pour un call GBM discrétisé par Euler sur S.

    Euler sur S (pas sur ln S) volontairement : le schéma exact n'aurait aucun biais, donc rien à étudier.
    """
    mu = r - q

    def f(l, n, rng):
        nf = 2**l
        dt = T / nf
        racine = np.sqrt(dt)
        s_f = np.full(n, S0)
        if l == 0:
            s_f = s_f * (1.0 + mu * dt + sigma * racine * rng.standard_normal(n))
            return _payoff(s_f, K, r, T), np.zeros(n)
        s_c = np.full(n, S0)
        for _ in range(nf // 2):
            dw1, dw2 = racine * rng.standard_normal(n), racine * rng.standard_normal(n)
            s_f = s_f * (1.0 + mu * dt + sigma * dw1)
            s_f = s_f * (1.0 + mu * dt + sigma * dw2)
            s_c = s_c * (1.0 + mu * 2 * dt + sigma * (dw1 + dw2))
        return _payoff(s_f, K, r, T), _payoff(s_c, K, r, T)

    return f


def echantillonneur_heston_euler(p: ParamsHeston, K, T):
    """Même chose pour Heston (Euler full-truncation sur (ln S, v)), incréments corrélés couplés."""
    c = np.sqrt(1.0 - p.rho**2)

    def pas(ln_s, v, dwv, dws, dt):
        v_plus = np.maximum(v, 0.0)
        ln_s = ln_s + (p.r - p.q - 0.5 * v_plus) * dt + np.sqrt(v_plus) * dws
        return ln_s, v + p.kappa * (p.theta - v_plus) * dt + p.xi * np.sqrt(v_plus) * dwv

    def f(l, n, rng):
        nf = 2**l
        dt = T / nf
        racine = np.sqrt(dt)
        ln_f, v_f = np.full(n, np.log(p.S0)), np.full(n, p.v0)
        if l == 0:
            z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
            ln_f, _ = pas(ln_f, v_f, racine * z1, racine * (p.rho * z1 + c * z2), dt)
            return _payoff(np.exp(ln_f), K, r=p.r, T=T), np.zeros(n)
        ln_c, v_c = ln_f.copy(), v_f.copy()
        for _ in range(nf // 2):
            dwv, dws = [], []
            for _ in range(2):
                z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
                dwv.append(racine * z1)
                dws.append(racine * (p.rho * z1 + c * z2))
                ln_f, v_f = pas(ln_f, v_f, dwv[-1], dws[-1], dt)
            ln_c, v_c = pas(ln_c, v_c, dwv[0] + dwv[1], dws[0] + dws[1], 2 * dt)
        return (_payoff(np.exp(ln_f), K, p.r, T), _payoff(np.exp(ln_c), K, p.r, T))

    return f


# ---------------------------------------------------------------- algorithme de Giles
def _nombres_optimaux(var, cout, epsilon):
    """n_l = ceil(2 eps^-2 sqrt(V_l/C_l) sum_k sqrt(V_k C_k)) : minimise le coût pour une variance eps^2/2."""
    return np.ceil(2.0 * epsilon**-2 * np.sqrt(var / cout) * np.sum(np.sqrt(var * cout))).astype(int)


def mlmc(echantillonneur, epsilon, graine=0, n0=10_000, l_min=2, l_max=12, alpha=1.0, beta=1.0):
    """MLMC adaptatif : RMSE visée epsilon (biais^2 <= eps^2/2, variance <= eps^2/2).

    alpha : ordre faible attendu (Euler : 1) ; beta : décroissance de V_l (Euler : ~1).
    Nouveau niveau ajouté tant que le biais résiduel estimé dépasse eps/sqrt(2).
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(graine)
    somme = {nom: [] for nom in ("n", "y", "y2", "p", "p2")}

    def ajouter_niveau():
        for nom in somme:
            somme[nom].append(0.0)

    def tirer(l, nombre):
        lot = max(1, TAILLE_LOT // 2**l)
        while nombre > 0:
            b = min(nombre, lot)
            p_f, p_c = echantillonneur(l, b, rng)
            y = p_f - p_c
            for nom, val in (("n", b), ("y", y.sum()), ("y2", (y**2).sum()),
                             ("p", p_f.sum()), ("p2", (p_f**2).sum())):
                somme[nom][l] += val
            nombre -= b

    L = l_min
    for _ in range(L + 1):
        ajouter_niveau()
    a_tirer = np.full(L + 1, n0)
    while a_tirer.sum() > 0:
        for l in range(L + 1):
            if a_tirer[l] > 0:
                tirer(l, int(a_tirer[l]))
        n = np.array(somme["n"])
        moy_y = np.array(somme["y"]) / np.maximum(n, 1)
        var_y = np.maximum(np.array(somme["y2"]) / np.maximum(n, 1) - moy_y**2, 1e-12)
        if n[L] == 0:  # niveau tout juste ajouté : variance devinée par extrapolation
            var_y[L] = var_y[L - 1] / 2.0**beta
        cout_l = 2.0 ** np.arange(L + 1)
        a_tirer = np.maximum(0, _nombres_optimaux(var_y, cout_l, epsilon) - n)
        if np.all(a_tirer <= 0.01 * n):  # quasi convergé en nombre d'échantillons : test du biais
            niveaux_test = np.arange(max(0, L - 2), L + 1)
            # Y_l ~ 2^(-alpha l) : on ramène |Y_l| au niveau L en le MULTIPLIANT par 2^(alpha (l - L)) (Giles 2008)
            reste = np.max(np.abs(moy_y[niveaux_test]) * 2.0 ** (alpha * (niveaux_test - L))) / (2.0**alpha - 1.0)
            if reste > epsilon / np.sqrt(2.0) and L < l_max:
                L += 1
                ajouter_niveau()
                var_y = np.append(var_y, var_y[-1] / 2.0**beta)
                n = np.append(n, 0)
                a_tirer = np.maximum(0, _nombres_optimaux(var_y, 2.0 ** np.arange(L + 1), epsilon) - n)
            else:
                break

    n = np.array(somme["n"])
    moy_y = np.array(somme["y"]) / n
    var_y = np.array(somme["y2"]) / n - moy_y**2
    moy_p = np.array(somme["p"]) / n
    var_p = np.array(somme["p2"]) / n - moy_p**2
    prix = float(moy_y.sum())
    se = float(np.sqrt(np.sum(var_y / n)))
    cout = float(np.sum(n * 2.0 ** np.arange(L + 1)))
    cout_std = float(2.0 * var_p[L] / epsilon**2 * 2.0**L)
    res = ResultatMC(prix, se, prix - Z95 * se, prix + Z95 * se, int(n.sum()), time.perf_counter() - t0)
    return ResultatMLMC(res, np.arange(L + 1), moy_y, var_y, moy_p, var_p, n, cout, cout_std)


def cout_vs_epsilon(echantillonneur, epsilons, graine=0, **options_mlmc):
    """Un MLMC par epsilon ; le coût du MC standard est lissé (même exigence de biais pour tous les epsilon).

    MC standard : biais ~ c h => il faut 2^L >= sqrt(2) c / eps pas, puis N = 2 Var[P_L] / eps^2 chemins.
    Le coefficient c est ajusté sur les niveaux fins du run le plus précis (|E[P_l - P_{l-1}]| ~ c 2^-l),
    sinon le nombre de niveaux L choisi par l'algorithme adaptatif fait « sauter » la courbe.
    """
    res = [mlmc(echantillonneur, eps, graine=graine + i, **options_mlmc) for i, eps in enumerate(epsilons)]
    fin = res[int(np.argmin(epsilons))]
    niveaux = fin.niveaux[len(fin.niveaux) // 2:]
    c = float(np.median(np.abs(fin.moy_Y[niveaux]) * 2.0**niveaux))
    sortie = []
    for r, eps in zip(res, epsilons):
        L_std = max(0, int(np.ceil(np.log2(np.sqrt(2.0) * c / eps))))
        sortie.append(replace(r, cout_mc_standard=float(2.0 * r.var_P[-1] / eps**2 * 2.0**L_std)))
    return sortie


# ---------------------------------------------------------------- Milstein (approfondissement, théorème de complexité de Giles)
def echantillonneur_gbm_milstein(S0, K, T, r, sigma, q=0.0):
    """Comme echantillonneur_gbm_euler, mais avec le pas de Milstein : S <- S (1 + mu h + sigma dW + sigma^2 (dW^2 - h) / 2).

    Pourquoi : le payoff du call est LIPSCHITZ, donc Var(P_fin - P_gros) <= L^2 E[(S_fin - S_gros)^2] ; avec un ordre
    fort 1 (Milstein) cette quantité est O(h^2), soit V_l ~ 2^(-2 l) (beta = 2) au lieu de 2^(-l) (Euler).
    Le couplage est inchangé : l'incrément grossier est la somme des deux incréments fins.
    """
    mu = r - q

    def pas(s, dw, h):
        return s * (1.0 + mu * h + sigma * dw + 0.5 * sigma**2 * (dw**2 - h))

    def f(l, n, rng):
        nf = 2**l
        dt = T / nf
        racine = np.sqrt(dt)
        s_f = np.full(n, S0)
        if l == 0:
            return _payoff(pas(s_f, racine * rng.standard_normal(n), dt), K, r, T), np.zeros(n)
        s_c = np.full(n, S0)
        for _ in range(nf // 2):
            dw1, dw2 = racine * rng.standard_normal(n), racine * rng.standard_normal(n)
            s_f = pas(pas(s_f, dw1, dt), dw2, dt)
            s_c = pas(s_c, dw1 + dw2, 2 * dt)
        return _payoff(s_f, K, r, T), _payoff(s_c, K, r, T)

    return f


def echantillonneur_heston_milstein(p: ParamsHeston, K, T):
    """Heston « Milstein SANS aire de Lévy » (full truncation) : variance en Milstein exact, ln S avec les termes symétriques.

    Termes de Milstein pour ln S (b_1 = (rho sqrt v, xi sqrt v), b_2 = (c sqrt v, 0), c = sqrt(1-rho^2)) :
      (xi rho / 4) (dW1^2 - h)  +  (xi c / 2) * I_12 ,   I_12 = dW1 dW2 / 2 + aire de Lévy.
    On garde dW1 dW2 / 2 et on JETTE l'aire (la simuler exactement est coûteux). Conséquence théorique
    (Giles-Szpruch 2014) : l'erreur forte sur S reste d'ordre 1/2 dès que rho != +-1, donc beta ~ 1 et non 2.
    """
    c = np.sqrt(1.0 - p.rho**2)

    def pas(ln_s, v, dw1, dw2, h):
        vp = np.maximum(v, 0.0)
        sv = np.sqrt(vp)
        dws = p.rho * dw1 + c * dw2
        ln_s = (ln_s + (p.r - p.q - 0.5 * vp) * h + sv * dws
                + 0.25 * p.xi * p.rho * (dw1**2 - h) + 0.25 * p.xi * c * dw1 * dw2)
        v = v + p.kappa * (p.theta - vp) * h + p.xi * sv * dw1 + 0.25 * p.xi**2 * (dw1**2 - h)
        return ln_s, v

    def f(l, n, rng):
        nf = 2**l
        h = T / nf
        racine = np.sqrt(h)
        ln_f, v_f = np.full(n, np.log(p.S0)), np.full(n, p.v0)
        if l == 0:
            ln_f, _ = pas(ln_f, v_f, racine * rng.standard_normal(n), racine * rng.standard_normal(n), h)
            return _payoff(np.exp(ln_f), K, p.r, T), np.zeros(n)
        ln_c, v_c = ln_f.copy(), v_f.copy()
        for _ in range(nf // 2):
            d1 = racine * rng.standard_normal(n), racine * rng.standard_normal(n)
            d2 = racine * rng.standard_normal(n), racine * rng.standard_normal(n)
            ln_f, v_f = pas(ln_f, v_f, d1[0], d1[1], h)
            ln_f, v_f = pas(ln_f, v_f, d2[0], d2[1], h)
            ln_c, v_c = pas(ln_c, v_c, d1[0] + d2[0], d1[1] + d2[1], 2 * h)
        return _payoff(np.exp(ln_f), K, p.r, T), _payoff(np.exp(ln_c), K, p.r, T)

    return f


def _pente_ic(x, y, poids=None):
    """Régression linéaire y ~ a + b x (moindres carrés, pondérée si poids) ; renvoie (b, IC95 de b) par Student."""
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    ddl = len(x) - 2
    if ddl < 1:   # moins de 3 points : pas d'intervalle de confiance
        return (float(np.polyfit(x, y, 1)[0]) if len(x) == 2 else float("nan")), [float("nan"), float("nan")]
    if poids is None:
        r = stats.linregress(x, y)
        b, se = r.slope, r.stderr
    else:
        w = np.asarray(poids, float)
        xm, ym = np.sum(w * x) / np.sum(w), np.sum(w * y) / np.sum(w)
        b = np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2)
        a = ym - b * xm
        se = np.sqrt(np.sum(w * (y - a - b * x) ** 2) / ddl / np.sum(w * (x - xm) ** 2))
    t = stats.t.ppf(0.975, ddl)
    return float(b), [float(b - t * se), float(b + t * se)]


def estimer_exposants(echantillonneur, L=8, n=200_000, l_min=2, graine=0, croissance=0.0, l_min_alpha=3):
    """Estime les exposants du théorème de Giles par régression sur les niveaux l = l_min..L.

        |E[P_l - P]| ~ 2^(-alpha l)   (on régresse E[Y_l] = E[P_l - P_{l-1}] ~ 2^(-alpha l))
        Var[Y_l]     ~ 2^(-beta l)
        coût d'un échantillon ~ 2^(gamma l)   (gamma = 1 par construction : nombre de pas ; on mesure aussi le TEMPS)

    n échantillons au niveau 0 puis n 2^(croissance l) au niveau l (croissance = 0 : les mêmes partout ; 0,5 compense
    la perte de signal de E[Y_l] aux niveaux fins). C'est une expérience d'estimation des exposants, pas l'algorithme
    adaptatif. Pour alpha on exclut les niveaux l < l_min_alpha : E[Y_l] y change encore de signe pour Euler (régime
    pré-asymptotique), et l'ajustement est non linéaire, pondéré par 1/se^2.
    """
    import time
    rng = np.random.default_rng(graine)
    niveaux = np.arange(L + 1)
    moy_y, var_y, se_y, temps = [], [], [], []
    for l in niveaux:
        t0 = time.perf_counter()
        lot = max(1, TAILLE_LOT // 2**int(l))
        s = s2 = 0.0
        n_l = int(n * 2 ** (croissance * l))
        restant = n_l
        while restant > 0:
            b = min(restant, lot)
            p_f, p_c = echantillonneur(int(l), b, rng)
            y = p_f - p_c
            s, s2, restant = s + y.sum(), s2 + (y**2).sum(), restant - b
        temps.append((time.perf_counter() - t0) / n_l)
        m = s / n_l
        moy_y.append(m)
        var_y.append(max(s2 / n_l - m**2, 1e-300))
        se_y.append(np.sqrt(var_y[-1] / n_l))
    moy_y, var_y, se_y, temps = (np.array(a) for a in (moy_y, var_y, se_y, temps))
    sel = niveaux >= l_min
    beta, ic_beta = _pente_ic(niveaux[sel], -np.log2(var_y[sel]))
    # alpha : ajustement NON linéaire pondéré E[Y_l] = c 2^(-alpha l) (poids 1/se^2). Une régression sur log|E[Y_l]| serait
    # faussée aux niveaux fins, où E[Y_l] est de l'ordre de son erreur-type (signe aléatoire, log|.| biaisé).
    from scipy.optimize import curve_fit
    sel_a = niveaux >= l_min_alpha
    try:
        (c_fit, alpha), cov = curve_fit(lambda l, c, a: c * 2.0 ** (-a * l), niveaux[sel_a], moy_y[sel_a],
                                        p0=[moy_y[sel_a][0] * 2.0 ** l_min_alpha, 1.0], sigma=se_y[sel_a],
                                        absolute_sigma=True, maxfev=20000)
        se_alpha = float(np.sqrt(cov[1, 1]))
        alpha, ic_alpha = float(alpha), [float(alpha - 1.96 * se_alpha), float(alpha + 1.96 * se_alpha)]
    except RuntimeError:
        alpha, ic_alpha = float("nan"), [float("nan"), float("nan")]
    sel_t = niveaux >= 3
    gamma_t, ic_gamma_t = _pente_ic(niveaux[sel_t], np.log2(temps[sel_t]))
    return {"niveaux": niveaux.tolist(), "moy_Y": moy_y.tolist(), "var_Y": var_y.tolist(), "se_Y": se_y.tolist(),
            "temps_par_echantillon": temps.tolist(), "alpha": alpha, "ic_alpha": ic_alpha,
            "beta": beta, "ic_beta": ic_beta, "gamma_nominal": 1.0, "gamma_temps": gamma_t,
            "ic_gamma_temps": ic_gamma_t, "niveaux_alpha": niveaux[sel_a].tolist()}


def regime_giles(alpha, beta, gamma, tol=0.1):
    """Coût MLMC prédit par le théorème de Giles : eps^-2 si beta > gamma, eps^-2 (log eps)^2 si beta = gamma, sinon plus lent.

    `tol` : écart toléré pour déclarer beta = gamma (les exposants sont estimés, pas exacts).
    """
    if beta > gamma + tol:
        return {"regime": "beta > gamma", "exposant_cout": 2.0, "forme": "eps^-2"}
    if beta >= gamma - tol:
        return {"regime": "beta ~ gamma", "exposant_cout": 2.0, "forme": "eps^-2 (log eps)^2"}
    return {"regime": "beta < gamma", "exposant_cout": 2.0 + (gamma - beta) / alpha,
            "forme": "eps^-(2+(gamma-beta)/alpha)"}
