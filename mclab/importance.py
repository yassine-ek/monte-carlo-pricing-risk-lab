"""Importance sampling gaussien optimal : second moment, dérive optimale, entropie croisée, grandes déviations.

Cadre. I = E[f(Z)], Z ~ N(0,1). On tire Y ~ N(mu, 1) et on moyenne f(Y) w(Y) avec le poids de Girsanov
w(y) = phi(y) / phi_mu(y) = exp(-mu y + mu^2 / 2) : l'estimateur est sans biais pour tout mu.
Sa variance vaut M(mu) - I^2 avec le SECOND MOMENT
        M(mu) = E_mu[f^2 w^2] = E_0[f(Z)^2 exp(-mu Z + mu^2 / 2)] = e^{mu^2} E_0[f(Z - mu)^2],
convexe en mu (somme d'exponentielles de mu à coefficients positifs) : le minimum est unique.
Deux « bonnes » dérives :
  - mu* qui minimise M(mu)                                   (variance minimale dans la famille) ;
  - mu_CE = E_0[f(Z) Z] / E_0[f(Z)] qui minimise l'entropie relative KL(g* || phi_mu), g* ∝ f phi étant la
    densité de variance nulle (Rubinstein) : c'est la MOYENNE de la loi optimale.
Grandes déviations : pour l'événement rare {Z > a}, M(mu) = e^{mu^2} P(Z > a + mu) et mu* ~ a quand a -> infini :
la variance relative ne croît plus que sous-exponentiellement (« logarithmiquement efficace »).
"""
import numpy as np
from scipy import integrate, optimize
from scipy.special import ndtr, log_ndtr, ndtri
from scipy.stats import binom, norm

BORNE = 14.0   # |z| <= 14 : la densité gaussienne y est < 1e-43, le reste de l'intégrale est négligeable


def esperance(f_z, a=-BORNE, b=BORNE, points=None):
    """I = E[f(Z)] par quadrature adaptative (points : abscisses singulières, ex. le strike)."""
    return integrate.quad(lambda z: f_z(z) * norm.pdf(z), a, b, points=points, limit=500, epsabs=0, epsrel=1e-12)[0]


def second_moment(f_z, mu, a=-BORNE, b=BORNE, points=None):
    """M(mu) = int f(z)^2 phi(z) exp(-mu z + mu^2/2) dz (l'exposant est regroupé pour éviter tout débordement)."""
    def g(z):
        return f_z(z) ** 2 * np.exp(-0.5 * z * z - mu * z + 0.5 * mu * mu) / np.sqrt(2.0 * np.pi)
    return integrate.quad(g, a, b, points=points, limit=500, epsabs=0, epsrel=1e-12)[0]


def variance_is(f_z, mu, moyenne=None, **kw):
    """Variance PAR TIRAGE de l'estimateur d'importance de dérive mu : M(mu) - I^2 (mu = 0 : MC naïf)."""
    moyenne = esperance(f_z, **kw) if moyenne is None else moyenne
    return second_moment(f_z, mu, **kw) - moyenne**2


def mu_optimal(f_z, bracket=(0.0, 12.0), **kw):
    """mu* qui minimise ln M(mu) (convexe en mu, recherche bornée de Brent)."""
    res = optimize.minimize_scalar(lambda m: np.log(second_moment(f_z, m, **kw)), bracket=bracket,
                                   method="brent", tol=1e-10)
    return float(res.x)


def mu_entropie_croisee_exact(f_z, **kw):
    """Point fixe de la méthode d'entropie croisée : moyenne de Z sous la loi de densité ∝ f phi."""
    num = integrate.quad(lambda z: z * f_z(z) * norm.pdf(z), kw.get("a", -BORNE), kw.get("b", BORNE),
                         points=kw.get("points"), limit=500, epsabs=0, epsrel=1e-12)[0]
    return float(num / esperance(f_z, **kw))


def entropie_croisee(f_z, n_iter=6, n=20_000, graine=0, mu0=0.0, lissage=0.7):
    """Méthode d'entropie croisée itérative (Rubinstein-Kroese) pour la famille N(mu, 1).

    À chaque itération : tirer Y ~ N(mu_t, 1) ; mu_{t+1} = somme(f(Y) w(Y) Y) / somme(f(Y) w(Y))  (moyenne pondérée
    des tirages par leur contribution f w), puis lissage mu <- lissage mu_new + (1 - lissage) mu_t. Renvoie la trajectoire.
    Si aucun tirage ne tombe dans la région utile (f = 0 partout), on avance d'un cran vers la queue.
    """
    rng = np.random.default_rng(graine)
    mu, trajet = float(mu0), [float(mu0)]
    for _ in range(n_iter):
        y = rng.standard_normal(n) + mu
        poids = f_z(y) * np.exp(-mu * y + 0.5 * mu * mu)
        s = poids.sum()
        mu_new = float(np.sum(poids * y) / s) if s > 0 else mu + 1.0
        mu = lissage * mu_new + (1.0 - lissage) * mu
        trajet.append(mu)
    return trajet


def estimer_is(f_z, mu, n, graine=0):
    """(estimation, variance empirique par tirage) de l'estimateur d'importance de dérive mu."""
    y = np.random.default_rng(graine).standard_normal(n) + mu
    x = f_z(y) * np.exp(-mu * y + 0.5 * mu * mu)
    return float(x.mean()), float(x.var(ddof=1))


# ---------------------------------------------------------------- cas d'étude
def payoff_call(S0, K, T, r, sigma, q=0.0):
    """z -> payoff actualisé d'un call, S_T = S0 exp((r - q - sigma^2/2) T + sigma sqrt(T) z)."""
    def f(z):
        return np.exp(-r * T) * np.maximum(S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z) - K, 0.0)
    return f


def z_strike(S0, K, T, r, sigma, q=0.0):
    return (np.log(K / S0) - (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def perte_portefeuille_credit(m=100, pd=0.01, rho=0.2, k=10):
    """y -> P(au moins k défauts parmi m | facteur Y = y) : modèle à un facteur gaussien (Vasicek).

    Obligé i en défaut si sqrt(rho) Y + sqrt(1-rho) eps_i < Phi^{-1}(pd). Sachant Y, les défauts sont
    binomiaux de probabilité p(y) = Phi((Phi^{-1}(pd) - sqrt(rho) y) / sqrt(1-rho)). La dérive d'importance
    porte sur Y (mu < 0 : mauvaises conjonctures), l'aléa idiosyncratique étant intégré exactement par la loi binomiale.
    """
    c = ndtri(pd)

    def f(y):
        p_cond = ndtr((c - np.sqrt(rho) * np.asarray(y)) / np.sqrt(1.0 - rho))
        return binom.sf(k - 1, m, p_cond)
    return f


def efficacite_log_evenement(a_valeurs):
    """Événement {Z > a} : (mu*, ln M(mu*) / (2 ln P)) pour chaque a. Le rapport -> 1 (efficacité logarithmique).

    M(mu) = e^{mu^2} P(Z > a + mu) en forme fermée ; on travaille en logarithmes (log_ndtr) pour a grand.
    """
    sortie = []
    for a in a_valeurs:
        def log_m(mu, a=a):
            return mu * mu + log_ndtr(-(a + mu))
        mu_star = optimize.minimize_scalar(log_m, bracket=(0.0, a + 1.0), method="brent", tol=1e-12).x
        log_p = log_ndtr(-a)
        log_mstar = log_m(mu_star)
        # variance relative de l'IS : (M - P^2) / P^2 ; naïve : (1 - P) / P
        re_is = float(np.sqrt(np.expm1(log_mstar - 2.0 * log_p)))
        with np.errstate(over="ignore"):          # a > 38 : l'erreur relative du MC naïf dépasse 1e300, on la laisse valoir +inf
            re_naif = float(np.sqrt(np.expm1(-log_p)))
        sortie.append({"a": float(a), "mu_star": float(mu_star), "log_P": float(log_p),
                       "rapport_log": float(log_mstar / (2.0 * log_p)),
                       "erreur_relative_is": re_is, "erreur_relative_naive": re_naif})
    return sortie
