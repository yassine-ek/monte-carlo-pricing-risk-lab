"""Briques communes : résultat d'un estimateur, paramètres de modèles, payoffs, vol implicite."""
from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import brentq
from scipy.stats import t as student

Z95 = 1.959963984540054  # quantile 97,5 % de la loi normale


@dataclass
class ResultatMC:
    """Un estimateur Monte Carlo : valeur, erreur-type, IC95, taille d'échantillon, temps."""
    prix: float
    erreur_type: float
    ic_bas: float
    ic_haut: float
    n_chemins: int
    temps_s: float

    def z(self, reference):
        """Écart à la référence, en nombre d'erreurs-types."""
        if self.erreur_type == 0:  # estimateur dégénéré (ex. dérivée pathwise d'une indicatrice)
            return 0.0 if self.prix == reference else float("inf")
        return (self.prix - reference) / self.erreur_type

    def contient(self, reference):
        return self.ic_bas <= reference <= self.ic_haut


@dataclass(frozen=True)
class ParamsHeston:
    """Paramètres de référence du projet (S0=100, K=100, T=1 se règlent à l'appel)."""
    S0: float = 100.0
    r: float = 0.02
    q: float = 0.0
    v0: float = 0.04
    kappa: float = 1.5
    theta: float = 0.04
    xi: float = 0.5
    rho: float = -0.7


@dataclass(frozen=True)
class ParamsMerton:
    """Diffusion de Merton : sauts lognormaux ln(1+J) ~ N(mu_j, delta_j^2), intensité lam."""
    S0: float = 100.0
    r: float = 0.02
    q: float = 0.0
    sigma: float = 0.2
    lam: float = 0.5
    mu_j: float = -0.1
    delta_j: float = 0.15


def resumer(echantillon, t0, n_chemins=None, student_ic=False):
    """Moyenne, erreur-type et IC95 d'un échantillon i.i.d. ; t0 = time.perf_counter() de départ.

    n_chemins : nombre de chemins réellement simulés si différent de la taille de
    l'échantillon (cas antithétique : l'échantillon contient des moyennes de paires).
    student_ic : quantile de Student (peu de réplications, ex. QMC randomisé).
    """
    x = np.asarray(echantillon, dtype=float)
    n = x.size
    moyenne = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n))
    q = student.ppf(0.975, n - 1) if student_ic else Z95
    return ResultatMC(moyenne, se, moyenne - q * se, moyenne + q * se,
                      int(n_chemins or n), time.perf_counter() - t0)


def payoff_actualise(nom, S, K, r, T):
    """Payoff actualisé pour chaque chemin S de forme (n, n_pas+1).

    Noms : call_europeen, put_europeen, call_asiatique_arith, call_asiatique_geo.
    Les dates de moyenne sont t_1..t_n (S[:, 0] = S0 est exclu).
    """
    if nom == "call_europeen":
        valeur = np.maximum(S[:, -1] - K, 0.0)
    elif nom == "put_europeen":
        valeur = np.maximum(K - S[:, -1], 0.0)
    elif nom == "call_asiatique_arith":
        valeur = np.maximum(S[:, 1:].mean(axis=1) - K, 0.0)
    elif nom == "call_asiatique_geo":
        valeur = np.maximum(np.exp(np.log(S[:, 1:]).mean(axis=1)) - K, 0.0)
    else:
        raise ValueError(f"payoff inconnu : {nom}")
    return np.exp(-r * T) * valeur


def resumer_paires(valeurs, t0):
    """Résumé d'un échantillon antithétique rangé en [moitié, miroir] : on moyenne chaque paire.

    Les deux membres d'une paire sont corrélés ; l'unité i.i.d. est la moyenne de la paire.
    """
    moitie = valeurs.size // 2
    paires = 0.5 * (valeurs[:moitie] + valeurs[moitie:])
    return resumer(paires, t0, n_chemins=valeurs.size)


def vol_implicite_bs(prix, S0, K, T, r, q=0.0, type_option="call"):
    """Volatilité implicite Black-Scholes par brentq ; NaN si le prix viole les bornes d'arbitrage."""
    from mclab.analytique import bs_prix  # import local : analytique importe utils

    def ecart(sigma):
        return bs_prix(S0, K, T, r, sigma, q, type_option) - prix

    bas, haut = 1e-6, 5.0
    if ecart(bas) > 0 or ecart(haut) < 0:
        return float("nan")
    return brentq(ecart, bas, haut, xtol=1e-12)
