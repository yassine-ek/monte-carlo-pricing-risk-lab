"""MLMC : respecte la tolérance, la variance décroît avec le niveau, et il coûte moins que le MC standard."""
import numpy as np

from mclab.analytique import bs_prix, heston_prix
from mclab.mlmc import echantillonneur_gbm_euler, echantillonneur_heston_euler, mlmc
from mclab.utils import ParamsHeston

K, T = 100.0, 1.0


def test_mlmc_gbm_a_la_tolerance():
    ref = bs_prix(100.0, K, T, 0.02, 0.2)
    eps = 0.02
    res = mlmc(echantillonneur_gbm_euler(100.0, K, T, 0.02, 0.2), eps, graine=1)
    assert abs(res.resultat.prix - ref) < 3 * eps  # RMSE visée eps -> erreur < 3 eps en pratique
    assert res.resultat.erreur_type < eps          # l'erreur-type statistique respecte le budget de variance
    assert res.cout < res.cout_mc_standard / 5     # gain net face au MC standard à même précision


def test_variance_par_niveau_decroit():
    res = mlmc(echantillonneur_gbm_euler(100.0, K, T, 0.02, 0.2), 0.05, graine=2)
    v = res.var_Y
    assert np.all(np.diff(v[1:]) < 0)                            # V_l décroissante pour l >= 1
    assert v[-1] < v[1] / 2 ** (len(v) - 2) * 3                  # décroissance ~ 2^-l (Euler, beta = 1)
    assert res.var_P[-1] > 50 * res.var_Y[-1]                    # la différence est bien moins variable que P_l
    assert np.all(np.diff(res.n_l) < 0)                          # moins de chemins sur les niveaux fins


def test_mlmc_heston_a_la_tolerance():
    p = ParamsHeston()
    eps = 0.1
    res = mlmc(echantillonneur_heston_euler(p, K, T), eps, graine=3)
    assert abs(res.resultat.prix - heston_prix(p, K, T)) < 3 * eps


def test_regle_d_arret_du_biais_suit_giles():
    """Régression (revue 2026-10-13) : le facteur 2^(alpha (l-L)) était inversé -> sur-raffinement (L=8 au lieu de 4).

    Échantillonneur synthétique sans bruit : E[Y_l] = 2^-l exactement, donc le biais restant après le niveau L
    vaut 2^-L. Arrêt attendu au premier L avec 2^-L <= eps/sqrt(2) (eps = 0,1 : 2^-4 = 0,0625 <= 0,0707 < 2^-3).
    """
    def f(l, n, rng):
        bruit = 1e-4 * rng.standard_normal(n)
        pf = 2.0 - 2.0**-l + bruit
        return pf, (2.0 - 2.0 ** -(l - 1) + bruit if l > 0 else np.zeros(n))

    res = mlmc(f, 0.1, graine=0)
    assert len(res.niveaux) - 1 == 4
    assert abs(res.resultat.prix - 2.0) < 0.07   # E[P_4] = 2 - 2^-4 ; biais restant 2^-4 < eps/sqrt(2)


def test_reproductibilite():
    f = echantillonneur_gbm_euler(100.0, K, T, 0.02, 0.2)
    assert mlmc(f, 0.1, graine=5).resultat.prix == mlmc(f, 0.1, graine=5).resultat.prix
