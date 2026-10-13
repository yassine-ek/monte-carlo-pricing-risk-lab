"""Longstaff-Schwartz : proche de CRR (< 1 %), biaisé bas hors échantillon, encadré par une borne duale."""
import numpy as np

from mclab.americaine import base_laguerre, borne_haute_duale, lsm_put_americain
from mclab.analytique import bs_prix, crr_americain

P = dict(S0=100.0, K=100.0, T=1.0, r=0.06, sigma=0.2)


def test_base_de_laguerre():
    x = np.array([0.0, 1.0, 2.0])
    attendu = np.column_stack([np.ones(3), 1 - x, 1 - 2 * x + x**2 / 2, 1 - 3 * x + 1.5 * x**2 - x**3 / 6])
    assert np.allclose(base_laguerre(x, 3), attendu)


def test_lsm_a_moins_de_1_pourcent_de_crr():
    crr = crr_americain(**P, N=2000)
    lsm = lsm_put_americain(**P, n_pas=50, n_apprentissage=50_000, n_evaluation=100_000, graine=5)
    assert abs(lsm.resultat.prix / crr - 1) < 0.01
    assert lsm.resultat.prix > bs_prix(100.0, 100.0, 1.0, 0.06, 0.2, type_option="put")  # prime d'exercice captée
    assert lsm.resultat.ic_bas < lsm.resultat.prix < lsm.resultat.ic_haut


def test_lsm_ne_depasse_pas_le_bermudeen_exact():
    """Estimateur biaisé bas : à l'erreur-type près, jamais au-dessus du prix bermudéen exact (même dates)."""
    bermudeen = crr_americain(**P, N=2000, pas_exercice=40)
    lsm = lsm_put_americain(**P, n_pas=50, n_apprentissage=50_000, n_evaluation=100_000, graine=6)
    assert lsm.resultat.prix < bermudeen + 3 * lsm.resultat.erreur_type


def test_encadrement_borne_basse_crr_borne_haute():
    bermudeen = crr_americain(**P, N=2000, pas_exercice=200)  # 10 dates
    lsm = lsm_put_americain(**P, n_pas=10, n_apprentissage=50_000, n_evaluation=100_000, graine=2)
    haute = borne_haute_duale(lsm, n_exterieurs=2000)
    assert lsm.resultat.prix - 3 * lsm.resultat.erreur_type <= bermudeen <= haute.prix + 3 * haute.erreur_type
    assert haute.prix < bermudeen * 1.03  # borne utile : moins de 3 % au-dessus


def test_frontiere_d_exercice_et_reproductibilite():
    a = lsm_put_americain(**P, n_pas=20, n_apprentissage=20_000, n_evaluation=20_000, graine=1)
    b = lsm_put_americain(**P, n_pas=20, n_apprentissage=20_000, n_evaluation=20_000, graine=1)
    assert a.resultat.prix == b.resultat.prix
    f = a.frontiere[~np.isnan(a.frontiere)]
    assert f.size > 5 and np.all((f > 50) & (f < P["K"]))
