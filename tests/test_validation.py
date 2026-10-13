"""Tests de second ordre : les IC95 sont-ils bien calibrés ? Les schémas respectent-ils la parité call-put ?"""
import pytest

from mclab.analytique import asiatique_geo_prix, bs_prix
from mclab.modeles import gbm_exact, heston_euler, heston_qe, merton_exact
from mclab.qmc import prix_gbm_qmc
from mclab.reduction_variance import mc_antithetique, mc_controle, mc_importance_call, mc_naif
from mclab.utils import ParamsHeston, ParamsMerton
from mclab.validation import couverture_ic95, ecart_parite

A = dict(S0=100.0, K=100.0, T=1.0, r=0.02, sigma=0.2)
REPS = 1000


def _dans_la_bande(c):
    """Couverture à 3 sigmas binomiaux de 95 % (sigma = 0,69 point pour 1000 répétitions)."""
    return abs(c["ecart_en_sigmas"]) < 3


def test_couverture_ic95_mc_naif_antithetique_controle():
    ref = bs_prix(**A)
    n = 2000
    estimateurs = [
        lambda g: mc_naif("call_europeen", n_pas=1, n_chemins=n, graine=g, **A),
        lambda g: mc_antithetique("call_europeen", n_pas=1, n_chemins=n, graine=g, **A),
        lambda g: mc_controle("call_europeen", n_pas=1, n_chemins=n, graine=g, n_pilote=5000, **A),
    ]
    for est in estimateurs:
        c = couverture_ic95(est, ref, REPS, graine=0)
        assert _dans_la_bande(c), c


def test_couverture_ic95_importance_sampling_option_tres_otm():
    b = dict(S0=100.0, K=180.0, T=1.0, r=0.02, sigma=0.2)
    c = couverture_ic95(lambda g: mc_importance_call(n_chemins=2000, graine=g, **b), bs_prix(**b), REPS)
    assert _dans_la_bande(c), c


def test_couverture_ic95_qmc_randomise():
    ref = asiatique_geo_prix(100.0, 100.0, 1.0, 0.02, 0.2, 16)
    c = couverture_ic95(lambda g: prix_gbm_qmc("call_asiatique_geo", n_pas=16, m=6, k_brouillages=16, graine=g, **A),
                        ref, 500)
    assert abs(c["couverture"] - 0.95) < 3 * 0.0097  # sigma binomial pour 500 répétitions


def test_l_ic_normal_se_degrade_pour_un_evenement_rare():
    """Contre-exemple instructif : MC naïf sur un call très OTM, n = 2000 -> l'IC normal sous-couvre."""
    b = dict(S0=100.0, K=180.0, T=1.0, r=0.02, sigma=0.2)
    c = couverture_ic95(lambda g: mc_naif("call_europeen", n_pas=1, n_chemins=2000, graine=g, **b), bs_prix(**b), 300)
    assert c["couverture"] < 0.88


def test_parite_call_put_des_schemas():
    H, M = ParamsHeston(), ParamsMerton()
    cas = {
        "gbm": (gbm_exact(100.0, 0.02, 0.2, 1.0, 5, 200_000, graine=21), 0.02),
        "heston_qe": (heston_qe(H, 1.0, 20, 200_000, graine=22), H.r),
        "heston_euler": (heston_euler(H, 1.0, 50, 100_000, graine=23), H.r),
        "merton": (merton_exact(M, 1.0, 5, 200_000, graine=24), M.r),
    }
    for nom, (S, r) in cas.items():
        res, exacte = ecart_parite(S, 100.0, 100.0, 1.0, r)
        # Euler a un biais de martingale O(dt) ; les autres schémas conservent E[S_T] exactement.
        tolerance = 3 * res.erreur_type + (0.05 if nom == "heston_euler" else 0.0)
        assert abs(res.prix - exacte) < tolerance, (nom, res.prix, exacte)
