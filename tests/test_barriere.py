"""Barrière discrète : le MC naïf est biaisé haut, la correction par pont brownien ramène à la formule continue."""
import pytest

from mclab.analytique import down_and_out_call
from mclab.barriere import down_and_out_mc

P = dict(S0=100.0, K=100.0, H=90.0, T=1.0, r=0.02, sigma=0.2)
REF = down_and_out_call(**P)


def test_correction_pont_reduit_l_erreur():
    m = 25
    naif = down_and_out_mc(m=m, n_chemins=100_000, graine=1, methode="naif", **P)
    pont = down_and_out_mc(m=m, n_chemins=100_000, graine=1, methode="pont", **P)
    bgk = down_and_out_mc(m=m, n_chemins=100_000, graine=1, methode="bgk", **P)
    assert naif.prix - REF > 5 * naif.erreur_type           # biais haut franc du MC naïf
    assert abs(pont.prix - REF) < abs(naif.prix - REF) / 5  # le pont le réduit d'au moins un facteur 5
    assert abs(pont.z(REF)) < 3                             # et il est compatible avec la formule continue
    assert abs(bgk.prix - REF) < abs(naif.prix - REF)


def test_biais_naif_diminue_avec_le_nombre_de_dates():
    erreurs = [down_and_out_mc(m=m, n_chemins=100_000, graine=2, methode="naif", **P).prix - REF for m in (5, 50)]
    assert erreurs[1] < erreurs[0]


def test_methode_inconnue():
    with pytest.raises(ValueError):
        down_and_out_mc(m=5, n_chemins=10, methode="x", **P)
