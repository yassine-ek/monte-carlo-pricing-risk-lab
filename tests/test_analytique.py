"""Les formules de référence doivent d'abord être cohérentes entre elles (limites connues, parités, différences finies)."""
import dataclasses

import numpy as np
import pytest

from mclab.analytique import (asiatique_geo_prix, bs_grecques, bs_prix, crr_americain, digitale_delta,
                              digitale_prix, down_and_out_call, heston_prix, merton_prix)
from mclab.utils import ParamsHeston, ParamsMerton, vol_implicite_bs

A = dict(S0=100.0, K=105.0, T=0.75, r=0.03, sigma=0.25, q=0.01)


def test_parite_call_put_bs():
    c, p = bs_prix(type_option="call", **A), bs_prix(type_option="put", **A)
    assert c - p == pytest.approx(A["S0"] * np.exp(-A["q"] * A["T"]) - A["K"] * np.exp(-A["r"] * A["T"]), abs=1e-12)


@pytest.mark.parametrize("type_option", ["call", "put"])
def test_grecques_bs_contre_differences_finies(type_option):
    g = bs_grecques(type_option=type_option, **A)
    f = lambda **kw: bs_prix(type_option=type_option, **{**A, **kw})
    h = 1e-4
    assert g["delta"] == pytest.approx((f(S0=A["S0"] + h) - f(S0=A["S0"] - h)) / (2 * h), abs=1e-6)
    assert g["gamma"] == pytest.approx((f(S0=A["S0"] + h) - 2 * f() + f(S0=A["S0"] - h)) / h**2, abs=1e-5)
    assert g["vega"] == pytest.approx((f(sigma=A["sigma"] + h) - f(sigma=A["sigma"] - h)) / (2 * h), abs=1e-5)
    assert g["rho"] == pytest.approx((f(r=A["r"] + h) - f(r=A["r"] - h)) / (2 * h), abs=1e-4)
    assert g["theta"] == pytest.approx(-(f(T=A["T"] + h) - f(T=A["T"] - h)) / (2 * h), abs=1e-4)


def test_heston_tend_vers_bs_quand_vol_de_vol_nulle():
    p = dataclasses.replace(ParamsHeston(), xi=1e-3)  # v0 = theta = 0.04 -> sigma = 0.2
    assert heston_prix(p, 100.0, 1.0) == pytest.approx(bs_prix(100.0, 100.0, 1.0, p.r, 0.2), abs=1e-4)


def test_heston_parite_et_bornes():
    p = ParamsHeston()
    c, put = heston_prix(p, 90.0, 2.0), heston_prix(p, 90.0, 2.0, "put")
    assert c - put == pytest.approx(p.S0 - 90.0 * np.exp(-p.r * 2.0), abs=1e-9)
    assert c >= p.S0 - 90.0 * np.exp(-p.r * 2.0)  # borne d'arbitrage inférieure
    assert heston_prix(p, 100.0, 1.0) > heston_prix(p, 110.0, 1.0)  # décroissant en K


def test_merton_sans_saut_egale_bs_et_parite():
    assert merton_prix(ParamsMerton(lam=0.0), 100.0, 1.0) == pytest.approx(bs_prix(100.0, 100.0, 1.0, 0.02, 0.2), abs=1e-10)
    p = ParamsMerton()
    c, put = merton_prix(p, 95.0, 1.0), merton_prix(p, 95.0, 1.0, "put")
    assert c - put == pytest.approx(p.S0 - 95.0 * np.exp(-p.r), abs=1e-10)


def test_asiatique_geometrique_une_date_egale_bs():
    assert asiatique_geo_prix(100.0, 100.0, 1.0, 0.02, 0.2, 1) == pytest.approx(bs_prix(100.0, 100.0, 1.0, 0.02, 0.2), abs=1e-12)
    assert asiatique_geo_prix(100.0, 100.0, 1.0, 0.02, 0.2, 12) < bs_prix(100.0, 100.0, 1.0, 0.02, 0.2)


def test_down_and_out_limites():
    v = bs_prix(100.0, 100.0, 1.0, 0.02, 0.2)
    assert down_and_out_call(100.0, 100.0, 1e-6, 1.0, 0.02, 0.2) == pytest.approx(v, abs=1e-9)  # barrière lointaine
    assert down_and_out_call(100.0, 100.0, 99.999, 1.0, 0.02, 0.2) < 0.05  # barrière collée au spot
    assert 0 < down_and_out_call(100.0, 100.0, 90.0, 1.0, 0.02, 0.2) < v
    with pytest.raises(ValueError):
        down_and_out_call(100.0, 100.0, 110.0, 1.0, 0.02, 0.2)


def test_americain_crr():
    europeen = bs_prix(100.0, 100.0, 1.0, 0.06, 0.2, type_option="put")
    am = crr_americain(100.0, 100.0, 1.0, 0.06, 0.2, N=1000)
    assert am > europeen + 0.3  # prime d'exercice anticipé
    assert crr_americain(100.0, 100.0, 1.0, 0.06, 0.2, N=2000) == pytest.approx(am, abs=5e-3)  # convergence en N
    # sans dividende, exercer un call américain avant l'échéance n'est jamais optimal
    assert crr_americain(100.0, 100.0, 1.0, 0.06, 0.2, N=1000, type_option="call") == pytest.approx(
        bs_prix(100.0, 100.0, 1.0, 0.06, 0.2), abs=5e-3)
    assert crr_americain(100.0, 100.0, 1.0, 0.06, 0.2, N=1000, pas_exercice=100) < am  # bermudéen <= américain


def test_digitale_et_vol_implicite():
    h = 1e-4
    delta_diff = (digitale_prix(100 + h, 100, 1, 0.02, 0.2) - digitale_prix(100 - h, 100, 1, 0.02, 0.2)) / (2 * h)
    assert digitale_delta(100, 100, 1, 0.02, 0.2) == pytest.approx(delta_diff, abs=1e-8)
    prix = bs_prix(100.0, 110.0, 1.0, 0.02, 0.27)
    assert vol_implicite_bs(prix, 100.0, 110.0, 1.0, 0.02) == pytest.approx(0.27, abs=1e-9)
    assert np.isnan(vol_implicite_bs(100.0, 100.0, 110.0, 1.0, 0.02))  # prix > S0 : hors bornes d'arbitrage
