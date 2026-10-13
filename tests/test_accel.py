"""La version numba doit donner le même prix (à l'erreur-type près), être reproductible et plus rapide."""
import numpy as np

from mclab.accel import benchmark, heston_qe_numba, prix_call_heston_numba
from mclab.analytique import heston_prix
from mclab.utils import ParamsHeston

H = ParamsHeston()


def test_prix_numba_contre_semi_ferme():
    ref = heston_prix(H, 100.0, 1.0)
    res = prix_call_heston_numba(H, 100.0, 1.0, 50, 400_000, graine=1)
    assert abs(res.z(ref)) < 3


def test_numba_reproductible_et_flux_distincts_par_chemin():
    a = heston_qe_numba(H, 1.0, 20, 1000, graine=5)
    assert np.array_equal(a, heston_qe_numba(H, 1.0, 20, 1000, graine=5))
    assert not np.array_equal(a, heston_qe_numba(H, 1.0, 20, 1000, graine=6))
    assert len(np.unique(a)) == 1000  # aucun chemin ne duplique un autre


def test_numba_plus_rapide_que_numpy():
    b = benchmark(n_chemins=50_000, n_pas=30)
    # le facteur dépend du nombre de cœurs (×23 sur 22 threads, ×1,9 sur un runner CI à 2 cœurs)
    assert b["acceleration_vs_numpy_terminal"] > 1
