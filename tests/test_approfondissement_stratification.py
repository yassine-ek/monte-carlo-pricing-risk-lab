"""Approfondissement Q4 : stratification (proportionnelle, Neyman) et hypercube latin, contre la théorie."""
import numpy as np
from scipy.special import ndtri

from mclab import stratification as st
from mclab.importance import payoff_call


def _call(K):
    fz = payoff_call(100.0, K, 1.0, 0.02, 0.2)
    return fz, (lambda u, rng: fz(ndtri(u)))


def test_allocation_neyman_somme_n_et_proportionnelle_aux_sigma():
    n_h = st.allocation_neyman(np.array([0.0, 1.0, 3.0, 6.0]), 1000, minimum=2)
    assert n_h.sum() == 1000 and n_h.min() >= 2
    assert n_h[3] > n_h[2] > n_h[1] > n_h[0]
    assert abs(n_h[3] / n_h[2] - 2.0) < 0.05


def test_ordre_des_variances_theoriques_naif_prop_neyman():
    rng = np.random.default_rng(0)
    for _ in range(20):
        mu, sg = rng.normal(size=12), np.abs(rng.normal(size=12))
        v = st.variances_theoriques(mu, sg)
        assert v["neyman"] <= v["proportionnelle"] + 1e-15 <= v["naif"] + 1e-15      # Cauchy-Schwarz puis loi de la variance totale
        assert abs(v["naif"] - (np.mean(sg**2) + np.var(mu))) < 1e-12


def test_variance_stratifiee_inferieure_a_la_naive_et_egale_a_la_theorie():
    fz, fu = _call(100.0)
    H, n, R = 16, 2048, 300
    mu, sg = st.moments_strates_numeriques(fz, H)
    th = st.variances_theoriques(mu, sg)
    rng = np.random.default_rng(1)
    v_naif = np.var([fu(rng.random(n), rng).mean() for _ in range(R)], ddof=1) * n
    v_prop = np.var([st.estimer_stratifie(fu, H, n, "proportionnelle", rng)[0] for _ in range(R)], ddof=1) * n
    assert v_prop < v_naif / 5                                       # gain réel
    assert abs(v_naif / th["naif"] - 1) < 0.25                        # la variance du MC naïf est sigma^2 / n
    assert abs(v_prop / th["proportionnelle"] - 1) < 0.25             # 300 répétitions : ~ 8 % d'erreur relative
    mean, var_est = st.estimer_stratifie(fu, H, n, "proportionnelle", rng)
    assert abs(mean - st.moments_strates_numeriques(fz, 1)[0][0]) < 5 * np.sqrt(var_est)    # sans biais (vrai prix = moyenne sur 1 strate)


def test_neyman_bat_la_proportionnelle_quand_les_sigma_sont_inegaux():
    fz, fu = _call(140.0)
    H, n, R = 16, 2048, 200
    mu, sg = st.moments_strates_numeriques(fz, H)
    th = st.variances_theoriques(mu, sg)
    assert th["neyman"] < th["proportionnelle"] / 5
    rng = np.random.default_rng(2)
    v_prop = np.var([st.estimer_stratifie(fu, H, n, "proportionnelle", rng)[0] for _ in range(R)], ddof=1)
    v_ney = np.var([st.estimer_stratifie(fu, H, n, "neyman_oracle", rng, sigmas=sg)[0] for _ in range(R)], ddof=1)
    assert v_ney < v_prop / 4
    assert abs(n * v_ney / th["neyman"] - 1) < 0.4


def test_hypercube_latin_une_observation_par_tranche_et_marges_uniformes():
    u = st.hypercube_latin(64, 5, np.random.default_rng(3))
    assert u.shape == (64, 5)
    for j in range(5):
        assert sorted((u[:, j] * 64).astype(int)) == list(range(64))


def test_lhs_elimine_la_partie_additive_et_rien_que_cela():
    d, n, R = 8, 256, 300
    additif = lambda Z: np.sum(Z**2, axis=1)                            # 100 % additif : somme S_j = 1
    interaction = lambda Z: Z[:, 0] * Z[:, 1]                           # 0 % additif : produit pur
    for f, borne in ((additif, "grand"), (interaction, "nul")):
        vl, _ = st.variance_estimateur(lambda rng: st.estimation_lhs(f, n, d, rng), R, graine=1)
        vm, _ = st.variance_estimateur(lambda rng: float(np.mean(f(rng.standard_normal((n, d))))), R, graine=2)
        if borne == "grand":
            assert vm / vl > 20
        else:
            assert 0.7 < vm / vl < 1.4                                   # Stein : aucun gain sans partie additive
