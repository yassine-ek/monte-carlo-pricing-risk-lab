"""Approfondissement Q7 : puissance de Kupiec / Christoffersen et contributions d'Euler à l'ES."""
import numpy as np
from conftest import sans_donnees

from mclab import contributions_es as ce
from mclab import puissance as pu
from mclab import risque as rk


def test_lr_pof_vectorise_egal_a_kupiec_du_module_risque():
    for x, T in ((0, 250), (3, 250), (7, 500), (25, 2500)):
        assert abs(pu.lr_pof(x, T) - rk.kupiec_pof(x, T)[0]) < 1e-12


def test_puissance_exacte_connue_et_monte_carlo_coherent():
    assert abs(pu.rejet_kupiec_exact(250, 0.02) - 0.2427) < 1e-3
    assert abs(pu.rejet_kupiec_exact(2500, 0.02) - 0.9846) < 1e-3
    for T in (250, 1250):
        ex = pu.rejet_kupiec_exact(T, 0.02)
        mc, se = pu.rejet_kupiec_mc(T, 0.02, 100_000, graine=T)
        assert abs(mc - ex) < 4 * se
    assert pu.rejet_kupiec_exact(250, 0.01) > 0.07          # la taille réelle dépasse 5 % à 250 jours (loi discrète)


def test_courbe_de_puissance_croit_et_horizon_80_pour_cent():
    Ts = np.arange(100, 3001, 50)
    p = pu.courbe_puissance_kupiec(Ts, 0.02)
    assert p[-1] > 0.99 and p[0] < 0.5
    assert np.mean(np.diff(p) > -0.1) > 0.98 and p[-10:].min() > p[:10].max()
    t80 = pu.horizon_puissance(Ts, p, 0.8)
    assert 900 <= t80 <= 1400
    assert 0.8 < pu.taille_echantillon_normale(0.01, 0.02) / t80 < 1.2        # la formule normale est de l'ordre juste
    z = pu.proba_zone_bale(0.02)
    assert abs(sum(z.values()) - 1) < 1e-12 and 0.4 < z["vert"] < 0.5
    assert pu.proba_zone_bale(0.01)["vert"] > 0.85


def test_christoffersen_vectorise_egal_a_la_version_du_module_risque():
    E = pu.simuler_exceptions(600, 0.03, 0.15, 8, np.random.default_rng(2))
    pof, ind, cc = pu.christoffersen_lot(E, 0.01)
    for i in range(8):
        c = rk.christoffersen(E[i].astype(int), 0.01)
        assert abs(c["lr_ind"] - ind[i]) < 1e-9 and abs(c["lr_cc"] - cc[i]) < 1e-9


def test_exceptions_markov_ont_le_bon_taux_et_la_bonne_persistance():
    E = pu.simuler_exceptions(3000, 0.02, 0.3, 400, np.random.default_rng(5))
    taux = E.mean()
    assert abs(taux - 0.02) < 4 * np.sqrt(0.02 * 0.98 / E.size * 5)           # variance gonflée par la persistance
    a, b = E[:, :-1], E[:, 1:]
    assert abs((a & b).sum() / a.sum() - 0.3) < 0.03
    assert abs(pu.pi01_pour_taux(0.02, 0.3) - 0.02 * 0.7 / 0.98) < 1e-15


def test_christoffersen_voit_les_clusters_pas_kupiec():
    h0 = pu.puissance_tests(2500, 0.01, 0.01, 1500, graine=1)
    cl = pu.puissance_tests(2500, 0.01, 0.30, 1500, graine=1)
    assert h0["ind"] < 0.08 and h0["cc"] < 0.08                    # i.i.d. : pas de fausse alerte
    assert cl["ind"] > 0.9 and cl["cc"] > 0.9                      # clusters (pi11 = 0,3) : détectés
    assert cl["kupiec"] < 0.25                                     # Kupiec ne profite pas du groupement (taux correct)


# ---------------------------------------------------------------- contributions d'Euler à l'ES
def _scenarios(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    C = np.array([[1.0, 0.6, -0.5], [0.6, 1.0, -0.3], [-0.5, -0.3, 1.0]])
    X = 0.01 * (rng.standard_normal((n, 3)) @ np.linalg.cholesky(C).T) * np.array([1.0, 2.0, 1.5])
    return X, np.array([0.5, 0.3, 0.2])


def test_poids_de_queue_somme_un_et_es_est_la_moyenne_des_pires_pertes():
    L = np.random.default_rng(1).standard_normal(1000)
    om = ce.poids_queue(L, 0.975)
    assert abs(om.sum() - 1) < 1e-12 and (om > 0).sum() == 25
    assert abs(om @ L - np.sort(L)[-25:].mean()) < 1e-12              # n (1 - alpha) = 25 exactement
    om2 = ce.poids_queue(L, 0.9755)                                     # frontière fractionnaire : 24,5 pertes
    assert abs(om2.sum() - 1) < 1e-12 and abs((om2 > 0).sum() - 25) <= 1


def test_somme_des_contributions_egale_l_es_et_theoreme_d_euler():
    X, w = _scenarios()
    r = ce.contributions_es(X, w, 1e6, 0.975)
    assert abs(r["contributions"].sum() - r["es"]) < 1e-9 * r["es"]
    assert abs(r["parts"].sum() - 1) < 1e-12
    deriv = ce.derivees_numeriques(X, w, 1e6, 0.975)                    # w_i dES/dw_i par différences finies
    assert np.max(np.abs(deriv - r["contributions"])) < 1e-6 * r["es"]
    assert abs(ce.es_homogene(X, 2.5 * w, 1e6) / ce.es_homogene(X, w, 1e6) - 2.5) < 1e-12     # homogénéité de degré 1
    assert abs(ce.es_homogene(X, w, 1e6) - r["es"]) < 1e-9 * r["es"]


def test_une_ligne_de_couverture_a_une_contribution_negative():
    X, w = _scenarios()
    r = ce.contributions_es(X, w, 1e6, 0.975)
    assert r["contributions"][2] < 0 < r["contributions"][1]          # la 3e ligne est négativement corrélée aux autres : elle couvre
    assert r["parts"][1] > w[1]                                         # la plus volatile pèse plus dans l'ES que dans le capital


@sans_donnees
def test_contributions_es_fhs_sur_les_vraies_donnees_le_9_mars_2020():
    import pandas as pd
    R = rk.rendements_log()
    w = np.array([rk.PORTEFEUILLE[a] for a in R.columns])
    t = int(np.searchsorted(R.index, pd.Timestamp("2020-03-09")))
    Rwin = R.values[t - rk.FENETRE:t]
    sigma, z = rk.etat_garch(Rwin, rk.ajuster_garch(Rwin)[0])
    X = z[np.random.default_rng(0).integers(0, z.shape[0], 100_000)] * sigma
    r = ce.contributions_es(X, w, rk.NOTIONNEL, rk.ALPHA_ES)
    assert abs(r["contributions"].sum() - r["es"]) < 1e-9 * r["es"]
    ref = rk.prevision_fhs(sigma, z, w, np.random.default_rng(0), 100_000).es975
    assert abs(r["es"] / ref - 1) < 0.01                               # même ES (règle du point milieu vs moyenne de la queue)
    j = list(R.columns)
    assert r["parts"][j.index("SPXUSD")] > w[j.index("SPXUSD")]        # l'indice actions pèse plus dans l'ES que dans le capital
