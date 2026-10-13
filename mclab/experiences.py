"""Expériences de l'approfondissement : une fonction par question (Q1 à Q8), chacune renvoie un bloc JSON-sérialisable.

Chaque bloc a la forme {question, theoreme, attendu, chiffres, conclusion, lecture (lignes du README)}.
Les CONCLUSIONS sont fabriquées à partir des nombres mesurés (f-strings) : aucune phrase ne peut contredire les chiffres.
Ce module est appelé à l'identique par scripts/run_approfondissement.py (JSON + figures) et par le notebook
notebooks/approfondissement.ipynb, donc les deux montrent les mêmes nombres (graines fixées, hors temps de calcul).
"""
import time
from dataclasses import replace

import numpy as np
from scipy import stats

from mclab import (aad, anova, contributions_es, importance, mlmc, puissance, schemas, stratification)
from mclab.analytique import asiatique_geo_prix, bs_grecques, bs_prix, heston_prix
from mclab.utils import ParamsHeston


def _l(x):
    """Convertit récursivement tableaux et scalaires numpy en types JSON natifs."""
    if isinstance(x, dict):
        return {str(k): _l(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_l(v) for v in x]
    if isinstance(x, np.ndarray):
        return _l(x.tolist())
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _fmt(x, n=2):
    return f"{x:.{n}f}".replace(".", ",")


def _ic(ic, n=2):
    return f"[{_fmt(ic[0], n)} ; {_fmt(ic[1], n)}]"


# ====================================================================================================
# Q1 — Ordres de convergence des schémas
# ====================================================================================================
def q1_ordres_convergence(n_gbm=600_000, n_cir=100_000, n_heston=40_000_000, n_heston_feller=20_000_000, graine=11):
    """Erreurs forte et faible vs dt : GBM (Euler, Milstein), CIR (Euler, Milstein), Heston (Euler-FT, QE)."""
    t0 = time.perf_counter()
    S0, mu, sigma, T, K = 100.0, 0.02, 0.2, 1.0, 100.0
    ks = list(range(2, 9))
    dts = [T / 2**k for k in ks]

    # ---- GBM : erreurs fortes (L1 et L2) et faible (call, nombres aléatoires communs)
    E = schemas.erreurs_gbm(S0, mu, sigma, T, K, ks, n_gbm, graine=graine)
    gbm = {}
    for s in ("euler", "milstein"):
        fort = schemas.pente_loglog_ic(dts, E[s]["fort"], graine=1)
        mse = schemas.pente_loglog_ic(dts, E[s]["fort_l2"], graine=2)     # MSE ~ dt^(2 p) : l'ordre L2 est la moitié de la pente
        faible = schemas.pente_loglog_ic(dts, E[s]["faible_call_cv"], graine=3)     # avec variable de contrôle S^dt - S
        faible_wls = schemas.ajuster_loi_puissance(dts, faible["erreurs"], faible["se"])
        exact_x2 = [abs(schemas.moment2_gbm_schema(s, S0, mu, sigma, T, 2**k) - schemas.moment2_gbm_exact(S0, mu, sigma, T))
                    for k in ks]
        pente_x2 = float(np.polyfit(np.log(dts), np.log(exact_x2), 1)[0])
        gbm[s] = {"fort_l1": fort, "fort_l2": {"pente": mse["pente"] / 2, "ic95": [mse["ic95"][0] / 2, mse["ic95"][1] / 2]},
                  "faible_call": {**faible, "wls": faible_wls}, "faible_x2_exact": {"pente": pente_x2, "erreurs": exact_x2}}
    pentes_locales = {s: {c: np.diff(np.log(np.abs(gbm[s][c]["erreurs"]))) / np.diff(np.log(dts))
                          for c in ("fort_l1", "faible_call")} for s in gbm}

    # ---- CIR : erreur forte, Feller violé (xi = 0,5) puis respecté (xi = 0,2)
    kappa, theta, v0 = 1.5, 0.04, 0.04
    ks_c = list(range(1, 8))
    dts_c = [T / 2**k for k in ks_c]
    cir = {}
    for nom, xi in (("feller_viole", 0.5), ("feller_respecte", 0.2)):
        err = schemas.erreurs_forte_cir(v0, kappa, theta, xi, T, ks_c, 11, n_cir, graine=graine + 5)
        cir[nom] = {"xi": xi, "ratio_feller": 2 * kappa * theta / xi**2, "dts": dts_c}
        for s in ("euler", "milstein"):
            cir[nom][s] = schemas.pente_loglog_ic(dts_c, err[s], graine=4)

    # ---- Heston : biais faible du prix d'un call, Euler full-truncation vs QE, contre le semi-fermé
    pas = [4, 8, 16, 32, 64, 128]
    dts_h = [T / n for n in pas]
    heston = {}
    for nom, p, n_ch in (("feller_viole", ParamsHeston(), n_heston), ("feller_respecte", replace(ParamsHeston(), xi=0.2), n_heston_feller)):
        exact = heston_prix(p, K, T)
        bloc = {"prix_exact": exact, "xi": p.xi, "n_chemins": n_ch, "pas": pas, "dts": dts_h}
        for sch in ("euler_ft", "qe"):
            r = [schemas.biais_faible_heston(p, K, T, sch, n, n_ch, graine + 1000 * i + (0 if sch == "euler_ft" else 77), exact)
                 for i, n in enumerate(pas)]
            biais, se = np.array([x["biais"] for x in r]), np.array([x["se"] for x in r])
            z = biais / se
            bloc[sch] = {"biais": biais, "se": se, "z": z, "chi2": float(np.sum(z**2)),
                         "p_chi2_zero": float(stats.chi2.sf(np.sum(z**2), len(pas)))}
            if sch == "euler_ft":
                bloc[sch]["wls"] = schemas.ajuster_loi_puissance(dts_h, biais, se)
                bloc[sch]["wls_grossier"] = schemas.ajuster_loi_puissance(dts_h[:4], biais[:4], se[:4])
        heston[nom] = bloc

    g, c, h = gbm, cir, heston
    hf = h["feller_viole"]
    conclusion = (
        f"GBM : erreur forte L1 d'Euler en dt^{_fmt(g['euler']['fort_l1']['pente'], 3)} {_ic(g['euler']['fort_l1']['ic95'], 3)} (théorie 1/2), "
        f"de Milstein en dt^{_fmt(g['milstein']['fort_l1']['pente'], 3)} {_ic(g['milstein']['fort_l1']['ic95'], 3)} en L1 et "
        f"dt^{_fmt(g['milstein']['fort_l2']['pente'], 3)} en L2 (théorie 1). Erreur faible (call, CRN) : Euler {_fmt(g['euler']['faible_call']['wls']['p'], 2)} "
        f"± {_fmt(1.96 * g['euler']['faible_call']['wls']['se_p'], 2)}, Milstein {_fmt(g['milstein']['faible_call']['pente'], 2)} {_ic(g['milstein']['faible_call']['ic95'], 2)} "
        f"(théorie 1 ; sans bruit, E[S^2] donne {_fmt(g['euler']['faible_x2_exact']['pente'], 3)} et {_fmt(g['milstein']['faible_x2_exact']['pente'], 3)}). "
        f"CIR : Feller respecté (2 kappa theta / xi^2 = {_fmt(c['feller_respecte']['ratio_feller'], 1)}) Euler {_fmt(c['feller_respecte']['euler']['pente'])} / "
        f"Milstein {_fmt(c['feller_respecte']['milstein']['pente'])} ; Feller violé ({_fmt(c['feller_viole']['ratio_feller'], 2)}) Euler {_fmt(c['feller_viole']['euler']['pente'])} / "
        f"Milstein {_fmt(c['feller_viole']['milstein']['pente'])} : les ordres du théorème ne tiennent que sous Feller. "
        f"Heston : biais faible Euler-FT {_fmt(hf['euler_ft']['biais'][0], 3)} (4 pas) -> {_fmt(hf['euler_ft']['biais'][-1], 3)} (128 pas) contre QE "
        f"{_fmt(hf['qe']['biais'][0], 3)} -> {_fmt(hf['qe']['biais'][-1], 3)} (erreur-type {_fmt(hf['qe']['se'][0], 3)})."
    )
    return _l({
        "question": "À quelle vitesse l'erreur de discrétisation tend-elle vers 0 pour Euler et Milstein (forte et faible), et ces ordres survivent-ils à Heston ?",
        "theoreme": "Euler : ordre fort 1/2, faible 1. Milstein (terme b b' (dW^2 - dt)/2) : ordre fort 1, faible 1 (Kloeden-Platen), pour des coefficients réguliers.",
        "attendu": "Pentes log-log : GBM Euler fort 0,5 et faible 1 ; Milstein fort 1 et faible 1. CIR : mêmes ordres SI la condition de Feller tient (sqrt(v) non lipschitzien en 0).",
        "chiffres": {"parametres_gbm": {"S0": S0, "mu": mu, "sigma": sigma, "T": T, "K": K, "n_chemins": n_gbm, "ks": ks},
                     "gbm": gbm, "pentes_locales_gbm": pentes_locales, "cir": cir, "heston": heston,
                     "parametres_cir": {"kappa": kappa, "theta": theta, "v0": v0, "k_ref": 11, "n_chemins": n_cir}},
        "conclusion": conclusion,
        "duree_s": time.perf_counter() - t0,
    })


# ====================================================================================================
# Q2 — MLMC avec Milstein
# ====================================================================================================
def q2_mlmc_milstein(n_exposants=100_000, epsilons=(0.05, 0.02, 0.01, 0.005, 0.002), graine=21):
    """Exposants alpha, beta, gamma du théorème de Giles et coût mesuré, Euler vs Milstein (GBM) ; beta de Heston."""
    t0 = time.perf_counter()
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.02, 0.2
    ech = {"euler": mlmc.echantillonneur_gbm_euler(S0, K, T, r, sigma),
           "milstein": mlmc.echantillonneur_gbm_milstein(S0, K, T, r, sigma)}
    ref = bs_prix(S0, K, T, r, sigma)
    gbm = {}
    for s, f in ech.items():
        e = mlmc.estimer_exposants(f, L=8, n=n_exposants, graine=graine, croissance=0.5)
        e["regime"] = mlmc.regime_giles(e["alpha"], e["beta"], 1.0)
        runs = mlmc.cout_vs_epsilon(f, list(epsilons), graine=graine, alpha=1.0, beta=1.0 if s == "euler" else 2.0)
        couts = np.array([r_.cout for r_ in runs])
        pente, ic = mlmc._pente_ic(np.log(1.0 / np.array(epsilons)), np.log(couts))
        pente_std, ic_std = mlmc._pente_ic(np.log(1.0 / np.array(epsilons)), np.log([r_.cout_mc_standard for r_ in runs]))
        e["cout"] = {"epsilons": list(epsilons), "cout": couts, "cout_x_eps2": couts * np.array(epsilons) ** 2,
                     "cout_mc_standard": [r_.cout_mc_standard for r_ in runs], "niveaux": [len(r_.niveaux) - 1 for r_ in runs],
                     "temps_s": [r_.resultat.temps_s for r_ in runs], "prix": [r_.resultat.prix for r_ in runs],
                     "se": [r_.resultat.erreur_type for r_ in runs], "pente_cout": pente, "ic_pente_cout": ic,
                     "pente_cout_standard": pente_std, "ic_pente_cout_standard": ic_std}
        gbm[s] = e
    for s in gbm:   # extrapolation ARITHMÉTIQUE du coût optimal 2 eps^-2 (somme_l sqrt(V_l C_l))^2 au-delà des niveaux mesurés
        v = np.array(gbm[s]["var_Y"])
        Ls = np.arange(2, 17)
        v_ext = np.array([v[l] if l < len(v) else v[-1] * 2.0 ** (-gbm[s]["beta"] * (l - len(v) + 1)) for l in Ls])
        v_all = np.concatenate([v[:2], v_ext])
        racines = np.sqrt(v_all * 2.0 ** np.arange(len(v_all)))
        gbm[s]["cout_theorique_x_eps2"] = {"L": Ls, "valeur": 2.0 * np.cumsum(racines)[Ls] ** 2}
    gain = [ce / cm for ce, cm in zip(gbm["euler"]["cout"]["cout"], gbm["milstein"]["cout"]["cout"])]
    gain_temps = [te / tm for te, tm in zip(gbm["euler"]["cout"]["temps_s"], gbm["milstein"]["cout"]["temps_s"])]
    ge, gm = gbm["euler"], gbm["milstein"]

    # ---- Heston : beta selon les deux obstacles théoriques (aire de Lévy si rho != -1 ; Feller)
    p0 = ParamsHeston()
    configs = {"A_projet (Feller violé, rho=-0,7)": p0,
               "B_aire_negligeable (Feller violé, rho=-0,999)": replace(p0, rho=-0.999),
               "C_Feller_respecte (xi=0,2, rho=-0,7)": replace(p0, xi=0.2),
               "D_les_deux (xi=0,3, rho=-0,999)": replace(p0, xi=0.3, rho=-0.999)}
    heston = {}
    for nom, p in configs.items():
        bloc = {"ratio_feller": 2 * p.kappa * p.theta / p.xi**2, "rho": p.rho, "xi": p.xi}
        for s, f in (("euler", mlmc.echantillonneur_heston_euler(p, K, T)), ("milstein", mlmc.echantillonneur_heston_milstein(p, K, T))):
            e = mlmc.estimer_exposants(f, L=8, n=n_exposants, graine=graine + 3)
            bloc[s] = {"beta": e["beta"], "ic_beta": e["ic_beta"], "var_Y": e["var_Y"]}
        heston[nom] = bloc
    hD, hA = heston["D_les_deux (xi=0,3, rho=-0,999)"], heston["A_projet (Feller violé, rho=-0,7)"]
    conclusion = (
        f"GBM (call, Lipschitz) : beta = {_fmt(ge['beta'])} {_ic(ge['ic_beta'])} pour Euler, {_fmt(gm['beta'])} {_ic(gm['ic_beta'])} pour Milstein (théorie 1 et 2) ; "
        f"alpha = {_fmt(ge['alpha'])} (Euler, niveaux exploitables : {len(ge['niveaux_alpha'])}) et {_fmt(gm['alpha'])} {_ic(gm['ic_alpha'])} (Milstein) ; gamma = 1 (nombre de pas), "
        f"{_fmt(ge['gamma_temps'])} en temps mesuré (effets de cache numpy). Giles : Euler beta ~ gamma => eps^-2 (log eps)^2, Milstein beta > gamma => eps^-2. "
        f"Coût MLMC mesuré en 1/eps^x : x = {_fmt(ge['cout']['pente_cout'])} {_ic(ge['cout']['ic_pente_cout'])} (Euler), {_fmt(gm['cout']['pente_cout'])} {_ic(gm['cout']['ic_pente_cout'])} (Milstein), "
        f"{_fmt(ge['cout']['pente_cout_standard'])} (MC standard). Gain Milstein sur Euler (coût en pas fins) : x{_fmt(gain[0], 1)} à eps={epsilons[0]} -> x{_fmt(gain[-1], 1)} à eps={epsilons[-1]} "
        f"(temps : x{_fmt(gain_temps[-1], 1)} à eps={epsilons[-1]}). Heston projet : beta = {_fmt(hA['euler']['beta'])} (Euler) / {_fmt(hA['milstein']['beta'])} (Milstein), "
        f"quasi inchangé ; il faut à la fois Feller respecté ET aire de Lévy négligeable pour atteindre {_fmt(hD['milstein']['beta'])} {_ic(hD['milstein']['ic_beta'])} (Euler : {_fmt(hD['euler']['beta'])})."
    )
    return _l({
        "question": "Milstein rend-il le MLMC en O(eps^-2) : que valent alpha, beta, gamma mesurés, et quel gain de coût ?",
        "theoreme": "Giles (2008) : si |E[P_l - P]| <= c 2^(-alpha l) avec alpha >= min(beta, gamma)/2, V_l <= c 2^(-beta l) et C_l <= c 2^(gamma l), le coût pour une RMSE eps est O(eps^-2) si beta > gamma, O(eps^-2 (log eps)^2) si beta = gamma, O(eps^-(2+(gamma-beta)/alpha)) si beta < gamma.",
        "attendu": "GBM/call : Euler beta=1=gamma (log^2), Milstein beta=2>gamma (eps^-2 pur). Heston : beta=2 exige un bruit commutatif (rho=-1) et Feller.",
        "chiffres": {"gbm": gbm, "gain_cout_milstein_sur_euler": gain, "gain_temps_milstein_sur_euler": gain_temps,
                     "heston": heston, "prix_ref_bs": ref, "n_exposants": n_exposants},
        "conclusion": conclusion,
        "duree_s": time.perf_counter() - t0,
    })


# ====================================================================================================
# Q3 — Pourquoi le pont brownien aide le QMC (dimension effective, ANOVA, Koksma-Hlawka)
# ====================================================================================================
def q3_pont_acp(ds=(16, 32, 64), ms=range(6, 15), repetitions=24, n_anova=131_072, graine=31):
    """Variance expliquée par les k premières coordonnées (Sobol fermé) et RMSE du QMC : incrémentale, pont, ACP."""
    t0 = time.perf_counter()
    ms = list(ms)
    blocs = {}
    for d in ds:
        ks = sorted({1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16} | set(range(24, d + 1, 8)) | {d})
        ks = [k for k in ks if k <= d]
        bloc = {"mc": anova.rmse_mc(d, ms, repetitions=60, graine=graine)}
        for c in anova.CONSTRUCTIONS:
            A = anova.matrice_construction(d, 1.0, c)

            def f(Z, A=A):
                return anova.payoff_asiatique(Z, A)

            idx = anova.variance_expliquee(f, d, ks, n_anova, graine + d)
            r = {"ks": ks, "indice": [idx[k][0] for k in ks], "se": [idx[k][1] for k in ks],
                 "dim_troncature_95": anova.dimension_troncature(idx, 0.95),
                 "dim_troncature_99": anova.dimension_troncature(idx, 0.99)}
            if d in (16, 64):
                S1, se1 = anova.indices_premier_ordre(f, d, 2 * n_anova, graine + 3)
                r["somme_premier_ordre"] = float(S1.sum())
                r["premier_ordre"] = S1
            r["rmse"] = anova.rmse_qmc_construction(c, d, ms, repetitions, graine)
            r["gain_rmse_vs_mc_N14"] = bloc["mc"]["rmse"][-1] / r["rmse"]["rmse"][-1]
            bloc[c] = r
        blocs[str(d)] = bloc
    b16, b64 = blocs["16"], blocs[str(max(ds))]

    def s_k(b, c, k):
        return b[c]["indice"][b[c]["ks"].index(k)]

    conclusion = (
        f"d=16 : part de variance expliquée par la 1re coordonnée {_fmt(100 * s_k(b16, 'incrementale', 1), 1)} % (incrémentale), "
        f"{_fmt(100 * s_k(b16, 'pont', 1), 1)} % (pont), {_fmt(100 * s_k(b16, 'acp', 1), 1)} % (ACP) ; par les 4 premières : "
        f"{_fmt(100 * s_k(b16, 'incrementale', 4), 1)} / {_fmt(100 * s_k(b16, 'pont', 4), 1)} / {_fmt(100 * s_k(b16, 'acp', 4), 1)} %. "
        f"Dimension de troncature à 99 % : {b16['incrementale']['dim_troncature_99']} / {b16['pont']['dim_troncature_99']} / {b16['acp']['dim_troncature_99']} "
        f"(d=64 : {b64['incrementale']['dim_troncature_99']} / {b64['pont']['dim_troncature_99']} / {b64['acp']['dim_troncature_99']}). "
        f"RMSE à N=2^14 (d=16) : MC {b16['mc']['rmse'][-1]:.1e}, incrémentale {b16['incrementale']['rmse']['rmse'][-1]:.1e}, pont {b16['pont']['rmse']['rmse'][-1]:.1e}, "
        f"ACP {b16['acp']['rmse']['rmse'][-1]:.1e}, soit des gains de x{_fmt(b16['incrementale']['gain_rmse_vs_mc_N14'], 0)} / x{_fmt(b16['pont']['gain_rmse_vs_mc_N14'], 0)} / "
        f"x{_fmt(b16['acp']['gain_rmse_vs_mc_N14'], 0)} sur le MC ; pentes log-log {_fmt(b16['mc']['pente'])} (MC), "
        f"{_fmt(b16['incrementale']['rmse']['pente'])} / {_fmt(b16['pont']['rmse']['pente'])} / {_fmt(b16['acp']['rmse']['pente'])}. "
        f"L'ordre des RMSE suit exactement l'ordre de la variance concentrée dans les premières coordonnées ; "
        f"le gain de l'ACP sur le pont vaut x{_fmt(b16['pont']['rmse']['rmse'][-1] / b16['acp']['rmse']['rmse'][-1], 1)} à d=16 et "
        f"x{_fmt(b64['pont']['rmse']['rmse'][-1] / b64['acp']['rmse']['rmse'][-1], 1)} à d={max(ds)}."
    )
    return _l({
        "question": "Pourquoi le pont brownien (et l'ACP) rendent-ils le QMC plus précis sur l'asiatique : que mesure la dimension effective ?",
        "theoreme": "Décomposition ANOVA f = somme_u f_u ; variance expliquée par {1..k} = Var(E[f | Z_1..Z_k]) / Var f (Sobol fermé). Koksma-Hlawka : |I_N - I| <= V_HK(f) D*_N, avec D*_N ~ (log N)^s / N en dimension s : une fonction presque entièrement portée par ses s premières coordonnées (s petit) est intégrée à la vitesse de la projection de dimension s.",
        "attendu": "Variance concentrée dans les premières coordonnées : ACP > pont > incrémentale ; RMSE QMC dans le même ordre, pente plus raide quand la dimension effective est petite.",
        "chiffres": {"blocs": blocs, "n_anova": n_anova, "repetitions": repetitions, "ms": ms,
                     "parametres": {"S0": 100.0, "K": 100.0, "T": 1.0, "r": 0.02, "sigma": 0.2}},
        "conclusion": conclusion,
        "duree_s": time.perf_counter() - t0,
    })


# ====================================================================================================
# Q4 — Stratification et hypercube latin
# ====================================================================================================
def q4_stratification_lhs(n=4096, repetitions=800, graine=41):
    """Facteurs de variance mesurés vs théorie (somme p_h sigma_h^2, (somme p_h sigma_h)^2, 1/(1 - somme S_j))."""
    from scipy.special import ndtri
    t0 = time.perf_counter()
    S0, T, r, sigma = 100.0, 1.0, 0.02, 0.2
    n_pilote = n // 8

    def euro(K):
        fz = importance.payoff_call(S0, K, T, r, sigma)
        return fz, (lambda u, rng: fz(ndtri(u)))

    d = 16
    A_pont = anova.matrice_construction(d, T, "pont")

    def asia_u(u, rng):                     # on stratifie Z_1 = W_T (1re coordonnée du pont) ; le reste est tiré librement
        Z = rng.standard_normal((u.size, d))
        Z[:, 0] = ndtri(u)
        return anova.payoff_asiatique(Z, A_pont)

    cas = {"europeen_K100": euro(100.0)[1], "europeen_K140": euro(140.0)[1], "asiatique_16_pont": asia_u}
    exact = {"europeen_K100": euro(100.0)[0], "europeen_K140": euro(140.0)[0]}
    strates = {}
    for nom, fu in cas.items():
        strates[nom] = {}
        for H in (4, 16, 64):
            if nom in exact:
                mu, sg = stratification.moments_strates_numeriques(exact[nom], H)
            else:
                mu, sg = stratification.moments_strates_empiriques(fu, H, 2_000_000, np.random.default_rng(graine))
            th = stratification.variances_theoriques(mu, sg)
            rng = np.random.default_rng(np.random.SeedSequence([graine, H, len(nom)]))
            v_naif = np.var([fu(rng.random(n), rng).mean() for _ in range(repetitions)], ddof=1)
            v_prop = np.var([stratification.estimer_stratifie(fu, H, n, "proportionnelle", rng)[0] for _ in range(repetitions)], ddof=1)
            v_pil = np.var([stratification.estimer_stratifie(fu, H, n, "neyman", rng, n_pilote=n_pilote)[0] for _ in range(repetitions)], ddof=1)
            v_ora = np.var([stratification.estimer_stratifie(fu, H, n, "neyman_oracle", rng, sigmas=sg)[0] for _ in range(repetitions)], ddof=1)
            n_main = n - H * max(2, n_pilote // H)
            strates[nom][str(H)] = {
                "theorie": {"var_par_point": th, "facteur_prop": th["naif"] / th["proportionnelle"],
                            "facteur_neyman_oracle": th["naif"] / th["neyman"],
                            "facteur_neyman_pilote": th["naif"] / th["neyman"] * n_main / n},
                "mesure": {"facteur_prop": v_naif / v_prop, "facteur_neyman_pilote": v_naif / v_pil,
                           "facteur_neyman_oracle": v_naif / v_ora, "n_fois_var_naif": n * v_naif},
                "sigmas_h": sg, "mu_h": mu}
    # ---- hypercube latin
    lhs = {}
    for nom, c in (("asiatique_16_incrementale", "incrementale"), ("asiatique_16_pont", "pont")):
        A = anova.matrice_construction(d, T, c)

        def fZ(Z, A=A):
            return anova.payoff_asiatique(Z, A)

        S1, se1 = anova.indices_premier_ordre(fZ, d, 524_288, graine + 3)
        bloc = {"somme_S1": float(S1.sum()), "gain_asymptotique_theorique": float(1.0 / (1.0 - S1.sum())), "n": {}}
        for nn in (256, 4096):
            vl, _ = stratification.variance_estimateur(lambda rng: stratification.estimation_lhs(fZ, nn, d, rng), repetitions, graine + 7)
            vm, _ = stratification.variance_estimateur(lambda rng: float(np.mean(fZ(rng.standard_normal((nn, d))))), repetitions, graine + 8)
            bloc["n"][str(nn)] = {"gain_mesure": vm / vl, "var_lhs": vl, "var_mc": vm}
        lhs[nom] = bloc
    f_eur = euro(100.0)[0]
    eur = {}
    for nn in (64, 256, 1024, 4096):
        vl, _ = stratification.variance_estimateur(lambda rng: stratification.estimation_lhs(lambda Z: f_eur(Z[:, 0]), nn, 1, rng), repetitions, graine + 1)
        vm, _ = stratification.variance_estimateur(lambda rng: float(np.mean(f_eur(rng.standard_normal(nn)))), repetitions, graine + 2)
        eur[str(nn)] = {"var_lhs": vl, "var_mc": vm, "gain": vm / vl}
    ns = np.array([64, 256, 1024, 4096])
    pente_lhs = float(np.polyfit(np.log(ns), np.log([eur[str(k)]["var_lhs"] for k in ns]), 1)[0])
    pente_mc = float(np.polyfit(np.log(ns), np.log([eur[str(k)]["var_mc"] for k in ns]), 1)[0])
    lhs["europeen_d1"] = {"par_n": eur, "pente_var_lhs": pente_lhs, "pente_var_mc": pente_mc}

    sa, se_ = strates["asiatique_16_pont"], strates["europeen_K140"]
    sk100 = strates["europeen_K100"]
    la, lb = lhs["asiatique_16_incrementale"], lhs["asiatique_16_pont"]
    conclusion = (
        f"Européen ATM, H=64 strates équiprobables (n={n}) : facteur de variance mesuré x{_fmt(sk100['64']['mesure']['facteur_prop'], 0)} (proportionnelle) contre x{_fmt(sk100['64']['theorie']['facteur_prop'], 0)} "
        f"prédit par naïf/(somme p_h sigma_h^2) ; Neyman avec sigma_h connus x{_fmt(sk100['64']['mesure']['facteur_neyman_oracle'], 0)} (théorie x{_fmt(sk100['64']['theorie']['facteur_neyman_oracle'], 0)}). "
        f"Call OTM K=140, H=64 : proportionnelle x{_fmt(se_['64']['mesure']['facteur_prop'], 1)}, Neyman x{_fmt(se_['64']['mesure']['facteur_neyman_oracle'], 0)} (oracle) / x{_fmt(se_['64']['mesure']['facteur_neyman_pilote'], 0)} (pilote n/8) : "
        f"Neyman gagne dès que les sigma_h sont très inégaux. Asiatique (Z_1 du pont stratifiée), H=64 : x{_fmt(sa['64']['mesure']['facteur_prop'], 1)} / Neyman pilote x{_fmt(sa['64']['mesure']['facteur_neyman_pilote'], 1)} / oracle "
        f"x{_fmt(sa['64']['mesure']['facteur_neyman_oracle'], 1)} (théorie {_fmt(sa['64']['theorie']['facteur_prop'], 1)} / {_fmt(sa['64']['theorie']['facteur_neyman_oracle'], 1)}) : avec 8 points par strate le pilote estime mal les sigma_h. "
        f"LHS (Stein) sur l'asiatique à 16 dates : somme des indices d'ordre 1 = {_fmt(la['somme_S1'], 3)} (incrémentale) et {_fmt(lb['somme_S1'], 3)} (pont) => gain asymptotique prédit "
        f"x{_fmt(la['gain_asymptotique_theorique'], 1)} et x{_fmt(lb['gain_asymptotique_theorique'], 1)} ; mesuré à n=4096 : x{_fmt(la['n']['4096']['gain_mesure'], 1)} et x{_fmt(lb['n']['4096']['gain_mesure'], 1)}. "
        f"En dimension 1 le LHS est une stratification à n strates : variance en n^{_fmt(pente_lhs, 2)} contre n^{_fmt(pente_mc, 2)} pour le MC."
    )
    return _l({
        "question": "Stratifier le brownien terminal (proportionnel vs Neyman) ou utiliser un hypercube latin : combien de variance gagne-t-on, et la théorie le prédit-elle ?",
        "theoreme": "Stratification : Var = somme p_h^2 sigma_h^2 / n_h ; proportionnelle -> (1/n) somme p_h sigma_h^2 <= sigma^2/n (variance totale) ; Neyman n_h ~ p_h sigma_h -> (1/n) (somme p_h sigma_h)^2 <= proportionnelle (Cauchy-Schwarz). LHS (Stein 1987) : variance asymptotique (1/n)(sigma^2 - somme_j sigma_j^2), il ne retire que la partie additive.",
        "attendu": "Facteurs mesurés = facteurs théoriques à ~8 % près (600 répétitions) ; Neyman > proportionnelle surtout si les sigma_h sont inégaux ; gain LHS = 1/(1 - somme S_j).",
        "chiffres": {"n": n, "repetitions": repetitions, "n_pilote_neyman": n_pilote, "strates": strates, "lhs": lhs,
                     "precision_relative_facteur": float(np.sqrt(2.0 / (repetitions - 1)) * np.sqrt(2.0))},
        "conclusion": conclusion,
        "duree_s": time.perf_counter() - t0,
    })


# ====================================================================================================
# Q5 — Importance sampling optimal
# ====================================================================================================
def q5_importance_sampling(n_emp=400_000, graine=51, k_defauts=20):
    """Dérive optimale par second moment, par entropie croisée, et efficacité logarithmique (grandes déviations)."""
    t0 = time.perf_counter()
    S0, K, T, r, sigma = 100.0, 180.0, 1.0, 0.02, 0.2
    cas = {}
    # ---- call très hors de la monnaie
    f = importance.payoff_call(S0, K, T, r, sigma)
    zk = importance.z_strike(S0, K, T, r, sigma)
    kw = dict(a=zk, b=20.0)
    I = importance.esperance(f, **kw)
    mu_star = importance.mu_optimal(f, bracket=(zk - 1.0, zk + 2.0), **kw)
    mu_ce = importance.mu_entropie_croisee_exact(f, **kw)
    cas["call_K180"] = {"f": f, "kw": kw, "I": I, "bs": bs_prix(S0, K, T, r, sigma), "mus_cles": {"naif": 0.0, "strike": zk, "mu_star": mu_star, "mu_ce": mu_ce},
                        "grille": np.linspace(0.0, 6.0, 61), "signe": 1.0, "libelle": f"call K={K:.0f}"}
    # ---- probabilité de perte extrême d'un portefeuille de crédit (Vasicek)
    fc = importance.perte_portefeuille_credit(100, 0.01, 0.2, k_defauts)
    Ic = importance.esperance(fc)
    mu_star_c = importance.mu_optimal(fc, bracket=(-5.0, 0.0))
    mu_ce_c = importance.mu_entropie_croisee_exact(fc)
    cas["credit"] = {"f": fc, "kw": {}, "I": Ic, "bs": None, "mus_cles": {"naif": 0.0, "mu_star": mu_star_c, "mu_ce": mu_ce_c},
                     "grille": np.linspace(-6.0, 0.0, 61), "signe": -1.0,
                     "libelle": f"P(au moins {k_defauts} défauts / 100), PD 1 %, rho 20 %"}
    sortie = {}
    for nom, c in cas.items():
        v0 = importance.variance_is(c["f"], 0.0, c["I"], **c["kw"])
        courbe = [importance.variance_is(c["f"], m, c["I"], **c["kw"]) for m in c["grille"]]
        points = {}
        for lib, m in c["mus_cles"].items():
            vth = importance.variance_is(c["f"], m, c["I"], **c["kw"])
            est, vemp = importance.estimer_is(c["f"], m, n_emp, graine + len(lib))
            points[lib] = {"mu": m, "variance_theorique": vth, "variance_empirique": vemp, "facteur_variance": v0 / vth,
                           "estimation": est, "erreur_relative_par_tirage": np.sqrt(vth) / c["I"]}
        trajets = [importance.entropie_croisee(c["f"], n_iter=8, n=20_000, graine=graine + s, mu0=0.0) for s in range(10)]
        mu_hat = np.array([t[-1] for t in trajets])
        facteur_hat = [v0 / importance.variance_is(c["f"], m, c["I"], **c["kw"]) for m in mu_hat]
        sortie[nom] = {"libelle": c["libelle"], "valeur_exacte": c["I"], "reference_bs": c["bs"], "variance_naive_par_tirage": v0,
                       "grille_mu": c["grille"], "variance_grille": courbe, "points": points,
                       "ce_trajets": trajets, "ce_mu_final": mu_hat, "ce_facteur_variance": facteur_hat}
    log_eff = importance.efficacite_log_evenement([2, 4, 6, 8, 10, 20])
    pc, pk = sortie["call_K180"]["points"], sortie["credit"]["points"]
    ce_c, ce_k = sortie["credit"], sortie["call_K180"]
    conclusion = (
        f"Call K=180 (prix BS {sortie['call_K180']['valeur_exacte']:.5f}, z_K = {_fmt(zk, 3)}) : le second moment M(mu) est convexe, minimum en mu* = {_fmt(mu_star, 3)} "
        f"(facteur de variance x{_fmt(pc['mu_star']['facteur_variance'], 0)}) ; l'ancienne dérive mu = z_K donne x{_fmt(pc['strike']['facteur_variance'], 0)}, l'entropie croisée exacte "
        f"mu_CE = {_fmt(mu_ce, 3)} donne x{_fmt(pc['mu_ce']['facteur_variance'], 0)} ; estimée par 8 itérations de 20 000 tirages : mu = {_fmt(ce_k['ce_mu_final'].mean(), 2)} ± "
        f"{_fmt(ce_k['ce_mu_final'].std(ddof=1), 2)} (10 graines), facteur moyen x{_fmt(np.mean(ce_k['ce_facteur_variance']), 0)}. "
        f"Défaut rare (P = {sortie['credit']['valeur_exacte']:.2e}) : mu* = {_fmt(mu_star_c, 3)}, mu_CE = {_fmt(mu_ce_c, 3)}, facteur x{_fmt(pk['mu_star']['facteur_variance'], 0)} (x{_fmt(pk['mu_ce']['facteur_variance'], 0)} pour CE). "
        f"Variances empiriques (n={n_emp}) = théoriques à {_fmt(100 * max(abs(p_['variance_empirique'] / p_['variance_theorique'] - 1) for p_ in list(pc.values()) + list(pk.values())), 1)} % près au pire. "
        f"Grandes déviations, événement {{Z > a}} : ln M(mu*) / (2 ln P) = {_fmt(log_eff[0]['rapport_log'], 3)} (a=2), {_fmt(log_eff[2]['rapport_log'], 3)} (a=6), {_fmt(log_eff[-1]['rapport_log'], 3)} (a=20) -> 1 : "
        f"erreur relative de l'IS {_fmt(log_eff[2]['erreur_relative_is'], 1)} à a=6 (naïve : {log_eff[2]['erreur_relative_naive']:.0f}) et seulement {_fmt(log_eff[-1]['erreur_relative_is'], 1)} à a=20."
    )
    return _l({
        "question": "Quelle dérive de Girsanov minimise la variance d'un estimateur d'événement rare, et l'entropie croisée la retrouve-t-elle ?",
        "theoreme": "Pour Y ~ N(mu,1) et w = exp(-mu Y + mu^2/2), Var = M(mu) - I^2, M(mu) = E[f(Z)^2 exp(-mu Z + mu^2/2)] convexe : minimum unique mu*. Entropie croisée : mu_CE = E[f(Z) Z] / E[f(Z)]. Grandes déviations : pour {Z > a}, M(mu) = e^{mu^2} P(Z > a + mu), mu* ~ a, ln M(mu*) ~ 2 ln P (efficacité logarithmique).",
        "attendu": "mu* proche mais différent de z_K ; CE proche de mu* ; ln M(mu*) / (2 ln P) -> 1 quand a croît ; variance empirique = M(mu) - I^2.",
        "chiffres": {"cas": sortie, "efficacite_logarithmique": log_eff, "n_empirique": n_emp, "k_defauts": k_defauts,
                     "parametres_call": {"S0": S0, "K": K, "T": T, "r": r, "sigma": sigma, "z_K": zk}},
        "conclusion": conclusion,
        "duree_s": time.perf_counter() - t0,
    })


# ====================================================================================================
# Q6 — Grecques par AAD (mode inverse)
# ====================================================================================================
def q6_aad(n_verif=400_000, n_cout=40_000, ds=(8, 16, 32, 64), graine=61):
    """AAD codée à la main : exactitude vs pathwise / analytique, et coût relatif au prix (principe de Griewank)."""
    t0 = time.perf_counter()
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.02, 0.2
    rng = np.random.default_rng(graine)
    # ---- exactitude sur les MÊMES chemins : AAD (boucle, vectorisé) = pathwise à l'arrondi près
    d = 12
    Z = rng.standard_normal((n_verif, d))
    exact = {}
    for pay in aad.PAYOFFS:
        a, b = aad.aad_boucle(pay, S0, K, T, r, sigma, Z), aad.aad_vectorise(pay, S0, K, T, r, sigma, Z)
        p, bp = aad.pathwise(pay, S0, K, T, r, sigma, Z), aad.bump(pay, S0, K, T, r, sigma, Z)
        ecart = lambda x, y: float(np.max(np.abs(x - y)))
        exact[pay] = {
            "max_abs_aad_boucle_vs_vectorise": max(ecart(a[c], b[c]) for c in ("prix", "delta", "rho", "dK")),
            "max_abs_aad_vs_pathwise": {"delta": ecart(a["delta"], p["delta"]), "vega": ecart(a["vega_seaux"].sum(1), p["vega"]),
                                        "rho": ecart(a["rho"], p["rho"]), "dK": ecart(a["dK"], p["dK"])},
            "max_valeur_absolue_grecque": {"delta": float(np.abs(p["delta"]).max()), "vega": float(np.abs(p["vega"]).max()),
                                           "rho": float(np.abs(p["rho"]).max())},
            "ecart_moyenne_vs_bump_crn_en_se": {c: float((a[k].mean() - bp[c].mean()) / (a[k].std(ddof=1) / np.sqrt(n_verif)))
                                                 for c, k in (("delta", "delta"), ("rho", "rho"))}
            | {"vega": float((a["vega_seaux"].sum(1).mean() - bp["vega"].mean()) / (a["vega_seaux"].sum(1).std(ddof=1) / np.sqrt(n_verif)))},
            "grecques_mc": {"delta": a["delta"].mean(), "vega": a["vega_seaux"].sum(1).mean(), "rho": a["rho"].mean(), "dK": a["dK"].mean(),
                            "se_delta": a["delta"].std(ddof=1) / np.sqrt(n_verif)}}
    # européen vs Black-Scholes (exact), géométrique vs différences finies de la formule fermée
    Z1 = rng.standard_normal((2_000_000, 1))
    a = aad.aad_vectorise("europeen", S0, K, T, r, sigma, Z1)
    g = bs_grecques(S0, K, T, r, sigma)
    vs_bs = {}
    for nom, x, ref in (("delta", a["delta"], g["delta"]), ("vega", a["vega_seaux"][:, 0], g["vega"]), ("rho", a["rho"], g["rho"])):
        se = x.std(ddof=1) / np.sqrt(x.size)
        vs_bs[nom] = {"aad": float(x.mean()), "se": float(se), "bs": float(ref), "z": float((x.mean() - ref) / se)}
    Zg = rng.standard_normal((2_000_000, d))
    ag = aad.aad_vectorise("asiatique_geo", S0, K, T, r, sigma, Zg)
    h = 1e-4
    G = lambda **kw: asiatique_geo_prix(**{**dict(S0=S0, K=K, T=T, r=r, sigma=sigma, n_dates=d), **kw})
    ref_geo = {"delta": (G(S0=S0 + h) - G(S0=S0 - h)) / (2 * h), "vega": (G(sigma=sigma + h) - G(sigma=sigma - h)) / (2 * h),
               "rho": (G(r=r + h) - G(r=r - h)) / (2 * h)}
    vs_geo = {}
    for nom, x in (("delta", ag["delta"]), ("vega", ag["vega_seaux"].sum(1)), ("rho", ag["rho"])):
        se = x.std(ddof=1) / np.sqrt(x.size)
        vs_geo[nom] = {"aad": float(x.mean()), "se": float(se), "reference": float(ref_geo[nom]), "z": float((x.mean() - ref_geo[nom]) / se)}
    # ---- coût relatif au prix, en fonction du nombre de grecques m = d + 3 (delta, rho, K et un vega par date)
    cout = []
    for dd in ds:
        Zc = rng.standard_normal((n_cout, dd))
        sv = np.full(dd, sigma)
        tp = aad.chronometrer(lambda: aad.valeur("asiatique", S0, K, T, r, sv, Zc), 5)
        tl = aad.chronometrer(lambda: aad.aad_boucle("asiatique", S0, K, T, r, sv, Zc), 5)
        tv = aad.chronometrer(lambda: aad.aad_vectorise("asiatique", S0, K, T, r, sv, Zc), 5)
        tb = aad.chronometrer(lambda: aad.bump("asiatique", S0, K, T, r, sigma, Zc), 3)
        ts = aad.chronometrer(lambda: aad.bump_seaux("asiatique", S0, K, T, r, sv, Zc), 2)
        cout.append({"d": dd, "m_grecques": dd + 3, "temps_prix_s": tp, "aad_boucle": tl / tp, "aad_vectorise": tv / tp,
                     "bump_4_grecques": tb / tp, "bump_seaux": ts / tp, "reevaluations_bump_seaux": 2 * dd + 1,
                     "acceleration_aad_vectorise_vs_bump_seaux": ts / tv, "acceleration_aad_boucle_vs_bump_seaux": ts / tl})
    last = cout[-1]
    worst = max(max(v["max_abs_aad_vs_pathwise"].values()) for v in exact.values())
    conclusion = (
        f"Sur les mêmes chemins (n={n_verif}, 12 dates), l'AAD en mode inverse égale le pathwise écrit à la main à {worst:.1e} près en valeur absolue "
        f"(grecques de l'ordre de 10 à 500 ; boucle et version vectorisée égales à {max(v['max_abs_aad_boucle_vs_vectorise'] for v in exact.values()):.1e}). "
        f"Contre l'analytique : européen delta z = {_fmt(vs_bs['delta']['z'])}, vega z = {_fmt(vs_bs['vega']['z'])}, rho z = {_fmt(vs_bs['rho']['z'])} ; "
        f"asiatique géométrique (différences finies de la formule fermée) delta z = {_fmt(vs_geo['delta']['z'])}, vega z = {_fmt(vs_geo['vega']['z'])}, rho z = {_fmt(vs_geo['rho']['z'])}. "
        f"Coût relatif au prix (asiatique, {last['d']} dates, {last['m_grecques']} grecques : delta, rho, strike et {last['d']} vegas par date) : AAD {_fmt(last['aad_boucle'], 1)}x (boucle) / "
        f"{_fmt(last['aad_vectorise'], 1)}x (vectorisé) contre {_fmt(last['bump_seaux'], 0)}x pour le bump-and-revalue des seuls vegas ({last['reevaluations_bump_seaux']} réévaluations), "
        f"soit une accélération de x{_fmt(last['acceleration_aad_vectorise_vs_bump_seaux'], 0)} (vectorisé) et x{_fmt(last['acceleration_aad_boucle_vs_bump_seaux'], 0)} (boucle). "
        f"Le rapport AAD/prix reste dans {_fmt(min(c['aad_boucle'] for c in cout), 1)}-{_fmt(max(c['aad_boucle'] for c in cout), 1)}x (boucle) quand m passe de {cout[0]['m_grecques']} à {last['m_grecques']} "
        f"(Griewank : 3 à 4 en comptage d'opérations ; les temps numpy dépendent de la machine et du cache)."
    )
    return _l({
        "question": "Peut-on obtenir toutes les grecques d'un payoff pour ~3 à 4 fois le coût du prix, et sont-elles exactes ?",
        "theoreme": "Principe du gradient bon marché (Griewank) : TEMPS{F, grad F} <= omega TEMPS{F}, omega ~ 3-4, indépendant du nombre m de paramètres, par propagation d'adjoints bar x = dP/dx en sens inverse du graphe. Le bump-and-revalue coûte 2 m réévaluations.",
        "attendu": "AAD = pathwise à l'arrondi (1e-12) ; coût AAD/prix ~ constant en m ; coût bump ~ 2 m.",
        "chiffres": {"exactitude": exact, "vs_black_scholes_europeen": vs_bs, "vs_geometrique_ferme": vs_geo, "cout": cout,
                     "n_verif": n_verif, "n_cout": n_cout},
        "conclusion": conclusion,
        "duree_s": time.perf_counter() - t0,
    })


# ====================================================================================================
# Q7 — Risque : puissance des backtests et contributions d'Euler à l'ES
# ====================================================================================================
def q7_risque(n_mc=200_000, n_sim_markov=4000, pas_dates=60, n_sim_es=400_000, n_sim_es_moyenne=100_000, graine=71):
    """Courbe de puissance de Kupiec/Christoffersen, puis allocation d'Euler de l'ES FHS (9 mars 2020 et moyenne)."""
    t0 = time.perf_counter()
    horizons = [250, 500, 1250, 2500]
    p1 = 0.02
    # ---- Kupiec : exact vs Monte Carlo, courbe complète
    Ts = np.arange(50, 5001, 10)
    puiss = puissance.courbe_puissance_kupiec(Ts, p1)
    kup = {"T": Ts, "puissance_exacte": puiss, "T_80": puissance.horizon_puissance(Ts, puiss, 0.8),
           "T_90": puissance.horizon_puissance(Ts, puiss, 0.9),
           "T_80_formule_normale": puissance.taille_echantillon_normale(puissance.P0, p1), "points": {}}
    for T in horizons:
        ex = puissance.rejet_kupiec_exact(T, p1)
        mc, se = puissance.rejet_kupiec_mc(T, p1, n_mc, graine + T)
        taille = puissance.rejet_kupiec_exact(T, puissance.P0)
        kup["points"][str(T)] = {"annees": T / 250, "puissance_exacte": ex, "puissance_mc": mc, "se_mc": se, "z": (mc - ex) / se, "taille_reelle": taille}
    kup["zones_bale_250j_taux_2pct"] = puissance.proba_zone_bale(p1, 250)
    # ---- Christoffersen : exceptions groupées de taux 1 % (Kupiec aveugle au groupement)
    scenarios = {"H0 : i.i.d. 1 %": (0.01, 0.01), "i.i.d. 2 %": (0.02, 0.02), "groupées 1 %, pi11 = 0,10": (0.01, 0.10),
                 "groupées 1 %, pi11 = 0,20": (0.01, 0.20), "groupées 1 %, pi11 = 0,30": (0.01, 0.30)}
    chri = {}
    for nom, (p, pi11) in scenarios.items():
        chri[nom] = {str(T): puissance.puissance_tests(T, p, pi11, n_sim_markov, graine + T) for T in horizons}
    # ---- contributions d'Euler à l'ES (portefeuille de risque, FHS)
    try:
        es = _contributions_es_fhs(pas_dates, n_sim_es, n_sim_es_moyenne, graine)
    except FileNotFoundError as e:   # données de marché absentes : on n'invente rien
        es = {"donnees_absentes": str(e)}
    pk = kup["points"]
    cc = chri["groupées 1 %, pi11 = 0,20"]
    conclusion = (
        f"Kupiec, modèle à 2 % d'exceptions au lieu de 1 % : puissance {_fmt(100 * pk['250']['puissance_exacte'], 0)} % à 1 an, {_fmt(100 * pk['500']['puissance_exacte'], 0)} % à 2 ans, "
        f"{_fmt(100 * pk['1250']['puissance_exacte'], 0)} % à 5 ans, {_fmt(100 * pk['2500']['puissance_exacte'], 0)} % à 10 ans (énumération exacte ; Monte Carlo à {_fmt(max(abs(v['z']) for v in pk.values()), 1)} erreurs-types au pire). "
        f"Il faut {kup['T_80']} jours ({_fmt(kup['T_80'] / 250, 1)} ans) pour 80 % de puissance durablement (formule normale : {_fmt(kup['T_80_formule_normale'] / 250, 1)} ans). "
        f"La taille réelle du test n'est pas 5 % : {_fmt(100 * pk['250']['taille_reelle'], 1)} % à 1 an, {_fmt(100 * pk['2500']['taille_reelle'], 1)} % à 10 ans (statistique discrète, chi2 asymptotique). "
        f"Bâle : un modèle à 2 % est en zone verte {_fmt(100 * kup['zones_bale_250j_taux_2pct']['vert'], 0)} % du temps sur 250 jours. "
        f"Exceptions groupées de taux correct 1 % (pi11 = 0,20) : Kupiec rejette {_fmt(100 * cc['2500']['kupiec'], 0)} % à 10 ans (déjà au-dessus de sa taille ; il ne voit pas le groupement), "
        f"Christoffersen (indépendance) {_fmt(100 * cc['250']['ind'], 0)} % / {_fmt(100 * cc['500']['ind'], 0)} % / {_fmt(100 * cc['1250']['ind'], 0)} % / {_fmt(100 * cc['2500']['ind'], 0)} % à 1/2/5/10 ans."
    )
    if "donnees_absentes" not in es:
        e9, em = es["jour_2020_03_09"], es["moyenne"]
        es_txt = f"{e9['es']:,.0f}".replace(",", " ")
        conclusion += (
            f" ES FHS 97,5 % au 9 mars 2020 : {es_txt} EUR ; somme des contributions d'Euler = ES à {e9['ecart_relatif_somme']:.1e} près, "
            f"dérivées numériques = contributions à {e9['ecart_relatif_euler_derivees']:.1e} près, homogénéité de degré 1 vérifiée ({e9['rapport_homogeneite']:.6f} pour lambda=2). "
            f"SPX pèse {_fmt(100 * e9['poids']['SPXUSD'], 0)} % du capital et {_fmt(100 * e9['parts']['SPXUSD'], 0)} % de l'ES ; en moyenne sur {em['n_dates']} dates : "
            f"{_fmt(100 * em['parts']['SPXUSD'], 0)} % ; XAUUSD (couverture) {_fmt(100 * e9['parts']['XAUUSD'], 0)} % ce jour-là."
        )
    return _l({
        "question": "Combien d'années de backtest faut-il pour détecter un modèle de VaR à 2 % d'exceptions au lieu de 1 %, et d'où vient l'ES du portefeuille ligne par ligne ?",
        "theoreme": "Puissance = P(LR > chi2_{1-a} | H1) ; sous H1 le nombre d'exceptions est Binomial(T, p1). Théorème d'Euler : F homogène de degré 1 => F(w) = somme_i w_i dF/dw_i ; pour l'ES, w_i dES/dw_i = E[perte de la ligne i | L >= VaR] et la somme des contributions vaut l'ES.",
        "attendu": "Puissance Kupiec ~ 25 % à 1 an, > 80 % vers 4-5 ans ; Christoffersen voit les clusters que Kupiec ne voit pas ; somme des contributions = ES à la précision machine.",
        "chiffres": {"kupiec": kup, "christoffersen": chri, "es_euler": es, "n_mc": n_mc, "n_sim_markov": n_sim_markov},
        "conclusion": conclusion,
        "duree_s": time.perf_counter() - t0,
    })


def _contributions_es_fhs(pas_dates, n_sim_jour, n_sim_moyenne, graine):
    """Contributions d'Euler de l'ES FHS (GARCH(1,1)-t refit à la date, bootstrap des résidus joints) : 9 mars 2020 + moyenne."""
    import pandas as pd
    from mclab import risque as rk
    R = rk.rendements_log()
    actifs = list(R.columns)
    w = np.array([rk.PORTEFEUILLE[a] for a in actifs])
    Rv = R.values

    def contributions_a(t, n_sim, precedents, seed):
        Rwin = Rv[t - rk.FENETRE:t]
        params, _ = rk.ajuster_garch(Rwin, precedents)
        sigma, z = rk.etat_garch(Rwin, params)
        X = z[np.random.default_rng(seed).integers(0, z.shape[0], n_sim)] * sigma
        return contributions_es.contributions_es(X, w, rk.NOTIONNEL, rk.ALPHA_ES), params, X, sigma

    t9 = int(np.searchsorted(R.index, pd.Timestamp("2020-03-09")))
    res, _, X, sigma9 = contributions_a(t9, n_sim_jour, None, graine)
    deriv = contributions_es.derivees_numeriques(X, w, rk.NOTIONNEL, rk.ALPHA_ES)
    lam = contributions_es.es_homogene(X, 2.0 * w, rk.NOTIONNEL, rk.ALPHA_ES) / contributions_es.es_homogene(X, w, rk.NOTIONNEL, rk.ALPHA_ES)
    perte_reelle = -rk.NOTIONNEL * np.expm1(Rv[t9]) * w
    jour = {"date": str(R.index[t9].date()), "es": res["es"],
            "contributions": dict(zip(actifs, res["contributions"])), "parts": dict(zip(actifs, res["parts"])),
            "poids": dict(zip(actifs, w)),
            "ecart_relatif_somme": abs(res["contributions"].sum() - res["es"]) / res["es"],
            "ecart_relatif_euler_derivees": float(np.max(np.abs(deriv - res["contributions"])) / res["es"]),
            "rapport_homogeneite": lam,
            "perte_reelle_ligne": dict(zip(actifs, perte_reelle)), "perte_reelle_totale": float(perte_reelle.sum()),
            "sigma_prevu_pct": dict(zip(actifs, 100.0 * sigma9))}
    # moyenne : une date tous les `pas_dates` jours, GARCH refit à chaque date (comme le backtest refit tous les 20 jours)
    dates = list(range(rk.FENETRE, len(Rv), pas_dates))
    parts, es_liste, precedents = [], [], None
    for i, t in enumerate(dates):
        r, precedents, _, _ = contributions_a(t, n_sim_moyenne, precedents, graine + 1 + i)
        parts.append(r["parts"])
        es_liste.append(r["es"])
    parts = np.array(parts)
    moyenne = {"n_dates": len(dates), "pas_jours": pas_dates, "parts": dict(zip(actifs, parts.mean(0))),
               "ecart_type_parts": dict(zip(actifs, parts.std(0, ddof=1))), "es_moyen": float(np.mean(es_liste)),
               "somme_parts_min": float(parts.sum(1).min()), "somme_parts_max": float(parts.sum(1).max()),
               "poids": dict(zip(actifs, w))}
    return {"jour_2020_03_09": jour, "moyenne": moyenne, "actifs": actifs}


# ====================================================================================================
# Q8 — Condition de Feller (bonus)
# ====================================================================================================
def q8_feller(n_chemins=200_000, pas=(8, 16, 32, 64, 128, 256), graine=81):
    """Fraction de pas tronqués (v < 0) de l'Euler full-truncation vs ratio de Feller F = 2 kappa theta / xi^2."""
    t0 = time.perf_counter()
    kappa, theta, T = 1.5, 0.04, 1.0
    blocs = {}
    for F in (0.25, 0.5, 1.0, 2.0, 4.0):
        xi = float(np.sqrt(2.0 * kappa * theta / F))
        fr = np.array([schemas.fraction_pas_tronques(kappa, theta, xi, T, n, n_chemins, graine + n) for n in pas])
        comptes = fr * n_chemins * np.array(pas)
        ok = comptes >= 100                          # on ne régresse que là où l'on a vu au moins 100 pas tronqués
        dts = T / np.array(pas)
        ajust, cov = np.polyfit(np.log(dts[ok]), np.log(fr[ok]), 1, w=np.sqrt(comptes[ok]), cov="unscaled")
        se = float(np.sqrt(cov[0, 0]))
        # biais de la moyenne : E[v_T] = theta sous la loi stationnaire ; Euler-FT renvoie v^+
        biais = {}
        rng = np.random.default_rng(graine + 99)
        for n in (16, 64):
            v0 = rng.gamma(F, xi**2 / (2.0 * kappa), 100_000)
            dW = np.sqrt(T / n) * rng.standard_normal((v0.size, n))
            vt, _ = schemas.cir_terminal("euler", v0, kappa, theta, xi, T, dW)
            biais[str(n)] = {"biais_relatif": float(vt.mean() / theta - 1.0), "se_relatif": float(vt.std(ddof=1) / np.sqrt(vt.size) / theta)}
        blocs[str(F)] = {"F": F, "xi": xi, "dts": dts, "fraction_tronquee": fr, "comptes": comptes, "pente": float(ajust[0]),
                         "ic95_pente": [float(ajust[0] - 1.96 * se), float(ajust[0] + 1.96 * se)], "points_utilises": int(ok.sum()),
                         "biais_moyenne": biais}
    b = blocs
    conclusion = (
        "Part des pas où la troncature agit, en dt^pente : "
        + " ; ".join(f"F = {_fmt(v['F'])} -> pente {_fmt(v['pente'])} {_ic(v['ic95_pente'])}" for v in b.values())
        + f". La pente suit F (2 kappa theta / xi^2) : P(v_n < c xi^2 dt) ~ dt^F car la densité stationnaire Gamma(F) se comporte en v^(F-1) près de 0. "
        + f"À 16 pas la troncature touche {_fmt(100 * b['0.25']['fraction_tronquee'][1], 0)} % des pas pour F=0,25 et {_fmt(100 * b['0.5']['fraction_tronquee'][1], 0)} % pour F=0,5 (cas du projet : 0,48), "
        + f"contre {_fmt(100 * b['4.0']['fraction_tronquee'][1], 3)} % pour F=4. Biais relatif de E[v_T^+] à 16 pas : {_fmt(100 * b['0.5']['biais_moyenne']['16']['biais_relatif'], 1)} % (F=0,5) contre "
        + f"{_fmt(100 * b['4.0']['biais_moyenne']['16']['biais_relatif'], 2)} % (F=4)."
    )
    return _l({
        "question": "Que vaut la condition de Feller pour la discrétisation : quelle fraction de pas est tronquée selon 2 kappa theta / xi^2 ?",
        "theoreme": "CIR : le bord 0 est inatteignable ssi F = 2 kappa theta / xi^2 >= 1 (Feller 1951). La loi stationnaire est Gamma(F, xi^2/(2 kappa)) : P(v < eps) ~ eps^F. L'Euler full-truncation tronque quand v_n + xi sqrt(v_n dt) Z < 0, donc avec probabilité ~ dt^F.",
        "attendu": "Pente log-log de la fraction tronquée en dt égale à F (quand le nombre de pas tronqués observés est assez grand) ; troncature massive et biais de la moyenne quand F < 1.",
        "chiffres": {"blocs": blocs, "kappa": kappa, "theta": theta, "n_chemins": n_chemins, "pas": list(pas)},
        "conclusion": conclusion,
        "duree_s": time.perf_counter() - t0,
    })


BLOCS = {"Q1": q1_ordres_convergence, "Q2": q2_mlmc_milstein, "Q3": q3_pont_acp, "Q4": q4_stratification_lhs,
         "Q5": q5_importance_sampling, "Q6": q6_aad, "Q7": q7_risque, "Q8": q8_feller}
