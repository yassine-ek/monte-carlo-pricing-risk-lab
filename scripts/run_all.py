"""Relance toute la validation : results/validation.json, figures/*.png, tableaux de docs/validation.md.

Usage : python scripts/run_all.py            (environ 2 minutes)
Toutes les graines sont fixées : deux exécutions donnent les mêmes estimations (hors temps de calcul).
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "scripts"))

import figures  # noqa: E402
from mclab import accel, americaine, barriere, grecques, mlmc, qmc, reduction_variance as rv  # noqa: E402
from mclab.analytique import (asiatique_geo_prix, bs_prix, crr_americain, down_and_out_call,  # noqa: E402
                              heston_prix, merton_prix)
from mclab.modeles import gbm_exact, heston_euler, heston_qe, merton_exact  # noqa: E402
from mclab.utils import (ParamsHeston, ParamsMerton, Z95, payoff_actualise, resumer,  # noqa: E402
                         vol_implicite_bs)
from mclab.validation import couverture_ic95, ecart_parite  # noqa: E402

FIGURES, RESULTATS = RACINE / "figures", RACINE / "results"
BS = dict(S0=100.0, K=100.0, T=1.0, r=0.02, sigma=0.2)   # cas de base Black-Scholes
H = ParamsHeston()                                        # Heston de référence
M = ParamsMerton()
CAS = []


def ajouter(id_, theme, methode, res, ref, type_ref="formule fermée", tolerance_biais=0.0,
            attendu_echec=False, note="", se_ref=0.0, **extra):
    """Enregistre un cas : estimation, SE, IC95, référence, z-score et verdict (|écart| < 3 SE + biais toléré)."""
    se = float(np.hypot(res.erreur_type, se_ref))
    ecart = float(res.prix - ref)
    z = ecart / se if se > 0 else (0.0 if ecart == 0 else float("inf"))
    if extra.get("est_une_borne"):  # borne HAUTE : valide si elle ne passe pas sous la référence et reste utile
        verdict = bool(res.prix >= ref - 3 * se and ecart <= tolerance_biais)
    else:
        verdict = bool(abs(ecart) < 3 * se + tolerance_biais)
    CAS.append({
        "id": id_, "theme": theme, "methode": methode, "estimation": float(res.prix),
        "erreur_type": float(res.erreur_type), "ic95_bas": float(res.ic_bas), "ic95_haut": float(res.ic_haut),
        "reference": float(ref), "type_reference": type_ref, "z": float(z),
        "ecart_relatif": float(ecart / ref) if ref else None, "n_chemins": int(res.n_chemins),
        "temps_s": float(res.temps_s), "tolerance_biais": tolerance_biais, "attendu_echec": attendu_echec,
        "verdict_ok": verdict, "note": note, **extra})


def section_europeenne():
    ref = bs_prix(**BS)
    n = 500_000
    ajouter("bs_naif", "Call européen BS", "MC naïf", rv.mc_naif("call_europeen", n_pas=1, n_chemins=n, graine=1, **BS), ref)
    ajouter("bs_antithetique", "Call européen BS", "MC antithétique",
            rv.mc_antithetique("call_europeen", n_pas=1, n_chemins=n, graine=1, **BS), ref)
    ajouter("bs_controle", "Call européen BS", "VC sur S_T",
            rv.mc_controle("call_europeen", n_pas=1, n_chemins=n, graine=1, **BS), ref)
    ajouter("bs_qmc", "Call européen BS", "QMC Sobol randomisé (16 x 2^14)",
            qmc.prix_gbm_qmc("call_europeen", n_pas=1, m=14, k_brouillages=16, graine=1, **BS), ref)


def section_modeles():
    ref_h = heston_prix(H, 100.0, 1.0)
    S = heston_euler(H, 1.0, 100, 200_000, graine=2)
    t0 = time.perf_counter()
    res = resumer(payoff_actualise("call_europeen", S, 100.0, H.r, 1.0), t0)
    ajouter("heston_euler", "Call Heston", "Euler full-truncation (100 pas)", res, ref_h, "semi-fermé (Fourier)",
            tolerance_biais=0.05, note="biais de discrétisation O(dt) toléré : 0,05")
    t0 = time.perf_counter()
    S = heston_qe(H, 1.0, 20, 500_000, graine=3)
    res = resumer(payoff_actualise("call_europeen", S, 100.0, H.r, 1.0), t0)
    ajouter("heston_qe", "Call Heston", "QE d'Andersen (20 pas)", res, ref_h, "semi-fermé (Fourier)")
    ajouter("heston_numba", "Call Heston", "QE numba parallèle (50 pas)",
            accel.prix_call_heston_numba(H, 100.0, 1.0, 50, 4_000_000, graine=4), ref_h, "semi-fermé (Fourier)")
    t0 = time.perf_counter()
    S = merton_exact(M, 1.0, 10, 500_000, graine=5)
    res = resumer(payoff_actualise("call_europeen", S, 100.0, M.r, 1.0), t0)
    ajouter("merton", "Call Merton", "schéma exact (Poisson)", res, merton_prix(M, 100.0, 1.0), "série de Merton")


def reference_asiatique_arith():
    """Pas de formule fermée : référence = 8 lots indépendants de 250 000 chemins avec variable de contrôle."""
    lots = [rv.mc_controle("call_asiatique_arith", n_pas=12, n_chemins=250_000, graine=900 + i, **BS) for i in range(8)]
    return float(np.mean([l.prix for l in lots])), float(np.sqrt(sum(l.erreur_type**2 for l in lots)) / len(lots))


def section_asiatiques():
    """Renvoie les lignes de la figure d'efficacité."""
    ref_geo = asiatique_geo_prix(100.0, 100.0, 1.0, 0.02, 0.2, 12)
    ajouter("asia_geo_naif", "Asiatique géométrique (12 dates)", "MC naïf",
            rv.mc_naif("call_asiatique_geo", n_pas=12, n_chemins=500_000, graine=6, **BS), ref_geo)
    ajouter("asia_geo_qmc", "Asiatique géométrique (12 dates)", "QMC pont (16 x 2^12)",
            qmc.prix_gbm_qmc("call_asiatique_geo", n_pas=12, m=12, k_brouillages=16, graine=6, **BS), ref_geo)

    ref, se_ref = reference_asiatique_arith()
    type_ref = f"MC haute précision (VC, 2 M chemins, SE={se_ref:.1e})"
    n = 200_000
    naif = rv.mc_naif("call_asiatique_arith", n_pas=12, n_chemins=n, graine=7, **BS)
    anti = rv.mc_antithetique("call_asiatique_arith", n_pas=12, n_chemins=n, graine=7, **BS)
    ctrl = rv.mc_controle("call_asiatique_arith", n_pas=12, n_chemins=n, graine=7, **BS)
    for id_, nom, res in (("asia_arith_naif", "MC naïf", naif), ("asia_arith_anti", "MC antithétique", anti),
                          ("asia_arith_ctrl", "VC asiatique géométrique", ctrl)):
        ajouter(id_, "Asiatique arithmétique (12 dates)", nom, res, ref, type_ref, se_ref=se_ref)
    ajouter("asia_arith_qmc", "Asiatique arithmétique (12 dates)", "QMC pont (16 x 2^12)",
            qmc.prix_gbm_qmc("call_asiatique_arith", n_pas=12, m=12, k_brouillages=16, graine=7, **BS), ref, type_ref, se_ref=se_ref)

    b = dict(BS, K=180.0)
    ref_otm = bs_prix(**b)
    naif_otm = rv.mc_naif("call_europeen", n_pas=1, n_chemins=n, graine=8, **b)
    imp = rv.mc_importance_call(n_chemins=n, graine=8, **b)
    ajouter("otm_naif", "Call très OTM (K=180)", "MC naïf", naif_otm, ref_otm)
    ajouter("otm_is", "Call très OTM (K=180)", f"importance sampling (mu={rv.mu_optimal_call(**b):.2f})", imp, ref_otm)

    naif_eur = rv.mc_naif("call_europeen", n_pas=1, n_chemins=500_000, graine=9, **BS)
    anti_eur = rv.mc_antithetique("call_europeen", n_pas=1, n_chemins=500_000, graine=9, **BS)
    ctrl_eur = rv.mc_controle("call_europeen", n_pas=1, n_chemins=500_000, graine=9, **BS)
    lignes = []
    for nom, ref_naif, vr in (("Européen : antithétique", naif_eur, anti_eur), ("Européen : VC sur S_T", naif_eur, ctrl_eur),
                              ("Asiatique arith. : antithétique", naif, anti), ("Asiatique arith. : VC géométrique", naif, ctrl),
                              ("Call OTM K=180 : importance sampling", naif_otm, imp)):
        lignes.append({"nom": nom, "facteur_variance": rv.facteur_variance(ref_naif, vr),
                       "facteur_efficacite": rv.facteur_efficacite(ref_naif, vr)})
    return lignes


def section_barriere():
    p = dict(S0=100.0, K=100.0, H=90.0, T=1.0, r=0.02, sigma=0.2)
    ref = down_and_out_call(**p)
    for methode, nom in (("naif", "MC naïf (grille)"), ("bgk", "décalage BGK"), ("pont", "pont brownien")):
        res = barriere.down_and_out_mc(m=50, n_chemins=500_000, graine=10, methode=methode, **p)
        ajouter(f"barriere_{methode}", "Down-and-out call (H=90, 50 dates)", nom, res, ref, "formule continue (Reiner-Rubinstein)",
                attendu_echec=(methode == "naif"),
                note="biais de surveillance discrète attendu" if methode == "naif" else "")
    return barriere.etude_biais(ms=(5, 10, 25, 50, 100, 250), n_chemins=200_000, graine=20)


def section_americaine():
    p = dict(S0=100.0, K=100.0, T=1.0, r=0.06, sigma=0.2)
    crr_am = crr_americain(**p, N=2000)
    crr_ber50 = crr_americain(**p, N=2000, pas_exercice=40)
    lsm50 = americaine.lsm_put_americain(**p, n_pas=50, n_apprentissage=50_000, n_evaluation=200_000, graine=0)
    ajouter("lsm_put_50", "Put américain LSM (50 dates)", "Longstaff-Schwartz, Laguerre deg. 3, évaluation indépendante",
            lsm50.resultat, crr_ber50, "arbre CRR bermudéen (N=2000, 50 dates)",
            note=f"CRR américain N=2000 : {crr_am:.4f} ; LSM/CRR américain - 1 = {lsm50.resultat.prix / crr_am - 1:+.2%} ; "
                 f"sur l'apprentissage : {lsm50.prix_apprentissage:.4f}")
    for s0 in (36.0, 40.0, 44.0):  # tableau 1 de Longstaff-Schwartz (2001), sigma=0,2, T=1
        q_ls = dict(S0=s0, K=40.0, T=1.0, r=0.06, sigma=0.2)
        lsm_ls = americaine.lsm_put_americain(**q_ls, n_pas=50, n_apprentissage=50_000, n_evaluation=100_000, graine=0)
        ajouter(f"lsm_ls2001_S{int(s0)}", f"Put américain LSM, cas Longstaff-Schwartz (S0={int(s0)}, K=40)",
                "Longstaff-Schwartz, 50 dates", lsm_ls.resultat, crr_americain(**q_ls, N=2000, pas_exercice=40),
                "arbre CRR bermudéen (N=2000, 50 dates)",
                note=f"CRR américain N=2000 : {crr_americain(**q_ls, N=2000):.4f} ; écart relatif LSM/américain "
                     f"{lsm_ls.resultat.prix / crr_americain(**q_ls, N=2000) - 1:+.2%}")
    encadrements = []
    for n_dates, n_ext in ((10, 5000), (50, 3000)):
        lsm = lsm50 if n_dates == 50 else americaine.lsm_put_americain(**p, n_pas=n_dates, n_apprentissage=50_000,
                                                                       n_evaluation=200_000, graine=0)
        haute = americaine.borne_haute_duale(lsm, n_exterieurs=n_ext)
        ber = crr_americain(**p, N=2000, pas_exercice=2000 // n_dates)
        encadrements.append({"n_dates": n_dates, "borne_basse": lsm.resultat.prix, "se_basse": lsm.resultat.erreur_type,
                             "sur_apprentissage": lsm.prix_apprentissage, "crr_bermudeen": ber,
                             "borne_haute": haute.prix, "se_haute": haute.erreur_type,
                             "ecart_haute_pct": 100 * (haute.prix / ber - 1),
                             "encadre": bool(lsm.resultat.prix - 3 * lsm.resultat.erreur_type <= ber
                                             <= haute.prix + 3 * haute.erreur_type)})
        if n_dates == 10:
            ajouter("lsm_put_10", "Put américain LSM (10 dates)", "Longstaff-Schwartz", lsm.resultat, ber,
                    "arbre CRR bermudéen (N=2000, 10 dates)")
            ajouter("lsm_borne_haute_10", "Put américain, borne duale (10 dates)",
                    "Rogers / Haugh-Kogan (valeur LSM deg. 7, quadrature)", haute, ber,
                    "arbre CRR bermudéen (N=2000, 10 dates)", tolerance_biais=0.05 * ber,
                    note="borne HAUTE : doit être >= référence ; verdict = au-dessus de la référence (à 3 SE près) et à moins de 5 %",
                    est_une_borne=True)
    # frontière : LSM (50 dates) contre arbre CRR (N=500)
    _, front_crr = crr_americain(**p, N=500, retourner_frontiere=True)
    dates = lsm50.dates
    donnees = {"K": p["K"], "t_lsm": dates[1:-1], "frontiere_lsm": lsm50.frontiere[1:],
               "t_crr": np.arange(500) * p["T"] / 500, "frontiere_crr": pd.Series(front_crr[:500]).rolling(21, center=True, min_periods=5).mean().to_numpy()}
    e10 = encadrements[0]
    donnees.update(basse=e10["borne_basse"], se_basse=e10["se_basse"], apprentissage=e10["sur_apprentissage"],
                   crr=e10["crr_bermudeen"], haute=e10["borne_haute"], se_haute=e10["se_haute"], n_dates=10)
    return donnees, encadrements


def section_grecques():
    lignes = []
    resultats = grecques.toutes_les_grecques(n_chemins=500_000, graine=11)
    for (payoff, grecque, methode), (res, exacte) in resultats.items():
        echec = payoff == "digitale" and methode == "pathwise"
        ajouter(f"grecque_{payoff}_{grecque}_{methode}", f"Grecque {grecque} ({payoff})", methode, res, exacte,
                "formule fermée BS", attendu_echec=echec,
                note="pathwise inapplicable : la dérivée de l'indicatrice est nulle p.s." if echec else "",
                payoff=payoff, grecque=grecque)
        lignes.append({"payoff": payoff, "grecque": grecque, "methode": methode, "estimation": res.prix,
                       "erreur_type": res.erreur_type, "exacte": exacte})
    return lignes


def section_mlmc():
    sorties = {}
    cfg = {"GBM (Euler)": (mlmc.echantillonneur_gbm_euler(100.0, 100.0, 1.0, 0.02, 0.2), [0.1, 0.05, 0.02, 0.01, 0.005],
                           bs_prix(**BS), "formule fermée BS", 0.01),
            "Heston (Euler)": (mlmc.echantillonneur_heston_euler(H, 100.0, 1.0), [0.2, 0.1, 0.05, 0.03, 0.02],
                               heston_prix(H, 100.0, 1.0), "semi-fermé (Fourier)", 0.05)}
    for nom, (ech, epsilons, ref, type_ref, eps_cas) in cfg.items():
        res = mlmc.cout_vs_epsilon(ech, epsilons, graine=30)
        sorties[nom] = {"epsilons": epsilons, "cout_mlmc": [r.cout for r in res], "cout_std": [r.cout_mc_standard for r in res],
                        "reference": res[-1], "niveaux_finaux": [len(r.niveaux) - 1 for r in res],
                        "pente_cout_mlmc": float(np.polyfit(np.log(1 / np.array(epsilons)), np.log([r.cout for r in res]), 1)[0]),
                        "pente_cout_std": float(np.polyfit(np.log(1 / np.array(epsilons)), np.log([r.cout_mc_standard for r in res]), 1)[0]),
                        # V_l ~ 2^(pente l) sur le run le plus précis (niveaux >= 1) : GBM attendu -1, Heston plus lent
                        "pente_var_niveaux": float(np.polyfit(res[-1].niveaux[1:], np.log2(res[-1].var_Y[1:]), 1)[0])}
        r = res[epsilons.index(eps_cas)]
        ajouter(f"mlmc_{nom.split()[0].lower()}", f"MLMC {nom}", f"MLMC de Giles, eps={eps_cas}", r.resultat, ref, type_ref,
                tolerance_biais=eps_cas / np.sqrt(2),
                note=f"{len(r.niveaux) - 1} niveaux ; coût MLMC {r.cout:.2e} contre {r.cout_mc_standard:.2e} pour un MC standard estimé "
                     f"(rapport MLMC/standard = {r.cout / r.cout_mc_standard:.2f}) ; biais toléré eps/sqrt(2)")
    return sorties


def section_qualite():
    """Parité call-put sur les prix MC et couverture des IC95 : les contrôles sur les estimateurs eux-mêmes."""
    cas_parite = {"GBM exact": (gbm_exact(100.0, 0.02, 0.2, 1.0, 5, 500_000, graine=40), 0.02, 0.0),
                  "Heston QE": (heston_qe(H, 1.0, 20, 500_000, graine=41), H.r, 0.0),
                  "Heston Euler": (heston_euler(H, 1.0, 100, 200_000, graine=42), H.r, 0.05),
                  "Merton exact": (merton_exact(M, 1.0, 5, 500_000, graine=43), M.r, 0.0)}
    for nom, (S, r, tol) in cas_parite.items():
        res, exacte = ecart_parite(S, 100.0, 100.0, 1.0, r)
        ajouter(f"parite_{nom.split()[0].lower()}_{nom.split()[-1].lower()}", "Parité call-put sur prix MC",
                nom, res, exacte, "S0 e^{-qT} - K e^{-rT}", tolerance_biais=tol,
                note="C_MC - P_MC sur mêmes chemins (teste E[S_T])" + ("; biais Euler O(dt) toléré 0,05" if tol else ""))
    n, ref = 2000, bs_prix(**BS)
    b = dict(BS, K=180.0)
    ref_otm, ref_geo = bs_prix(**b), asiatique_geo_prix(100.0, 100.0, 1.0, 0.02, 0.2, 16)
    estimateurs = [
        ("MC naïf", lambda g: rv.mc_naif("call_europeen", n_pas=1, n_chemins=n, graine=g, **BS), ref, 1000),
        ("antithétique", lambda g: rv.mc_antithetique("call_europeen", n_pas=1, n_chemins=n, graine=g, **BS), ref, 1000),
        ("variable de contrôle", lambda g: rv.mc_controle("call_europeen", n_pas=1, n_chemins=n, graine=g, n_pilote=5000, **BS), ref, 1000),
        ("importance sampling\n(call K=180)", lambda g: rv.mc_importance_call(n_chemins=n, graine=g, **b), ref_otm, 1000),
        ("QMC randomisé\n(asiatique géo., 16 x 2^6)", lambda g: qmc.prix_gbm_qmc("call_asiatique_geo", n_pas=16, m=6, k_brouillages=16, graine=g, **BS), ref_geo, 1000),
        ("MC naïf sur call K=180\n(événement rare, contre-exemple)", lambda g: rv.mc_naif("call_europeen", n_pas=1, n_chemins=n, graine=g, **b), ref_otm, 1000),
    ]
    lignes = []
    for nom, est, reference, reps in estimateurs:
        c = couverture_ic95(est, reference, reps, graine=50)
        lignes.append({"nom": nom, **{k: float(v) for k, v in c.items()}})
    return lignes


def section_smile():
    strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    n = 400_000
    exact = [vol_implicite_bs(heston_prix(H, k, 1.0), H.S0, k, 1.0, H.r) for k in strikes]
    sortie = {"strikes": strikes, "exact": exact, "n_chemins": n}
    for cle, S in (("qe", heston_qe(H, 1.0, 20, n, graine=60)), ("euler", heston_euler(H, 1.0, 50, n, graine=61))):
        iv, demi = [], []
        for k in strikes:
            res = resumer(payoff_actualise("call_europeen", S, float(k), H.r, 1.0), time.perf_counter())
            iv.append(vol_implicite_bs(res.prix, H.S0, k, 1.0, H.r))
            bas = vol_implicite_bs(res.prix - Z95 * res.erreur_type, H.S0, k, 1.0, H.r)
            haut = vol_implicite_bs(res.prix + Z95 * res.erreur_type, H.S0, k, 1.0, H.r)
            demi.append((haut - bas) / 2)
        sortie[cle] = {"iv": iv, "demi_largeur": demi, "ecart_max_bp": 1e4 * float(np.max(np.abs(np.array(iv) - exact)))}
    return sortie


def en_json(x):
    if isinstance(x, dict):
        return {str(k): en_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [en_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return en_json(x.tolist())
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def formater(x, chiffres=4):
    return f"{x:.{chiffres}f}"


def tableaux_markdown(cas, couverture, encadrements, convergence, benchmark, efficacite, donnees_mlmc, etude_biais):
    """Tableaux de docs/validation.md : lus dans les résultats, jamais saisis à la main."""
    def verdict(c):
        if c["attendu_echec"]:
            return "échec attendu (démo)"
        if c.get("est_une_borne"):
            return "borne valide" if c["verdict_ok"] else "BORNE INVALIDE"
        return "OK" if c["verdict_ok"] else "HORS TOLÉRANCE"

    def lignes(filtre):
        sortie = []
        for c in cas:
            if filtre(c):
                sortie.append(f"| {c['theme']} | {c['methode'].replace('|', '/')} | {formater(c['estimation'])} ± {formater(c['erreur_type'])} "
                              f"| [{formater(c['ic95_bas'])} ; {formater(c['ic95_haut'])}] | {formater(c['reference'])} "
                              f"| {c['z']:+.2f} | {verdict(c)} |")
        return sortie

    entete = ("| Cas | Méthode | Estimation ± SE | IC95 | Référence | z | Verdict |\n"
              "|---|---|---|---|---|---|---|")
    est_grecque = lambda c: c["theme"].startswith("Grecque")
    est_parite = lambda c: c["theme"].startswith("Parité")
    prix = "\n".join([entete] + lignes(lambda c: not est_grecque(c) and not est_parite(c)))
    gr = "\n".join([entete] + lignes(est_grecque))
    par = "\n".join([entete] + lignes(est_parite))
    cov = "\n".join(["| Estimateur | Couverture observée | Attendu | Écart (en sigmas binomiaux) |", "|---|---|---|---|"] + [
        f"| {l['nom'].replace(chr(10), ' ')} | {100 * l['couverture']:.1f} % ({int(round(l['couverture'] * l['repetitions']))}/{int(l['repetitions'])}) "
        f"| 95 % ± {196 * l['sigma_binomial']:.1f} pt | {l['ecart_en_sigmas']:+.1f} |" for l in couverture])
    enc = "\n".join(["| Dates d'exercice | LSM (borne basse) | Sur l'apprentissage | CRR bermudéen | Borne haute duale | Écart borne haute |",
                     "|---|---|---|---|---|---|"] + [
        f"| {e['n_dates']} | {e['borne_basse']:.4f} ± {e['se_basse']:.4f} | {e['sur_apprentissage']:.4f} | {e['crr_bermudeen']:.4f} "
        f"| {e['borne_haute']:.4f} ± {e['se_haute']:.4f} | {e['ecart_haute_pct']:+.1f} % |" for e in encadrements])
    pentes = convergence["pentes"]
    conv = (f"pente MC = {pentes['mc']:.2f} ; QMC incréments = {pentes['qmc_incrementale']:.2f} ; "
            f"QMC pont brownien = {pentes['qmc_pont']:.2f}")
    bench = (f"numba parallèle : {benchmark['numba_parallele']['chemins_par_s']:,.0f} chemins/s, soit x{benchmark['acceleration_vs_numpy_terminal']:.0f} "
             f"face à numpy (S_T seul) et x{benchmark['acceleration_vs_numpy_complet']:.0f} face à numpy (trajectoires complètes)").replace(",", " ")
    eff = "\n".join(["| Technique | Facteur de variance | Facteur d'efficacité (variance x temps) |", "|---|---|---|"] + [
        f"| {l['nom']} | {l['facteur_variance']:.1f}x | {l['facteur_efficacite']:.1f}x |" for l in efficacite])
    lignes_mlmc = ["| Modèle | eps | niveaux | coût MLMC | coût MC standard (estimé) | rapport MLMC / standard (< 1 : MLMC moins cher) |", "|---|---|---|---|---|---|"]
    pentes_mlmc = []
    for nom, d in donnees_mlmc.items():
        for eps, niv, cm, cs in zip(d["epsilons"], d["niveaux_finaux"], d["cout_mlmc"], d["cout_std"]):
            lignes_mlmc.append(f"| {nom} | {eps} | {niv} | {cm:.2e} | {cs:.2e} | {cm / cs:.2f} |")
        pentes_mlmc.append(f"{nom} : exposant du coût en 1/eps = {d['pente_cout_mlmc']:.2f} (MLMC) contre {d['pente_cout_std']:.2f} (MC standard)"
                           + (f", variance par niveau V_l ~ 2^({d['pente_var_niveaux']:.2f} l)" if "pente_var_niveaux" in d else ""))
    ml = "\n".join(lignes_mlmc) + "\n\n" + " ; ".join(pentes_mlmc) + "."
    bar = "\n".join(["| Dates m | MC naïf | Décalage BGK | Pont brownien |", "|---|---|---|---|"] + [
        f"| {m} | " + " | ".join(f"{etude_biais[k]['erreur'][i]:+.4f} ± {etude_biais[k]['erreur_type'][i]:.4f}" for k in ("naif", "bgk", "pont")) + " |"
        for i, m in enumerate(etude_biais["ms"])])
    return {"prix": prix, "grecques": gr, "parite": par, "couverture": cov, "encadrement": enc, "convergence": conv,
            "benchmark": bench, "efficacite": eff, "mlmc": ml, "barriere": bar}


def mettre_a_jour_validation(tables):
    chemin = RACINE / "docs" / "validation.md"
    if not chemin.exists():
        return
    texte = chemin.read_text(encoding="utf-8")
    for cle, contenu in tables.items():
        debut, fin = f"<!-- AUTO:{cle}:DEBUT -->", f"<!-- AUTO:{cle}:FIN -->"
        if debut in texte and fin in texte:
            avant, reste = texte.split(debut, 1)
            _, apres = reste.split(fin, 1)
            texte = f"{avant}{debut}\n{contenu}\n{fin}{apres}"
    chemin.write_text(texte, encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t_debut = time.perf_counter()
    FIGURES.mkdir(exist_ok=True)
    RESULTATS.mkdir(exist_ok=True)

    def etape(nom, f, *a):
        t = time.perf_counter()
        sortie = f(*a)
        print(f"  {nom:<28} {time.perf_counter() - t:6.1f} s", flush=True)
        return sortie

    print("Validation du laboratoire Monte Carlo")
    etape("call européen", section_europeenne)
    etape("modèles (Heston, Merton)", section_modeles)
    lignes_efficacite = etape("asiatiques + réduc. variance", section_asiatiques)
    etude_biais = etape("barrière discrète", section_barriere)
    donnees_lsm, encadrements = etape("américain LSM + dualité", section_americaine)
    lignes_grecques = etape("grecques", section_grecques)
    donnees_mlmc = etape("MLMC", section_mlmc)
    couverture = etape("couverture IC95 + parité", section_qualite)
    smile = etape("smile Heston", section_smile)
    convergence = etape("convergence MC vs QMC", qmc.etude_convergence, 100.0, 100.0, 1.0, 0.02, 0.2, 16, range(5, 15), 60, 0)
    benchmark = etape("benchmark numba", accel.benchmark, H, 1.0, 50, 200_000)

    figures.convergence(convergence, FIGURES)
    figures.efficacite(lignes_efficacite, FIGURES)
    figures.smile(smile, FIGURES)
    figures.frontiere_lsm(donnees_lsm, FIGURES)
    figures.grecques(lignes_grecques, FIGURES)
    figures.mlmc(donnees_mlmc, FIGURES)
    figures.biais_barriere(etude_biais, FIGURES)
    figures.benchmark_numba(benchmark, FIGURES)
    figures.couverture(couverture, FIGURES)

    resume_mlmc = {nom: {k: v for k, v in d.items() if k != "reference"} for nom, d in donnees_mlmc.items()}
    sortie = {
        "meta": {"python": platform.python_version(), "numpy": np.__version__, "cpu_logiques": os.cpu_count(),
                 "duree_totale_s": time.perf_counter() - t_debut,
                 "regle_verdict": "OK si |estimation - reference| < 3 erreurs-types (+ biais de discrétisation documenté)"},
        "cas": CAS,
        "couverture_ic95": couverture,
        "encadrement_americain": encadrements,
        "convergence": {"N": convergence["N"], "rmse": convergence["rmse"], "pentes": convergence["pentes"]},
        "efficacite": lignes_efficacite,
        "barriere_biais": etude_biais,
        "mlmc": resume_mlmc,
        "smile_heston": {"ecart_max_bp_qe": smile["qe"]["ecart_max_bp"], "ecart_max_bp_euler": smile["euler"]["ecart_max_bp"]},
        "benchmark_numba": benchmark,
    }
    (RESULTATS / "validation.json").write_text(json.dumps(en_json(sortie), indent=2, ensure_ascii=False), encoding="utf-8")
    tables = tableaux_markdown(CAS, couverture, encadrements, convergence, benchmark, lignes_efficacite, donnees_mlmc, etude_biais)
    (RESULTATS / "tableau_validation.md").write_text("\n\n".join(f"## {k}\n\n{v}" for k, v in tables.items()), encoding="utf-8")
    mettre_a_jour_validation(tables)
    n_ok = sum(c["verdict_ok"] or c["attendu_echec"] for c in CAS)
    print(f"{n_ok}/{len(CAS)} cas conformes (|z|<3 ou échec attendu) ; durée {time.perf_counter() - t_debut:.0f} s")


if __name__ == "__main__":
    main()
