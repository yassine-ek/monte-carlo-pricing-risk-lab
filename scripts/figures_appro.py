"""Figures de l'approfondissement (une par question). Chaque fonction reçoit le bloc `chiffres` d'une question
(listes JSON natives) et renvoie une figure matplotlib ; l'enregistrement en PNG (dpi 130) est fait par le script appelant."""
import sys

import matplotlib

if "ipykernel" not in sys.modules:   # en notebook, on garde le backend « inline »
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DPI = 130
# Okabe-Ito : lisible en daltonisme ; une couleur fixe par objet
C = {"euler": "#D55E00", "milstein": "#0072B2", "qe": "#009E73", "mc": "#6b7280", "incrementale": "#E69F00",
     "pont": "#0072B2", "acp": "#CC79A7", "ref": "#111827", "ok": "#009E73", "alerte": "#D55E00"}
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "legend.fontsize": 8})


def _loglog_ref(ax, x, y0, pente, **kw):
    x = np.asarray(x, float)
    ax.loglog(x, y0 * (x / x[0]) ** pente, **kw)


# ----------------------------------------------------------------------------------------------- Q1
def fig_q1(ch):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    g, par = ch["gbm"], ch["parametres_gbm"]
    dts = [par["T"] / 2**k for k in par["ks"]]
    a = ax[0, 0]
    for s in ("euler", "milstein"):
        a.loglog(dts, g[s]["fort_l1"]["erreurs"], "o-", color=C[s], label=f"{s} : pente {g[s]['fort_l1']['pente']:.2f}")
    _loglog_ref(a, dts, g["euler"]["fort_l1"]["erreurs"][0], 0.5, ls="--", color=C["euler"], alpha=0.5, label="référence dt^0,5")
    _loglog_ref(a, dts, g["milstein"]["fort_l1"]["erreurs"][0], 1.0, ls="--", color=C["milstein"], alpha=0.5, label="référence dt^1")
    a.set(title="GBM : erreur forte E|S_T^dt - S_T| (mêmes browniens)", xlabel="dt", ylabel="erreur L1")
    a.legend()
    a = ax[0, 1]
    for s in ("euler", "milstein"):
        w = g[s]["faible_call"]
        a.errorbar(dts, np.abs(w["erreurs"]), yerr=1.96 * np.array(w["se"]), fmt="o-", color=C[s], capsize=2,
                   label=f"{s}, call (CRN + contrôle) : pente {w['pente']:.2f}")
        a.loglog(dts, g[s]["faible_x2_exact"]["erreurs"], ":", color=C[s], label=f"{s}, E[S^2] exact : pente {g[s]['faible_x2_exact']['pente']:.3f}")
    _loglog_ref(a, dts, g["milstein"]["faible_call"]["erreurs"][0] if g["milstein"]["faible_call"]["erreurs"][0] else 1e-2, 1.0, ls="--", color=C["ref"], alpha=0.4, label="référence dt^1")
    a.set(title="GBM : erreur faible |E f(S^dt) - E f(S)|", xlabel="dt", ylabel="erreur (valeur absolue, IC95)", xscale="log", yscale="log")
    a.legend()
    a = ax[1, 0]
    for nom, ls in (("feller_respecte", "-"), ("feller_viole", "--")):
        for s in ("euler", "milstein"):
            r = ch["cir"][nom][s]
            a.loglog(ch["cir"][nom]["dts"], r["erreurs"], marker="o", ls=ls, color=C[s],
                     label=f"{s}, F={ch['cir'][nom]['ratio_feller']:.2f} : pente {r['pente']:.2f}")
    a.set(title="CIR : erreur forte (référence = Milstein à 2^11 pas)", xlabel="dt", ylabel="E|v_T^dt - v_T|")
    a.legend()
    a = ax[1, 1]
    for nom, ls in (("feller_viole", "-"), ("feller_respecte", "--")):
        h = ch["heston"][nom]
        for s, lib in (("euler_ft", "Euler full-truncation"), ("qe", "QE (Andersen)")):
            col = C["euler"] if s == "euler_ft" else C["qe"]
            a.errorbar(h["dts"], h[s]["biais"], yerr=1.96 * np.array(h[s]["se"]), fmt="o" + ls, color=col, capsize=2,
                       label=f"{lib}, F={2 * 1.5 * 0.04 / h['xi'] ** 2:.2f}", alpha=1.0 if nom == "feller_viole" else 0.55)
    a.axhline(0, color=C["ref"], lw=0.8)
    a.set(xscale="log", yscale="symlog", yscatter=None) if False else a.set(xscale="log", yscale="symlog")
    a.set(title="Heston : biais faible du prix du call (IC95 Monte Carlo)", xlabel="dt", ylabel="prix MC - prix semi-fermé")
    a.legend()
    return fig


# ----------------------------------------------------------------------------------------------- Q2
def fig_q2(ch):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    g = ch["gbm"]
    a = ax[0, 0]
    for s in ("euler", "milstein"):
        l = np.array(g[s]["niveaux"])
        a.semilogy(l, g[s]["var_Y"], "o-", color=C[s], label=f"{s} : beta = {g[s]['beta']:.2f}")
        a.semilogy(l[1:], g[s]["var_Y"][2] * 2.0 ** (-g[s]["beta"] * (l[1:] - 2)), "--", color=C[s], alpha=0.5)
    a.set(title="Variance par niveau V_l = Var(P_l - P_{l-1}) (call GBM)", xlabel="niveau l", ylabel="V_l")
    a.legend()
    a = ax[0, 1]
    for s in ("euler", "milstein"):
        l = np.array(g[s]["niveaux"])
        m, se = np.abs(g[s]["moy_Y"]), np.array(g[s]["se_Y"])
        a.errorbar(l[1:], m[1:], yerr=1.96 * se[1:], fmt="o-", color=C[s], capsize=2,
                   label=f"{s} : alpha = {g[s]['alpha']:.2f}" if g[s]["alpha"] is not None else f"{s} : alpha non identifié")
    a.set(yscale="log", title="|E[P_l - P_{l-1}]| (IC95) : ordre faible alpha", xlabel="niveau l", ylabel="|E[Y_l]|")
    a.legend()
    a = ax[1, 0]
    for s in ("euler", "milstein"):
        c = g[s]["cout"]
        a.loglog(c["epsilons"], c["cout_x_eps2"], "o-", color=C[s], label=f"MLMC {s} (mesuré)")
    c = g["euler"]["cout"]
    a.loglog(c["epsilons"], np.array(c["cout_mc_standard"]) * np.array(c["epsilons"]) ** 2, "s--", color=C["mc"], label="MC standard (estimé à la Giles)")
    a.set(title="Coût x eps^2 (pas fins) : plat = O(eps^-2)", xlabel="eps", ylabel="coût x eps^2")
    a.invert_xaxis()
    a.legend()
    a = ax[1, 1]
    noms = list(ch["heston"])
    x = np.arange(len(noms))
    a.bar(x - 0.2, [ch["heston"][n]["euler"]["beta"] for n in noms], 0.4, color=C["euler"], label="Euler")
    a.bar(x + 0.2, [ch["heston"][n]["milstein"]["beta"] for n in noms], 0.4, color=C["milstein"], label="Milstein (sans aire)")
    for y, ls in ((1, ":"), (2, "--")):
        a.axhline(y, color=C["ref"], ls=ls, lw=0.8)
    a.set_xticks(x, [n.split(" ")[0] for n in noms])
    a.set(title="Heston : beta selon Feller et aire de Lévy", ylabel="beta (pente de log2 V_l)")
    a.legend()
    return fig


# ----------------------------------------------------------------------------------------------- Q3
def fig_q3(ch):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    b16, ds = ch["blocs"]["16"], sorted(int(d) for d in ch["blocs"])
    bm = ch["blocs"][str(ds[-1])]
    nom = {"incrementale": "incrémentale", "pont": "pont brownien", "acp": "ACP"}
    a = ax[0]
    for c in ("incrementale", "pont", "acp"):
        a.plot(b16[c]["ks"], np.array(b16[c]["indice"]) * 100, "o-", color=C[c], label=f"{nom[c]} (d=16)")
        a.plot(bm[c]["ks"], np.array(bm[c]["indice"]) * 100, "--", color=C[c], alpha=0.6, label=f"{nom[c]} (d={ds[-1]})")
    a.axhline(99, color=C["ref"], lw=0.8, ls=":")
    a.set(xscale="log", title="Variance expliquée par les k premières coordonnées", xlabel="k", ylabel="% de la variance (Sobol fermé)")
    a.legend(ncol=2)
    for a, d in ((ax[1], 16), (ax[2], ds[-1])):
        b = ch["blocs"][str(d)]
        N = b["mc"]["N"]
        a.loglog(N, b["mc"]["rmse"], "o-", color=C["mc"], label=f"MC : pente {b['mc']['pente']:.2f}")
        for c in ("incrementale", "pont", "acp"):
            r = b[c]["rmse"]
            a.loglog(N, r["rmse"], "s-", color=C[c], label=f"{nom[c]} : pente {r['pente']:.2f}")
        a.set(title=f"RMSE du QMC randomisé, asiatique d={d}", xlabel="N", ylabel="RMSE (écart-type sur brouillages)")
        a.legend()
    return fig


# ----------------------------------------------------------------------------------------------- Q4
def fig_q4(ch):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    a = ax[0]
    cas = [("europeen_K100", "européen K=100"), ("europeen_K140", "européen K=140"), ("asiatique_16_pont", "asiatique 16 (pont)")]
    x, larg = np.arange(len(cas)), 0.12
    series = [("facteur_prop", "proportionnelle", "#56B4E9"), ("facteur_neyman_pilote", "Neyman (pilote)", "#E69F00"),
              ("facteur_neyman_oracle", "Neyman (oracle)", "#D55E00")]
    for i, (cle, lib, col) in enumerate(series):
        a.bar(x + (2 * i - 1) * larg, [ch["strates"][k]["64"]["mesure"][cle] for k, _ in cas], larg, color=col, label=f"{lib} (mesuré)")
        a.scatter(x + (2 * i - 1) * larg, [ch["strates"][k]["64"]["theorie"][cle] for k, _ in cas], marker="_", s=250, color="black",
                  zorder=5, label="théorie" if i == 0 else None)
    a.set_xticks(x, [n for _, n in cas])
    a.set(yscale="log", title="Facteur de variance, H=64 strates (barre : mesuré ; trait : théorie)", ylabel="variance MC naïf / variance stratifiée")
    a.legend()
    a = ax[1]
    noms = list(ch["lhs"])[:2]
    xs = np.arange(len(noms))
    a.bar(xs - 0.2, [ch["lhs"][n]["gain_asymptotique_theorique"] for n in noms], 0.2, color="black", label="théorie 1/(1 - somme S_j)")
    for i, nn in enumerate(("256", "4096")):
        a.bar(xs + 0.2 * i, [ch["lhs"][n]["n"][nn]["gain_mesure"] for n in noms], 0.2, color=["#56B4E9", "#0072B2"][i], label=f"mesuré n={nn}")
    a.set_xticks(xs, ["asiatique 16\nincrémentale", "asiatique 16\npont"])
    a.set(title="LHS : gain de variance, asiatique 16 dates", ylabel="variance MC / variance LHS")
    a.legend()
    a = ax[2]
    par_n = ch["lhs"]["europeen_d1"]["par_n"]
    ns = sorted(int(k) for k in par_n)
    a.loglog(ns, [par_n[str(n)]["var_mc"] for n in ns], "o-", color=C["mc"], label=f"MC : pente {ch['lhs']['europeen_d1']['pente_var_mc']:.2f}")
    a.loglog(ns, [par_n[str(n)]["var_lhs"] for n in ns], "o-", color=C["milstein"], label=f"LHS (d=1) : pente {ch['lhs']['europeen_d1']['pente_var_lhs']:.2f}")
    a.set(title="Call européen : variance de l'estimateur vs n", xlabel="n", ylabel="variance")
    a.legend()
    return fig


# ----------------------------------------------------------------------------------------------- Q5
def fig_q5(ch):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    couleurs = {"naif": C["mc"], "strike": "#E69F00", "mu_star": C["milstein"], "mu_ce": C["acp"]}
    noms = {"naif": "naïf (mu=0)", "strike": "mu = z_K (strike)", "mu_star": "mu* (second moment)", "mu_ce": "mu_CE (entropie croisée)"}
    for a, k in ((ax[0], "call_K180"), (ax[1], "credit")):
        c = ch["cas"][k]
        v0 = c["variance_naive_par_tirage"]
        a.semilogy(c["grille_mu"], np.array(c["variance_grille"]), color=C["ref"], lw=1.3, label="variance théorique M(mu) - I^2")
        for lib, p in c["points"].items():
            a.scatter([p["mu"]], [p["variance_empirique"]], color=couleurs[lib], zorder=5, s=40,
                      label=f"{noms[lib]} : x{p['facteur_variance']:.0f}")
        a.set(title=c["libelle"], xlabel="dérive mu", ylabel="variance par tirage")
        a.legend()
    a = ax[2]
    le = ch["efficacite_logarithmique"]
    aa = [r["a"] for r in le]
    a.plot(aa, [r["rapport_log"] for r in le], "o-", color=C["milstein"], label="ln M(mu*) / (2 ln P)")
    a.axhline(1, color=C["ref"], ls=":", lw=0.8)
    a.set(title="Grandes déviations, événement {Z > a}", xlabel="seuil a", ylabel="rapport logarithmique", ylim=(0.8, 1.02))
    b = a.twinx()
    b.semilogy(aa, [r["erreur_relative_is"] for r in le], "s--", color=C["qe"], label="erreur relative IS")
    b.semilogy(aa, [r["erreur_relative_naive"] for r in le], "s--", color=C["alerte"], label="erreur relative MC naïf")
    b.set_ylabel("erreur relative par tirage")
    b.grid(False)
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = b.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, loc="center right")
    return fig


# ----------------------------------------------------------------------------------------------- Q6
def fig_q6(ch):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    a = ax[0]
    cout = ch["cout"]
    m = [c["m_grecques"] for c in cout]
    a.loglog(m, [c["aad_boucle"] for c in cout], "o-", color=C["milstein"], label="AAD, boucle inverse")
    a.loglog(m, [c["aad_vectorise"] for c in cout], "s-", color=C["qe"], label="AAD, somme cumulée inverse")
    a.loglog(m, [c["bump_seaux"] for c in cout], "^-", color=C["alerte"], label="bump-and-revalue (vegas par seaux)")
    a.loglog(m, [c["reevaluations_bump_seaux"] for c in cout], ":", color=C["ref"], label="2 d + 1 réévaluations (idéal)")
    a.axhspan(3, 4, color=C["ok"], alpha=0.15, label="bande de Griewank (3-4)")
    a.set(title="Coût de toutes les grecques / coût du prix (asiatique)", xlabel="nombre de grecques m = d + 3", ylabel="temps / temps du prix")
    a.legend()
    a = ax[1]
    z = []
    etiquettes = []
    for lib, bloc in (("européen/BS", ch["vs_black_scholes_europeen"]), ("asiat. géo./formule", ch["vs_geometrique_ferme"])):
        for g in ("delta", "vega", "rho"):
            z.append(bloc[g]["z"])
            etiquettes.append(f"{lib} {g}")
    a.bar(range(len(z)), z, color=[C["ok"] if abs(v) < 3 else C["alerte"] for v in z])
    a.axhspan(-3, 3, color=C["ok"], alpha=0.12)
    a.set_xticks(range(len(z)), etiquettes, fontsize=7, rotation=30, ha="right")
    a.set(title="AAD (Monte Carlo, 2 M chemins) vs références indépendantes", ylabel="écart en erreurs-types (z)")
    return fig


# ----------------------------------------------------------------------------------------------- Q7
def fig_q7(ch):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    k = ch["kupiec"]
    a = ax[0]
    a.plot(np.array(k["T"]) / 250, np.array(k["puissance_exacte"]) * 100, color=C["milstein"], lw=1.2, label="puissance exacte (énumération binomiale)")
    for T, p in k["points"].items():
        a.errorbar([p["annees"]], [100 * p["puissance_mc"]], yerr=[196 * p["se_mc"]], fmt="o", color=C["alerte"], capsize=3,
                   label="Monte Carlo (IC95)" if T == "250" else None)
    a.axhline(80, color=C["ref"], ls=":", lw=0.8)
    a.axvline(k["T_80"] / 250, color=C["ref"], ls="--", lw=0.8, label=f"80 % durable : {k['T_80'] / 250:.1f} ans")
    a.set(xlim=(0, 10), title="Kupiec : puissance contre un modèle à 2 % d'exceptions", xlabel="années de backtest (250 j)", ylabel="probabilité de rejeter H0 : 1 % (%)")
    a.legend()
    a = ax[1]
    couleurs = ["#999999", "#56B4E9", "#E69F00", "#D55E00", "#CC79A7"]
    for (nom, bloc), col in zip(ch["christoffersen"].items(), couleurs):
        T = sorted(int(t) for t in bloc)
        a.plot(np.array(T) / 250, [100 * bloc[str(t)]["ind"] for t in T], "o-", color=col, label=f"Christoffersen (indép.) : {nom}")
        a.plot(np.array(T) / 250, [100 * bloc[str(t)]["kupiec"] for t in T], "x--", color=col, alpha=0.6)
    a.set(title="Taux de rejet (trait plein : indépendance ; pointillé : Kupiec)", xlabel="années", ylabel="% de rejets")
    a.legend(fontsize=6)
    a = ax[2]
    es = ch["es_euler"]
    if "donnees_absentes" in es:
        a.text(0.5, 0.5, "données de marché absentes", ha="center")
    else:
        actifs = es["actifs"]
        x = np.arange(len(actifs))
        a.bar(x - 0.27, [100 * es["jour_2020_03_09"]["poids"][t] for t in actifs], 0.27, color="#999999", label="poids du capital")
        a.bar(x, [100 * es["jour_2020_03_09"]["parts"][t] for t in actifs], 0.27, color=C["alerte"], label="part de l'ES au 2020-03-09")
        a.bar(x + 0.27, [100 * es["moyenne"]["parts"][t] for t in actifs], 0.27, yerr=[100 * es["moyenne"]["ecart_type_parts"][t] for t in actifs],
              color=C["milstein"], capsize=2, label=f"part moyenne de l'ES ({es['moyenne']['n_dates']} dates)")
        a.axhline(0, color=C["ref"], lw=0.8)
        a.set_xticks(x, [t.replace("USD", "") for t in actifs], fontsize=8)
        a.set(title="Contributions d'Euler à l'ES FHS 97,5 %", ylabel="% (somme des parts = 100 %)")
        a.legend()
    return fig


# ----------------------------------------------------------------------------------------------- Q8
def fig_q8(ch):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    a = ax[0]
    palette = ["#D55E00", "#E69F00", "#009E73", "#0072B2", "#CC79A7"]
    for (k, b), col in zip(ch["blocs"].items(), palette):
        a.loglog(b["dts"], b["fraction_tronquee"], "o-", color=col, label=f"F = {b['F']:g} (pente {b['pente']:.2f})")
    a.set(title="Part des pas tronqués (v < 0), Euler full-truncation", xlabel="dt", ylabel="fraction de pas")
    a.legend()
    a = ax[1]
    F = [b["F"] for b in ch["blocs"].values()]
    pente = [b["pente"] for b in ch["blocs"].values()]
    ic = np.array([b["ic95_pente"] for b in ch["blocs"].values()])
    a.errorbar(F, pente, yerr=[np.array(pente) - ic[:, 0], ic[:, 1] - np.array(pente)], fmt="o", color=C["milstein"], capsize=3, label="pente mesurée (IC95 statistique)")
    a.plot([0, 4.3], [0, 4.3], "--", color=C["ref"], label="pente = F")
    a.axvline(1, color=C["alerte"], ls=":", label="condition de Feller F = 1")
    a.set(title="Pente de la fraction tronquée en dt vs F = 2 kappa theta / xi^2", xlabel="F", ylabel="pente log-log")
    a.legend()
    return fig


FIGURES = {"Q1": (fig_q1, "appro_q1_ordres_convergence.png"), "Q2": (fig_q2, "appro_q2_mlmc_milstein.png"),
           "Q3": (fig_q3, "appro_q3_pont_acp_dimension_effective.png"), "Q4": (fig_q4, "appro_q4_stratification_lhs.png"),
           "Q5": (fig_q5, "appro_q5_importance_sampling.png"), "Q6": (fig_q6, "appro_q6_aad.png"),
           "Q7": (fig_q7, "appro_q7_puissance_backtests_es.png"), "Q8": (fig_q8, "appro_q8_feller.png")}


def sauver(fig, dossier, nom):
    fig.tight_layout()
    chemin = dossier / nom
    fig.savefig(chemin, dpi=DPI)
    plt.close(fig)
    return chemin
