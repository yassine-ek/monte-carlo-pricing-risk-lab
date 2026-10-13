"""Tracés des figures du laboratoire (PNG, dpi 130). Chaque fonction reçoit des données déjà calculées."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 130
COULEURS = {"mc": "#6b7280", "pont": "#1d4ed8", "inc": "#60a5fa", "exact": "#111827",
            "alerte": "#dc2626", "ok": "#16a34a", "orange": "#ea580c", "violet": "#7c3aed"}
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})


def _sauver(fig, dossier, nom):
    fig.tight_layout()
    chemin = Path(dossier) / nom
    fig.savefig(chemin, dpi=DPI)
    plt.close(fig)
    return chemin


def convergence(etude, dossier):
    """RMSE vs N en log-log : MC (pente -0,5) contre QMC randomisé (pente proche de -1)."""
    N, rmse, pentes = etude["N"], etude["rmse"], etude["pentes"]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    styles = [("mc", "MC pseudo-aléatoire", COULEURS["mc"], "o"),
              ("qmc_incrementale", "QMC Sobol, incréments", COULEURS["inc"], "s"),
              ("qmc_pont", "QMC Sobol, pont brownien", COULEURS["pont"], "D")]
    for cle, nom, couleur, marqueur in styles:
        ax.loglog(N, rmse[cle], marqueur + "-", color=couleur, label=f"{nom} (pente {pentes[cle]:.2f})")
    ax.loglog(N, rmse["mc"][0] * (N / N[0]) ** -0.5, "--", color=COULEURS["mc"], alpha=0.6, label="référence N$^{-1/2}$")
    ax.loglog(N, rmse["qmc_pont"][0] * (N / N[0]) ** -1.0, "--", color=COULEURS["pont"], alpha=0.6, label="référence N$^{-1}$")
    ax.set(xlabel="N (nombre de chemins)", ylabel="RMSE vs formule fermée",
           title="Asiatique géométrique, 16 dates : convergence de la RMSE")
    ax.legend(fontsize=8)
    return _sauver(fig, dossier, "convergence_mc_vs_qmc.png")


def efficacite(lignes, dossier):
    """Barres (échelle log) : facteur de variance et facteur d'efficacité (variance x temps) par technique."""
    noms = [l["nom"] for l in lignes]
    x = np.arange(len(lignes))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - 0.2, [l["facteur_variance"] for l in lignes], 0.4, label="facteur de variance", color=COULEURS["inc"])
    ax.bar(x + 0.2, [l["facteur_efficacite"] for l in lignes], 0.4, label="facteur d'efficacité (variance x temps)",
           color=COULEURS["pont"])
    ax.axhline(1, color=COULEURS["exact"], lw=1)
    ax.set_yscale("log")
    ax.set_xticks(x, noms, rotation=20, ha="right", fontsize=8)
    ax.set(ylabel="gain vs MC naïf (x)", title="Réduction de variance : gains mesurés")
    for i, l in enumerate(lignes):
        ax.text(i + 0.2, l["facteur_efficacite"] * 1.1, f"{l['facteur_efficacite']:.0f}x", ha="center", fontsize=8)
    ax.legend(fontsize=8)
    return _sauver(fig, dossier, "efficacite_vr.png")


def smile(donnees, dossier):
    """Smile de vol implicite Heston : MC (QE, Euler) contre semi-fermé, avec IC95 propagés."""
    K = np.array(donnees["strikes"])
    fig, (ax, bas) = plt.subplots(2, 1, figsize=(6.8, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    ax.plot(K, 100 * np.array(donnees["exact"]), "-", color=COULEURS["exact"], lw=2, label="Heston semi-fermé")
    couleurs = {"qe": COULEURS["pont"], "euler": COULEURS["orange"]}
    for cle, nom in (("qe", "MC schéma QE"), ("euler", "MC Euler full-truncation")):
        d = donnees[cle]
        ax.errorbar(K, 100 * np.array(d["iv"]), yerr=100 * np.array(d["demi_largeur"]), fmt="o", ms=4, capsize=2,
                    color=couleurs[cle], label=nom)
        bas.errorbar(K, 1e4 * (np.array(d["iv"]) - np.array(donnees["exact"])), yerr=1e4 * np.array(d["demi_largeur"]),
                     fmt="o", ms=4, capsize=2, color=couleurs[cle])
    ax.set(ylabel="vol implicite (%)", title="Smile Heston, T=1 (" + f"{donnees['n_chemins']:,}".replace(",", " ") + " chemins)")
    ax.legend(fontsize=8)
    bas.axhline(0, color=COULEURS["exact"], lw=1)
    bas.set(xlabel="strike K", ylabel="MC - exact (points de base)")
    return _sauver(fig, dossier, "smile_heston.png")


def frontiere_lsm(donnees, dossier):
    """Frontière d'exercice LSM contre arbre CRR, et encadrement du prix (borne basse / CRR / borne haute)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.5), gridspec_kw={"width_ratios": [3, 2]})
    a1.plot(donnees["t_crr"], donnees["frontiere_crr"], "-", color=COULEURS["exact"], lw=1.5, label="arbre CRR (N=500)")
    a1.plot(donnees["t_lsm"], donnees["frontiere_lsm"], "o", ms=3.5, color=COULEURS["pont"], label="LSM (régression Laguerre deg. 3)")
    a1.axhline(donnees["K"], color=COULEURS["mc"], ls="--", label="strike K")
    a1.set(xlabel="temps t (années)", ylabel="S*(t) : exercer si S < S*", title="Frontière d'exercice du put américain")
    a1.legend(fontsize=8)
    noms = ["LSM\n(borne basse)", "LSM\nsur l'apprentissage", "CRR\nbermudéen", "borne haute\nduale"]
    valeurs = [donnees["basse"], donnees["apprentissage"], donnees["crr"], donnees["haute"]]
    erreurs = [1.96 * donnees["se_basse"], 0, 0, 1.96 * donnees["se_haute"]]
    a2.errorbar(range(4), valeurs, yerr=erreurs, fmt="o", color=COULEURS["pont"], capsize=4)
    a2.errorbar([2], [donnees["crr"]], fmt="D", color=COULEURS["exact"])
    a2.set_xticks(range(4), noms, fontsize=8)
    a2.set(ylabel="prix", title=f"Encadrement ({donnees['n_dates']} dates d'exercice)")
    return _sauver(fig, dossier, "frontiere_exercice_lsm.png")


def grecques(lignes, dossier):
    """Rapport estimation / valeur exacte (IC95) pour chaque (payoff, grecque, méthode) : le pathwise échoue sur la digitale."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    couleur_methode = {"pathwise": COULEURS["violet"], "lr": COULEURS["orange"], "bump": COULEURS["pont"]}
    for ax, payoff, titre in ((axes[0], "call", "Call européen"), (axes[1], "digitale", "Digitale (cash-or-nothing)")):
        sous = [l for l in lignes if l["payoff"] == payoff]
        grecques_presentes = sorted({l["grecque"] for l in sous})
        for gi, g in enumerate(grecques_presentes):
            for mi, methode in enumerate(("pathwise", "lr", "bump")):
                for l in sous:
                    if l["grecque"] == g and l["methode"] == methode:
                        x = gi + (mi - 1) * 0.22
                        ax.errorbar(x, l["estimation"] / l["exacte"], yerr=1.96 * l["erreur_type"] / abs(l["exacte"]),
                                    fmt="o", color=couleur_methode[methode], capsize=3,
                                    label=methode if gi == 0 else None)
                        if payoff == "digitale" and methode == "pathwise":
                            ax.annotate("pathwise = 0\n(faux)", (x, 0.0), textcoords="offset points", xytext=(0, 12),
                                        ha="center", fontsize=8, color=COULEURS["alerte"])
        ax.axhline(1, color=COULEURS["exact"], lw=1)
        ax.set_xticks(range(len(grecques_presentes)), grecques_presentes)
        ax.set(title=titre)
        ax.legend(fontsize=8, title="méthode")
    axes[0].set_ylabel("estimation MC / valeur exacte")
    return _sauver(fig, dossier, "grecques_mc_vs_exact.png")


def mlmc(donnees, dossier):
    """Pour GBM et Heston : variance et espérance par niveau, nombre de chemins, coût x eps^2 vs eps."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    for ligne, (nom, d) in enumerate(donnees.items()):
        r = d["reference"]  # ResultatMLMC pour le plus petit epsilon
        l = r.niveaux
        a1, a2, a3 = axes[ligne]
        a1.semilogy(l, r.var_P, "o-", color=COULEURS["mc"], label="Var[P$_l$]")
        a1.semilogy(l[1:], r.var_Y[1:], "s-", color=COULEURS["pont"], label="Var[P$_l$ - P$_{l-1}$]")
        a1.semilogy(l[1:], r.var_Y[1] * 2.0 ** -(l[1:] - 1.0), "--", color=COULEURS["pont"], alpha=0.5, label="référence 2$^{-l}$")
        a1.set(xlabel="niveau l", title=f"{nom} : variance par niveau")
        a1.legend(fontsize=8)
        a2.semilogy(l, np.abs(r.moy_P), "o-", color=COULEURS["mc"], label="|E[P$_l$]|")
        a2.semilogy(l[1:], np.abs(r.moy_Y[1:]), "s-", color=COULEURS["pont"], label="|E[P$_l$ - P$_{l-1}$]|")
        a2.set(xlabel="niveau l", title=f"{nom} : espérance par niveau (biais faible)")
        a2.legend(fontsize=8)
        eps = np.array(d["epsilons"])
        a3.loglog(eps, eps**2 * np.array(d["cout_mlmc"]), "o-", color=COULEURS["pont"], label="MLMC")
        a3.loglog(eps, eps**2 * np.array(d["cout_std"]), "s-", color=COULEURS["mc"], label="MC standard (estimé)")
        a3.set(xlabel="précision $\\varepsilon$ (RMSE visée)", ylabel="coût x $\\varepsilon^2$",
               title=f"{nom} : coût vs précision")
        a3.legend(fontsize=8)
        a3.invert_xaxis()
        a3.set_xticks(eps, [f"{e:g}" for e in eps])
        a3.minorticks_off()
    return _sauver(fig, dossier, "mlmc.png")


def biais_barriere(etude, dossier):
    """Erreur signée (prix MC - formule continue) selon le nombre de dates de surveillance m."""
    ms = np.array(etude["ms"])
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for cle, nom, couleur in (("naif", "naïf (grille)", COULEURS["alerte"]), ("bgk", "décalage BGK", COULEURS["orange"]),
                              ("pont", "pont brownien", COULEURS["pont"])):
        ax.errorbar(ms, etude[cle]["erreur"], yerr=1.96 * np.array(etude[cle]["erreur_type"]), fmt="o-",
                    color=couleur, capsize=3, label=nom)
    ax.axhline(0, color=COULEURS["exact"], lw=1)
    ax.set(xscale="log", xlabel="nombre de dates de surveillance m", ylabel="prix MC - formule continue (barres : IC95)",
           title="Down-and-out call : biais de surveillance discrète")
    ax.legend(fontsize=8)
    return _sauver(fig, dossier, "biais_barriere.png")


def benchmark_numba(b, dossier):
    noms = [("numpy_trajectoires_completes", "numpy\n(trajectoires complètes)"),
            ("numpy_terminal_seul", "numpy\n(S_T seul)"), ("numba_parallele", "numba\nparallèle")]
    fig, ax = plt.subplots(figsize=(6, 4.4))
    valeurs = [b[c]["chemins_par_s"] for c, _ in noms]
    ax.bar(range(3), valeurs, color=[COULEURS["mc"], COULEURS["inc"], COULEURS["pont"]])
    for i, v in enumerate(valeurs):
        ax.text(i, v * 1.05, f"{v:,.0f}".replace(",", " ") + "\nchemins/s", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(range(3), [n for _, n in noms], fontsize=8)
    ax.set(ylabel="chemins / seconde", title="Heston QE, " + f"{b['n_chemins']:,}".replace(",", " ") + f" chemins x {b['n_pas']} pas",
           ylim=(min(valeurs) / 2, max(valeurs) * 4))
    return _sauver(fig, dossier, "benchmark_numba.png")


def couverture(lignes, dossier):
    """Couverture empirique des IC95 sur des répétitions indépendantes, avec la bande binomiale à 95 %."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(lignes))
    sigma = lignes[0]["sigma_binomial"]
    ax.axhspan(0.95 - 1.96 * sigma, 0.95 + 1.96 * sigma, color=COULEURS["ok"], alpha=0.15, label="bande binomiale 95 %")
    ax.axhline(0.95, color=COULEURS["ok"], lw=1)
    couleurs = [COULEURS["ok"] if abs(l["ecart_en_sigmas"]) < 3 else COULEURS["alerte"] for l in lignes]
    ax.bar(x, [l["couverture"] for l in lignes], color=couleurs, width=0.6)
    for i, l in enumerate(lignes):
        ax.text(i, l["couverture"] + 0.006, f"{100 * l['couverture']:.1f} %", ha="center", fontsize=8)
    ax.set_xticks(x, [l["nom"] for l in lignes], rotation=18, ha="right", fontsize=8)
    ax.set(ylim=(0.6, 1.0), ylabel="part des IC95 contenant le vrai prix",
           title="Calibrage des IC95 (1 000 répétitions indépendantes par estimateur)")
    ax.legend(fontsize=8, loc="lower right")
    return _sauver(fig, dossier, "couverture_ic95.png")
