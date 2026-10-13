"""Backtest VaR 99 % / ES 97,5 % a 1 jour de 5 methodes sur un portefeuille multi-actifs reel.

Usage (depuis le dossier du projet) :  python scripts/run_risque.py [--cache fichier.npz]
Sorties : results/risque_backtest.json, figures/risque_*.png
--cache : sauvegarde/relit les previsions brutes (pour regenerer les figures sans tout recalculer).
"""
import argparse
import json
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from mclab import risque as rk  # noqa: E402

STRESS = {
    "covid_mars_2020": ("2020-03-01", "2020-03-31"),
    "ukraine_mars_2022": ("2022-03-01", "2022-03-31"),
    "annee_2022": ("2022-01-01", "2022-12-31"),
}
# Okabe-Ito (lisible en daltonisme) ; une couleur fixe par methode
COULEURS = {"hs": "#0072B2", "normale": "#E69F00", "mc_t": "#009E73",
            "fhs": "#CC79A7", "copule_t": "#D55E00"}
MARQUEURS = {"hs": "o", "normale": "s", "mc_t": "^", "fhs": "D", "copule_t": "v"}
GRIS = "#8a8a8a"


def calculer_previsions(R, w, cache):
    if cache and Path(cache).is_file():
        z = np.load(cache)
        champs = ("var99", "var975", "es975", "queue")
        print(f"Previsions relues depuis {cache}")
        return {"jours": z["jours"], "echecs_garch": int(z["echecs_garch"]), "n_refits": int(z["n_refits"]),
                "duree_s": float(z["duree_s"]),
                "methodes": {m: {c: z[f"{m}_{c}"] for c in champs} for m in rk.METHODES}}
    t0 = time.time()
    out = rk.backtest_roulant(R, w, verbose=True)
    out["duree_s"] = time.time() - t0
    if cache:
        flat = {f"{m}_{c}": v for m, d in out["methodes"].items() for c, v in d.items()}
        np.savez(cache, jours=out["jours"], echecs_garch=out["echecs_garch"],
                 n_refits=out["n_refits"], duree_s=out["duree_s"], **flat)
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def style():
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                         "font.size": 9, "axes.titlesize": 10, "legend.frameon": False})


def fig_pnl_vs_var(dates, pnl, prev, bt, chemin):
    fig, axes = plt.subplots(len(rk.METHODES), 1, figsize=(12, 13), sharex=True, sharey=True)
    for ax, m in zip(axes, rk.METHODES):
        ax.plot(dates, pnl / 1e3, color=GRIS, lw=0.5, label="P&L réalisé")
        ax.plot(dates, -prev[m]["var99"] / 1e3, color=COULEURS[m], lw=1.1, label="−VaR 99 %")
        exc = bt[m]["exceptions"].astype(bool)
        ax.scatter(dates[exc], pnl[exc] / 1e3, s=16, color="black", zorder=3,
                   label=f"exceptions ({exc.sum()}, attendu {0.01 * exc.size:.0f})")
        ax.set_title(rk.LIBELLES[m], loc="left", color=COULEURS[m], fontweight="bold")
        ax.set_ylabel("k€")
        ax.legend(loc="lower left", ncol=3)
    axes[0].set_ylim(-640, 330)
    fig.suptitle("P&L quotidien réalisé vs −VaR 99 % à 1 jour (portefeuille 10 M€)", y=0.995)
    fig.tight_layout()
    fig.savefig(chemin)
    plt.close(fig)


def fig_exceptions_cumulees(dates, bt, chemin):
    T = len(dates)
    t = np.arange(1, T + 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.fill_between(dates, stats.binom.ppf(0.025, t, 0.01), stats.binom.ppf(0.975, t, 0.01),
                    color=GRIS, alpha=0.2, label="intervalle binomial 95 % (modèle exact)")
    ax.plot(dates, 0.01 * t, color="black", ls="--", lw=1.2, label="attendu (1 % × t)")
    for m in rk.METHODES:
        ax.plot(dates, np.cumsum(bt[m]["exceptions"]), color=COULEURS[m], lw=1.6,
                label=f"{rk.LIBELLES[m]} ({bt[m]['n_exceptions']})")
    ax.set_ylabel("exceptions cumulées (VaR 99 %)")
    ax.set_title("Exceptions cumulées vs attendu")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(chemin)
    plt.close(fig)


def fig_zoom(dates, pnl, prev, bt, debut, fin, titre, chemin):
    m_ = (dates >= pd.Timestamp(debut)) & (dates <= pd.Timestamp(fin))
    d = dates[m_]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(d, pnl[m_] / 1e3, color=GRIS, width=0.8, label="P&L réalisé", zorder=2)
    for m in rk.METHODES:
        ax.plot(d, -prev[m]["var99"][m_] / 1e3, color=COULEURS[m], lw=1.6, label=f"−VaR 99 % {rk.LIBELLES[m]}")
        e = bt[m]["exceptions"][m_].astype(bool)
        ax.scatter(d[e], pnl[m_][e] / 1e3, s=46, marker=MARQUEURS[m], color=COULEURS[m],
                   edgecolor="black", zorder=4)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("k€")
    ax.set_title(titre + "  (marqueurs = exceptions, un symbole par méthode)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(chemin)
    plt.close(fig)


def fig_tableau(res, chemin):
    entetes = ["Méthode", "Exceptions\n(taux ; attendu 1 %)", "Kupiec\np", "Christoffersen\nindép. p",
               "Cond. cov.\np", "Bâle : % fenêtres\nvertes", "Bâle : pire zone\n(max / 250 j)",
               "ES Z2\np", "Perte / ES\nsi > VaR 97,5 %"]
    lignes, couleurs = [], []
    rouge, vert, orange = "#f4c7b8", "#cde9d6", "#fbe3a8"
    for m in rk.METHODES:
        v, e = res[m]["backtest_var"], res[m]["backtest_es"]
        ch, b = v["christoffersen"], v["bale"]
        lignes.append([rk.LIBELLES[m], f"{v['n_exceptions']} ({100 * v['taux_exceptions']:.2f} %)",
                       f"{v['kupiec_p']:.3f}", f"{ch['p_ind']:.3f}", f"{ch['p_cc']:.3f}",
                       f"{100 * b['part_vert']:.0f} %", f"{b['pire_zone']} ({b['max_exceptions_250j']})",
                       f"{e['p_value']:.3f}", f"{e['ratio_perte_sur_es_apres_var975']:.2f}"])
        zc = {"vert": vert, "orange": orange, "rouge": rouge}[b["pire_zone"]]
        pc = lambda p: vert if p >= 0.05 else rouge
        couleurs.append(["white", "white", pc(v["kupiec_p"]), pc(ch["p_ind"]), pc(ch["p_cc"]),
                         "white", zc, pc(e["p_value"]), "white"])
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.axis("off")
    tb = ax.table(cellText=lignes, colLabels=entetes, cellColours=couleurs, loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1, 2.3)
    for c in range(len(entetes)):
        tb[0, c].set_facecolor("#e8e8e8")
        tb[0, c].set_text_props(fontweight="bold")
        tb[0, c].set_height(0.2)
    ax.set_title("Backtest VaR 99 % / ES 97,5 % à 1 jour — vert : test non rejeté à 5 %, rouge : rejeté",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(chemin)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def diagnostic_asymetrie(R, w, pas=160, n=100_000):
    """FHS (bootstrap joint, asymetrique) vs copule t (symetrique) : asymetrie, kurtosis et VaR 99 %
    du rendement simule du portefeuille, a des dates echantillonnees tous les `pas` jours."""
    lignes, params = [], None
    for t in range(rk.FENETRE, len(R), pas):
        Rwin = R[t - rk.FENETRE:t]
        params, _ = rk.ajuster_garch(Rwin, params)
        sg, z = rk.etat_garch(Rwin, params)
        Rc, nu_c = rk.estimer_copule_t(z)
        pf = (z[np.random.default_rng(t).integers(0, len(z), n)] * sg) @ w
        pc = rk.simuler_copule_t(Rc, nu_c, params[:, 3], sg, n, np.random.default_rng(t)) @ w
        lignes.append((nu_c, stats.skew(pf), stats.skew(pc), stats.kurtosis(pf), stats.kurtosis(pc),
                       -np.quantile(pf, 0.01), -np.quantile(pc, 0.01)))
    a = np.array(lignes)
    return {"n_dates": int(len(a)), "pas_jours": pas,
            "nu_copule_moyen": float(a[:, 0].mean()),
            "asymetrie_fhs": float(a[:, 1].mean()), "asymetrie_copule": float(a[:, 2].mean()),
            "exces_kurtosis_fhs": float(a[:, 3].mean()), "exces_kurtosis_copule": float(a[:, 4].mean()),
            "var99_copule_sur_fhs": float((a[:, 6] / a[:, 5]).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()
    (RACINE / "results").mkdir(exist_ok=True)
    (RACINE / "figures").mkdir(exist_ok=True)
    style()

    portefeuille = rk.PORTEFEUILLE
    assert abs(sum(portefeuille.values()) - 1.0) < 1e-12
    w = np.array(list(portefeuille.values()))
    dossier = rk.trouver_dossier_donnees()
    R = rk.rendements_log(portefeuille, dossier)
    print(f"Donnees : {dossier.name} | {len(R)} rendements quotidiens {R.index[0].date()} -> {R.index[-1].date()}")
    print("Portefeuille :", {a: f"{100 * p:.0f} %" for a, p in portefeuille.items()}, "| notionnel", rk.NOTIONNEL)

    out = calculer_previsions(R.values, w, args.cache)
    jours = out["jours"]
    dates = R.index[jours]
    L = rk.pertes(R.values[jours], w)
    pnl = -L
    prev = out["methodes"]

    bt, res = {}, {}
    for i, m in enumerate(rk.METHODES):
        p = prev[m]
        assert np.all(np.isfinite(p["var99"])) and np.all(np.isfinite(p["queue"]))
        bt[m] = rk.backtest_var(L, p["var99"])
        es = rk.z2_acerbi_szekely(L, p["var975"], p["es975"], p["queue"], seed=1000 + i)
        v = {k: x for k, x in bt[m].items() if k != "exceptions"}
        res[m] = {
            "libelle": rk.LIBELLES[m],
            "var99_moyenne_eur": float(p["var99"].mean()),
            "es975_moyenne_eur": float(p["es975"].mean()),
            "var99_moyenne_pct_notionnel": float(100 * p["var99"].mean() / rk.NOTIONNEL),
            "backtest_var": v,
            "backtest_es": es,
            "stress": {nom: rk.stress_fenetre(dates, L, p, d, f) for nom, (d, f) in STRESS.items()},
        }

    meta = {
        "portefeuille_poids": portefeuille,
        "notionnel_eur": rk.NOTIONNEL,
        "actifs_morts_exclus": list(rk.ACTIFS_MORTS),
        "donnees": f"data/processed/{dossier.name}/<ACTIF>_1h.csv -> cloture quotidienne (derniere barre du jour de trading, 17h NY)",
        "periode_rendements": [str(R.index[0].date()), str(R.index[-1].date())],
        "n_rendements": int(len(R)),
        "periode_backtest": [str(dates[0].date()), str(dates[-1].date())],
        "n_jours_backtest": int(len(dates)),
        "fenetre_jours": rk.FENETRE, "lambda_ewma": rk.LAMBDA, "n_simulations_mc": rk.N_SIM,
        "refit_garch_jours": rk.REFIT, "graine_base": rk.SEED,
        "mc_evalue_tous_les_jours": True,
        "garch_refits": out["n_refits"], "garch_ajustements_echoues_remplaces_par_precedents": out["echecs_garch"],
        "duree_calcul_s": round(out["duree_s"], 1),
        "stress_fenetres": STRESS,
        "perte_max_realisee_eur": float(L.max()), "date_perte_max": str(dates[int(np.argmax(L))].date()),
    }
    diag = diagnostic_asymetrie(R.values, w)
    # nombre de jours ou k methodes (sur 5) sont simultanement en exception
    n_exc_jour = np.sum([bt[m]["exceptions"] for m in rk.METHODES], axis=0)
    recoupement = {f"{k}_methodes_en_exception": int(np.sum(n_exc_jour == k)) for k in range(len(rk.METHODES) + 1)}
    chemin_json = RACINE / "results" / "risque_backtest.json"
    chemin_json.write_text(json.dumps({"meta": meta, "methodes": res, "diagnostic_fhs_vs_copule": diag,
                                       "recoupement_exceptions_jours": recoupement},
                                      indent=2, ensure_ascii=False), encoding="utf-8")

    fig_pnl_vs_var(dates, pnl, prev, bt, RACINE / "figures" / "risque_pnl_vs_var.png")
    fig_exceptions_cumulees(dates, bt, RACINE / "figures" / "risque_exceptions_cumulees.png")
    fig_zoom(dates, pnl, prev, bt, "2020-02-15", "2020-04-30", "Zoom COVID : 15 fév. – 30 avr. 2020",
             RACINE / "figures" / "risque_zoom_covid.png")
    fig_zoom(dates, pnl, prev, bt, "2022-02-15", "2022-04-30", "Zoom 2022 (Ukraine / pétrole) : 15 fév. – 30 avr.",
             RACINE / "figures" / "risque_zoom_2022.png")
    fig_tableau(res, RACINE / "figures" / "risque_tableau_recap.png")

    print(f"\nBacktest {meta['periode_backtest'][0]} -> {meta['periode_backtest'][1]} ({meta['n_jours_backtest']} jours)")
    print(f"{'methode':<18}{'exc':>5}{'taux%':>7}{'Kupiec p':>10}{'Chr ind p':>10}{'CC p':>8}{'Bale pire':>14}{'Z2 p':>8}{'ratio':>7}")
    for m in rk.METHODES:
        v, e = res[m]["backtest_var"], res[m]["backtest_es"]
        print(f"{rk.LIBELLES[m]:<18}{v['n_exceptions']:>5}{100 * v['taux_exceptions']:>7.2f}{v['kupiec_p']:>10.4f}"
              f"{v['christoffersen']['p_ind']:>10.4f}{v['christoffersen']['p_cc']:>8.4f}"
              f"{v['bale']['pire_zone'] + '(' + str(v['bale']['max_exceptions_250j']) + ')':>14}"
              f"{e['p_value']:>8.4f}{e['ratio_perte_sur_es_apres_var975']:>7.2f}")
    print("GARCH : refits", out["n_refits"], "| echecs", out["echecs_garch"], "| duree calcul", round(out["duree_s"]), "s")
    print("Ecrit :", chemin_json.relative_to(RACINE))


if __name__ == "__main__":
    main()
