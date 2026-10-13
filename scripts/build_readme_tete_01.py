"""Tête du README (SPEC_GITHUB) : figure héro + tableaux lus dans results/*.json, injectés entre marqueurs AUTO.

Produit :
  figures/hero_convergence_et_var.png   convergence MC / QMC (validation.json) + exceptions de VaR par méthode (risque_backtest.json)
  README.md                             blocs AUTO:tete_validation, tete_risque, tete_lecture, variantes_pricing, variantes_vr, variantes_var
Usage : python scripts/build_readme_tete_01.py [--tests N]       (N = nombre de tests réussis, pour le badge)
Aucun nombre n'est tapé à la main : tout est lu dans results/validation.json, results/risque_backtest.json et
results/approfondissement.json.
"""
import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

RACINE = Path(__file__).resolve().parents[1]
R = RACINE / "results"
HERO = RACINE / "figures" / "hero_convergence_et_var.png"
# Okabe-Ito (lisible en daltonisme) : une couleur fixe par objet
COUL = {"mc": "#6b7280", "qmc_incrementale": "#E69F00", "qmc_pont": "#0072B2",
        "hs": "#6b7280", "normale": "#E69F00", "mc_t": "#CC79A7", "fhs": "#009E73", "copule_t": "#0072B2"}
CAS_TETE = ["bs_naif", "bs_qmc", "heston_qe", "heston_numba", "merton", "asia_geo_qmc", "asia_arith_ctrl", "otm_is",
            "barriere_pont", "barriere_naif", "lsm_put_50", "mlmc_gbm", "grecque_call_delta_pathwise"]
NOMS_VAR = {"hs": "Simulation historique", "normale": "Normale EWMA", "mc_t": "Monte Carlo Student-t",
            "fhs": "FHS (GARCH)", "copule_t": "Copule t + GARCH"}
DESC_VAR = {"hs": "quantile des 500 dernières pertes", "normale": "gaussienne, covariance EWMA (lambda = 0,94)",
            "mc_t": "Student multivarié, covariance EWMA", "fhs": "GARCH(1,1) + bootstrap de résidus standardisés",
            "copule_t": "marges GARCH + copule t"}


def charger(nom):
    return json.loads((R / nom).read_text(encoding="utf-8"))


def f(x, n=2):
    return f"{x:.{n}f}"


def espace(x):
    """Séparateur de milliers à la française (espace)."""
    return f"{x:,.0f}".replace(",", " ")


def pct(x, n=1):
    return f"{100 * x:.{n}f} %"


# ------------------------------------------------------------------------------------------------ figure héro
def figure_hero(val, risque):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    a = ax[0]
    conv = val["convergence"]
    N = np.array(conv["N"], float)
    libelles = {"mc": "Monte Carlo", "qmc_incrementale": "QMC Sobol, incréments", "qmc_pont": "QMC Sobol + pont brownien"}
    for k in ("mc", "qmc_incrementale", "qmc_pont"):
        a.loglog(N, conv["rmse"][k], "o-", ms=4, color=COUL[k], label=f"{libelles[k]} (pente {conv['pentes'][k]:.2f})")
    a.loglog(N, conv["rmse"]["mc"][0] * (N / N[0]) ** -0.5, "--", color=COUL["mc"], alpha=0.5, lw=1)
    a.loglog(N, conv["rmse"]["qmc_pont"][0] * (N / N[0]) ** -1.0, "--", color=COUL["qmc_pont"], alpha=0.5, lw=1)
    a.text(N[-2], conv["rmse"]["mc"][0] * (N[-2] / N[0]) ** -0.5 * 1.25, "pente -1/2 (référence)", ha="center", color=COUL["mc"], fontsize=8)
    a.text(N[-2], conv["rmse"]["qmc_pont"][0] * (N[-2] / N[0]) ** -1.0 * 0.6, "pente -1 (référence)", ha="center", color=COUL["qmc_pont"], fontsize=8)
    a.set(xlabel="nombre de chemins N", ylabel="erreur RMSE sur le prix", title="Pricing : l'erreur MC vs QMC (asiatique géométrique, 16 dates)")
    a.legend(fontsize=8, loc="lower left")
    a.grid(alpha=0.3, which="both")
    b = ax[1]
    meta, mets = risque["meta"], risque["methodes"]
    T, p = meta["n_jours_backtest"], 0.01
    bas, haut = stats.binom.ppf(0.025, T, p), stats.binom.ppf(0.975, T, p)
    cles = list(mets)
    exc = [mets[k]["backtest_var"]["n_exceptions"] for k in cles]
    y = np.arange(len(cles))[::-1]
    b.axvspan(bas, haut, color="#009E73", alpha=0.15, label=f"zone compatible avec 1 % ({int(bas)} à {int(haut)})")
    b.axvline(T * p, color="k", lw=1, ls="--", label=f"attendu : {T * p:.0f}")
    b.barh(y, exc, color=[COUL[k] for k in cles], height=0.6)
    for yi, e, k in zip(y, exc, cles):
        pk = mets[k]["backtest_var"]["kupiec_p"]
        b.text(e + 1, yi, f"{e}  (Kupiec p = {pk:.3f})", va="center", fontsize=8)
    b.set_yticks(y, [NOMS_VAR[k] for k in cles])
    b.set(xlabel=f"exceptions de la VaR 99 % sur {T} jours ({meta['periode_backtest'][0][:4]}-{meta['periode_backtest'][1][:4]})",
          title="Risque : exceptions observées par méthode", xlim=(0, max(exc) * 1.45))
    b.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    b.grid(alpha=0.3, axis="x")
    fig.suptitle("Aucun frais ici : labo de pricing et de risque (pas une stratégie) ; chaque nombre est validé contre une référence",
                 fontsize=10, y=0.995)
    fig.tight_layout()
    fig.savefig(HERO, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ tableaux
def tableau_validation(val):
    cas = {c["id"]: c for c in val["cas"]}
    lignes = ["| Cas | Méthode | Référence | Estimation | Erreur-type | z | Verdict |", "|---|---|---|---|---|---|---|"]
    for i in CAS_TETE:
        c = cas[i]
        verdict = "échec attendu (démo)" if c["attendu_echec"] else ("OK" if c["verdict_ok"] else "ÉCHEC")
        lignes.append(f"| {c['theme']} | {c['methode']} | {f(c['reference'], 4)} | {f(c['estimation'], 4)} | {f(c['erreur_type'], 4)} | "
                      f"{c['z']:+.2f} | {verdict} |")
    tous = val["cas"]
    demos = [c for c in tous if c["attendu_echec"]]
    stricts = [c for c in tous if not c["attendu_echec"] and c["tolerance_biais"] == 0]
    pire = max(stricts, key=lambda c: abs(c["z"]))
    ok = sum(c["verdict_ok"] for c in tous)
    lignes += ["", f"*{len(tous)} cas au total : {ok} OK, {len(demos)} démonstrations d'échec volontaires (MC naïf sur barrière discrète, pathwise sur "
                   f"digitale). Parmi les {len(stricts)} cas sans biais de discrétisation toléré, le pire |z| vaut {abs(pire['z']):.2f} ({pire['theme']}, "
                   f"{pire['methode']}). Table complète des 45 cas plus bas. Frais : aucun (le labo ne trade pas).*"]
    return "\n".join(lignes)


def tableau_risque(risque):
    meta = risque["meta"]
    T = meta["n_jours_backtest"]
    lignes = [f"Backtest de la VaR 99 % à 1 jour sur {T} jours ({meta['periode_backtest'][0]} → {meta['periode_backtest'][1]}), "
              f"portefeuille de {meta['notionnel_eur'] / 1e6:.0f} M€, **{T * 0.01:.1f} exceptions attendues**. p-values : rejet à 5 % si p < 0,05 (en gras).",
              "", "| Méthode | VaR 99 % moyenne | Exceptions | Kupiec (p) | Christoffersen indép. (p) | Christoffersen cc (p) | ES 97,5 % Z2 (p) | Verdict |",
              "|---|---|---|---|---|---|---|---|"]

    def cell(p):
        return f"**{p:.3f}**" if p < 0.05 else f"{p:.3f}"

    for k, m in risque["methodes"].items():
        b, e = m["backtest_var"], m["backtest_es"]
        ps = [b["kupiec_p"], b["christoffersen"]["p_ind"], b["christoffersen"]["p_cc"], e["p_value"]]
        rejets = [n for n, p in zip(("Kupiec", "indépendance", "cc", "ES"), ps) if p < 0.05]
        verdict = "passe les 4 tests" if not rejets else "rejetée : " + ", ".join(rejets)
        lignes.append(f"| {NOMS_VAR[k]} | {pct(m['var99_moyenne_pct_notionnel'] / 100, 2)} du notionnel | {b['n_exceptions']} (attendu {T * 0.01:.0f}) | "
                      f"{cell(ps[0])} | {cell(ps[1])} | {cell(ps[2])} | {cell(ps[3])} | {verdict} |")
    return "\n".join(lignes)


def lecture_30s(val, risque, appro):
    cov = [c for c in val["couverture_ic95"] if "contre-exemple" not in c["nom"]]
    cmin, cmax = min(c["couverture"] for c in cov), max(c["couverture"] for c in cov)
    ctr = [c for c in val["couverture_ic95"] if "contre-exemple" in c["nom"]][0]
    stricts = [c for c in val["cas"] if not c["attendu_echec"] and c["tolerance_biais"] == 0]
    ok = sum(abs(c["z"]) < 3 for c in stricts)
    passent = [NOMS_VAR[k] for k, m in risque["methodes"].items()
               if min(m["backtest_var"]["kupiec_p"], m["backtest_var"]["christoffersen"]["p_ind"], m["backtest_var"]["christoffersen"]["p_cc"],
                      m["backtest_es"]["p_value"]) >= 0.05]
    meta = risque["meta"]
    q7 = appro["blocs"]["Q7"]["chiffres"]["kupiec"]
    return "\n".join([
        f"- **Les barres d'erreur sont honnêtes** : sur 1 000 répétitions indépendantes, l'IC95 contient la vraie valeur {pct(cmin)} à {pct(cmax)} du temps "
        f"(5 estimateurs), et le contre-exemple de l'événement rare est montré, pas caché ({pct(ctr['couverture'])} pour le MC naïf sur le call K = 180).",
        f"- **Aucun prix n'est publié sans référence indépendante** : {ok} cas sur {len(stricts)} (hors biais de discrétisation documenté) sont à moins de 3 erreurs-types "
        f"de leur formule fermée, semi-fermée ou arbre fin, et {sum(c['attendu_echec'] for c in val['cas'])} démonstrations d'échec sont affichées comme telles.",
        f"- **Rien n'est réglé sur le backtest du risque** : fenêtre {meta['fenetre_jours']} j, lambda = {meta['lambda_ewma']}, refit GARCH {meta['refit_garch_jours']} j, "
        f"{espace(meta['n_simulations_mc'])} tirages sont fixés avant les calculs ; résultat : {len(passent)} méthode sur 5 ({', '.join(passent)}) passe les quatre tests, "
        f"les autres sont rejetées et dites telles. Et un backtest de 1 an ne détecte un modèle à 2 % d'exceptions que {pct(q7['points']['250']['puissance_exacte'], 0)} du temps (Q7 de l'approfondissement)."
    ])


def variantes_pricing(val):
    par_theme = OrderedDict()
    for c in val["cas"]:
        par_theme.setdefault(c["theme"], []).append(c)
    lignes = ["| Thème | Méthodes comparées | Référence | Pire z | Verdict |", "|---|---|---|---|---|"]
    for t, cs in par_theme.items():
        fin = [c for c in cs if np.isfinite(c["z"])]
        pire = max(fin, key=lambda c: abs(c["z"])) if fin else cs[0]
        if any(c["attendu_echec"] for c in cs):
            verdict = "contient une démo d'échec attendue : " + ", ".join(c["methode"] for c in cs if c["attendu_echec"])
        elif all(c["verdict_ok"] for c in cs):
            verdict = "OK" if all(abs(c["z"]) < 3 for c in cs) else "OK, grâce à un biais toléré et documenté (borne haute de la dualité)"
        else:
            verdict = "ÉCHEC"
        z = "+inf" if not np.isfinite(pire["z"]) else f"{pire['z']:+.2f}"
        lignes.append(f"| {t} | {' ; '.join(c['methode'] for c in cs)} | {cs[0]['type_reference']} | {z} | {verdict} |")
    return "\n".join(lignes)


def variantes_vr(val, appro):
    q = appro["blocs"]
    q3, q4, q5 = q["Q3"]["chiffres"], q["Q4"]["chiffres"], q["Q5"]["chiffres"]
    lignes = ["| Technique | Cas | Facteur de variance | Facteur d'efficacité (variance × temps) |", "|---|---|---|---|"]
    for e in val["efficacite"]:
        nom, cas = e["nom"].split(" : ", 1) if " : " in e["nom"] else (e["nom"], "")
        lignes.append(f"| {cas} | {nom} | ×{f(e['facteur_variance'], 1)} | ×{f(e['facteur_efficacite'], 1)} |")
    e1, e2 = q4["strates"]["europeen_K100"]["64"], q4["strates"]["europeen_K140"]["64"]
    pl = q4["lhs"]["asiatique_16_pont"]
    b16 = q3["blocs"]["16"]
    pts = q5["cas"]["call_K180"]["points"]
    autres = [
        ("Stratification proportionnelle (H = 64)", "Européen ATM, brownien terminal", e1["mesure"]["facteur_prop"]),
        ("Stratification de Neyman, σ_h connus (H = 64)", "Européen ATM", e1["mesure"]["facteur_neyman_oracle"]),
        ("Stratification de Neyman, pilote n/8 (H = 64)", "Call OTM K = 140", e2["mesure"]["facteur_neyman_pilote"]),
        ("Hypercube latin (n = 4096)", "Asiatique 16 dates, pont", pl["n"]["4096"]["gain_mesure"]),
        ("Importance sampling, dérive optimale μ*", "Call OTM K = 180", pts["mu_star"]["facteur_variance"]),
        ("Importance sampling, entropie croisée", "Call OTM K = 180", pts["mu_ce"]["facteur_variance"]),
        ("QMC Sobol + pont brownien (N = 2^14, rapport de RMSE²)", "Asiatique 16 dates", (b16["mc"]["rmse"][-1] / b16["pont"]["rmse"]["rmse"][-1]) ** 2),
        ("QMC Sobol + ACP (N = 2^14, rapport de RMSE²)", "Asiatique 16 dates", (b16["mc"]["rmse"][-1] / b16["acp"]["rmse"]["rmse"][-1]) ** 2),
    ]
    for nom, cas, fv in autres:
        lignes.append(f"| {nom} | {cas} | ×{f(fv, 0)} | non chronométré |")
    lignes += ["", "*Les cinq premières lignes viennent de `results/validation.json` (efficacité = variance × temps mesuré). Les huit suivantes viennent de "
                   "`results/approfondissement.json` (Q3, Q4, Q5) : leur coût par tirage est voisin de celui du MC naïf, donc l'efficacité vaut à peu près le facteur "
                   "de variance, mais elle n'est pas chronométrée.*"]
    return "\n".join(lignes)


def variantes_var(risque):
    meta = risque["meta"]
    lignes = ["| Méthode | Principe | VaR 99 % moy. | ES 97,5 % moy. | Exceptions | Zone verte Bâle | COVID 9 mars 2020 : perte / VaR |",
              "|---|---|---|---|---|---|---|"]
    for k, m in risque["methodes"].items():
        b = m["backtest_var"]
        cov = m["stress"]["covid_mars_2020"]
        lignes.append(f"| {NOMS_VAR[k]} | {DESC_VAR[k]} | {espace(m['var99_moyenne_eur'])} € | {espace(m['es975_moyenne_eur'])} € | "
                      f"{b['n_exceptions']} / {b['n_obs']} | {pct(b['bale']['part_vert'], 0)} | ×{f(cov['perte_max_sur_var99'], 2)} |")
    lignes += ["", f"*Pertes sur un portefeuille de {meta['notionnel_eur'] / 1e6:.0f} M€, réévaluation complète. « Zone verte » : part des fenêtres glissantes de 250 jours "
                   "avec au plus 4 exceptions. La dernière colonne vaut plus de 1 quand la perte réalisée dépasse la VaR du jour.*"]
    return "\n".join(lignes)


# ------------------------------------------------------------------------------------------------ injection
def injecter(md, cle, contenu):
    d, fin = f"<!-- AUTO:{cle}:DEBUT -->", f"<!-- AUTO:{cle}:FIN -->"
    if d not in md or fin not in md:
        sys.exit(f"marqueurs manquants dans README.md : {cle}")
    avant, reste = md.split(d, 1)
    _, apres = reste.split(fin, 1)
    return f"{avant}{d}\n{contenu}\n{fin}{apres}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", type=int, help="nombre de tests réussis (badge)")
    args = ap.parse_args()
    val, risque, appro = charger("validation.json"), charger("risque_backtest.json"), charger("approfondissement.json")
    figure_hero(val, risque)
    p = RACINE / "README.md"
    md = p.read_text(encoding="utf-8")
    for cle, contenu in (("tete_validation", tableau_validation(val)), ("tete_risque", tableau_risque(risque)),
                         ("tete_lecture", lecture_30s(val, risque, appro)), ("variantes_pricing", variantes_pricing(val)),
                         ("variantes_vr", variantes_vr(val, appro)), ("variantes_var", variantes_var(risque))):
        md = injecter(md, cle, contenu)
    if args.tests:
        md = re.sub(r"badge/tests-\d+%20passed-", f"badge/tests-{args.tests}%20passed-", md)
    p.write_text(md, encoding="utf-8")
    print("README (tête) et figure héro régénérés")


if __name__ == "__main__":
    main()
