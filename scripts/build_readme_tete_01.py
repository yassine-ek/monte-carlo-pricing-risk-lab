"""Tête du README : figure héro + blocs lus dans results/*.json, injectés entre marqueurs AUTO.

Produit :
  figures/hero_convergence_et_var.png   2 panneaux : convergence MC / QMC (validation.json) + exceptions de VaR par méthode (risque_backtest.json)
  README.md                             blocs AUTO:tete_glance, tete_cles, tete_validation, tete_risque, tete_lecture,
                                        variantes_pricing, variantes_vr, variantes_var
Usage : python scripts/build_readme_tete_01.py
Aucun nombre n'est tapé à la main : tout est lu dans results/validation.json, results/risque_backtest.json et
results/approfondissement.json. Le badge de CI du README est vivant (GitHub Actions) : aucun nombre de tests n'y figure.
"""
import json
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
from scipy import stats  # noqa: E402

RACINE = Path(__file__).resolve().parents[1]
R = RACINE / "results"
HERO = RACINE / "figures" / "hero_convergence_et_var.png"
# Lisibilité de la figure héro : 12,5 pt * 130 dpi / 72 * 50 % = 11,3 px une fois l'image réduite de moitié par GitHub.
DPI_HERO, POLICE_MIN, POLICE_TITRE = 130, 12.5, 14
# Okabe-Ito (lisible en daltonisme) : une couleur fixe par objet
COUL = {"mc": "#6b7280", "qmc_incrementale": "#E69F00", "qmc_pont": "#0072B2",
        "hs": "#6b7280", "normale": "#E69F00", "mc_t": "#CC79A7", "fhs": "#009E73", "copule_t": "#0072B2"}
CAS_TETE = ["bs_naif", "bs_qmc", "heston_qe", "heston_numba", "merton", "asia_geo_qmc", "asia_arith_ctrl", "otm_is",
            "barriere_pont", "barriere_naif", "lsm_put_50", "mlmc_gbm", "grecque_call_delta_pathwise"]
NOMS_VAR = {"hs": "Simulation historique", "normale": "Normale EWMA", "mc_t": "Monte Carlo Student-t",
            "fhs": "FHS (GARCH)", "copule_t": "Copule t + GARCH"}
NOMS_COURTS = {"hs": "Historique", "normale": "Normale EWMA", "mc_t": "MC Student-t", "fhs": "FHS (GARCH)", "copule_t": "Copule t"}
NOMS_EN = {"hs": "historical simulation", "normale": "EWMA normal", "mc_t": "Student-t Monte Carlo",
           "fhs": "filtered historical simulation (GARCH)", "copule_t": "t-copula + GARCH"}
DESC_VAR = {"hs": "quantile des 500 dernières pertes", "normale": "gaussienne, covariance EWMA (lambda = 0,94)",
            "mc_t": "Student multivarié, covariance EWMA", "fhs": "GARCH(1,1) + bootstrap de résidus standardisés",
            "copule_t": "marges GARCH + copule t"}


def charger(nom):
    return json.loads((R / nom).read_text(encoding="utf-8"))


def f(x, n=2):
    return f"{x:.{n}f}"


def fr(x, n=1):
    """Décimale à la française pour la prose (les tableaux gardent le point)."""
    return f"{x:.{n}f}".replace(".", ",")


def espace(x):
    """Séparateur de milliers à la française (espace)."""
    return f"{x:,.0f}".replace(",", " ")


def pct(x, n=1):
    return f"{fr(100 * x, n)} %"


# ------------------------------------------------------------------------------------------------ figure héro
def figure_hero(val, risque):
    """Deux panneaux, police >= POLICE_MIN : la figure reste lisible quand GitHub l'affiche à environ 60 % de sa taille."""
    plt.rcParams.update({"font.size": POLICE_MIN, "axes.labelsize": POLICE_MIN + 0.5, "axes.titlesize": POLICE_TITRE,
                         "xtick.labelsize": POLICE_MIN, "ytick.labelsize": POLICE_MIN, "legend.fontsize": POLICE_MIN})
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.2), gridspec_kw={"width_ratios": [1.05, 1]})
    a = ax[0]
    conv = val["convergence"]
    N = np.array(conv["N"], float)
    libelles = {"mc": "Monte Carlo", "qmc_incrementale": "QMC incréments", "qmc_pont": "QMC + pont"}
    for k in ("mc", "qmc_incrementale", "qmc_pont"):
        a.loglog(N, conv["rmse"][k], "o-", ms=5, lw=2, color=COUL[k], label=f"{libelles[k]} ({conv['pentes'][k]:.2f})".replace(".", ",").replace("-", "−"))
    a.loglog(N, conv["rmse"]["mc"][0] * (N / N[0]) ** -0.5, "--", color="#444444", alpha=0.6, lw=1.2, label="références −1/2 et −1")
    a.loglog(N, conv["rmse"]["qmc_pont"][0] * (N / N[0]) ** -1.0, "--", color="#444444", alpha=0.6, lw=1.2)
    a.set(xlabel="nombre de chemins N", ylabel="erreur RMSE sur le prix", title="Erreur de pricing : MC contre QMC")
    a.yaxis.set_minor_formatter(NullFormatter())
    a.legend(loc="lower left", frameon=False, handlelength=1.6, borderaxespad=0.2)
    a.grid(alpha=0.3, which="both")
    b = ax[1]
    meta, mets = risque["meta"], risque["methodes"]
    T, p = meta["n_jours_backtest"], 0.01
    bas, haut = stats.binom.ppf(0.025, T, p), stats.binom.ppf(0.975, T, p)
    cles = list(mets)
    exc = [mets[k]["backtest_var"]["n_exceptions"] for k in cles]
    y = np.arange(len(cles))[::-1]
    b.axvspan(bas, haut, color="#009E73", alpha=0.18)
    b.axvline(T * p, color="k", lw=1.2, ls="--")
    b.barh(y, exc, color=[COUL[k] for k in cles], height=0.62)
    for yi, e, k in zip(y, exc, cles):
        rejet = mets[k]["backtest_var"]["kupiec_p"] < 0.05
        b.text(e + max(exc) * 0.02, yi, f"{e}{' *' if rejet else ''}", va="center", fontweight="bold" if rejet else "normal")
    b.set_yticks(y, [NOMS_COURTS[k] for k in cles])
    b.set(xlabel=f"exceptions sur {espace(T)} jours (attendu : {T * p:.0f})\n* rejeté par Kupiec à 5 %",
          title="Exceptions de VaR 99 %", xlim=(0, max(exc) * 1.22))
    b.xaxis.label.set_fontsize(POLICE_MIN)
    b.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


def taille_police_min(fig):
    """Plus petite police (pt) de tous les textes visibles de la figure : texte d'axe, graduations, légendes, annotations."""
    return min(t.get_fontsize() for t in fig.findobj(Text) if t.get_text().strip() and t.get_visible())


def sauver_hero(val, risque):
    fig = figure_hero(val, risque)
    assert taille_police_min(fig) >= POLICE_MIN, "police de la figure héro sous le seuil de lisibilité"
    fig.savefig(HERO, dpi=DPI_HERO)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ comptes communs
def comptes_validation(val):
    tous = val["cas"]
    demos = [c for c in tous if c["attendu_echec"]]
    valides = [c for c in tous if not c["attendu_echec"] and c["verdict_ok"]]
    stricts = [c for c in valides if c["tolerance_biais"] == 0]
    toleres = [c for c in valides if c["tolerance_biais"] != 0]
    return tous, demos, valides, stricts, toleres


def reussites_risque(risque):
    """Méthodes qui passent les quatre tests (Kupiec, indépendance, cc, ES) à 5 %."""
    return [k for k, m in risque["methodes"].items()
            if min(m["backtest_var"]["kupiec_p"], m["backtest_var"]["christoffersen"]["p_ind"], m["backtest_var"]["christoffersen"]["p_cc"],
                   m["backtest_es"]["p_value"]) >= 0.05]


# ------------------------------------------------------------------------------------------------ blocs de tête
def bloc_glance(val, risque):
    """Trois lignes en anglais (deux espaces en fin de ligne = saut de ligne Markdown)."""
    tous, demos, valides, _, _ = comptes_validation(val)
    meta, passent = risque["meta"], reussites_risque(risque)
    T = meta["n_jours_backtest"]
    ligne_risque = (f"Five VaR/ES methods backtested over {T:,} days (Kupiec, Christoffersen, Basel, Acerbi-Székely): "
                    + (f"only {NOMS_EN[passent[0]]} passes all four tests, on a linear 7-asset portfolio." if len(passent) == 1
                       else f"{len(passent)} of 5 pass all four tests, on a linear 7-asset portfolio."))
    return "\n".join([
        f"> **At a glance.** A Monte Carlo option-pricing and market-risk lab (Python, numpy, scipy, numba): {len(valides)} validated cases against independent references "
        f"plus {len(demos)} deliberate failure demos, and a {int(val['couverture_ic95'][0]['repetitions']):,}-repetition coverage test of the 95% intervals.  ",
        f"> {ligne_risque}  ",
        "> No trading strategy and no transaction costs: every number is regenerated from fixed seeds (`python scripts/run_all.py`) and checked by `pytest` and CI.",
    ])


def bloc_cles(val, risque, appro):
    """Trois puces : chaque chiffre est accompagné de sa réserve."""
    tous, demos, valides, stricts, toleres = comptes_validation(val)
    pire = max(stricts, key=lambda c: abs(c["z"]))
    heston = next(c for c in tous if c["id"] == "heston_qe")
    biais_invisible = 3 * heston["erreur_type"] / heston["reference"]
    z_heston = f"{heston['z']:+.2f}".replace(".", ",")
    cov = [c for c in val["couverture_ic95"] if "contre-exemple" not in c["nom"]]
    ctr = [c for c in val["couverture_ic95"] if "contre-exemple" in c["nom"]][0]
    sigma_pt = 196 * cov[0]["sigma_binomial"]
    meta, passent = risque["meta"], reussites_risque(risque)
    T = meta["n_jours_backtest"]
    q7 = appro["blocs"]["Q7"]["chiffres"]["kupiec"]["points"]["250"]["puissance_exacte"]
    if len(passent) == 1:
        k = passent[0]
        resultat = (f"Une seule méthode passe les quatre tests ({NOMS_VAR[k]} : {risque['methodes'][k]['backtest_var']['n_exceptions']} exceptions "
                    f"pour {T * 0.01:.0f} attendues), les quatre autres sont rejetées et dites telles.")
    else:
        resultat = f"{len(passent)} méthodes sur 5 passent les quatre tests, les autres sont rejetées et dites telles."
    return "\n".join([
        f"- **Pricing : {len(valides)} cas validés + {len(demos)} contre-exemples volontaires.** Chaque prix Monte Carlo est à moins de 3 erreurs-types d'une référence "
        f"indépendante ({len(stricts)} cas sans biais toléré, pire |z| = {fr(abs(pire['z']), 2)} ; {len(toleres)} avec un biais de discrétisation documenté). "
        f"*Réserve* : un |z| compatible établit la cohérence à l'erreur-type près, pas l'exactitude ; sur Heston QE (z = {z_heston}, "
        f"erreur-type {fr(heston['erreur_type'], 4)}) un biais de {fr(100 * biais_invisible, 1)} % du prix resterait invisible.",
        f"- **Barres d'erreur honnêtes.** L'IC95 contient la vraie valeur {pct(min(c['couverture'] for c in cov))} à {pct(max(c['couverture'] for c in cov))} du temps "
        f"sur {espace(cov[0]['repetitions'])} répétitions (cinq estimateurs) ; le contre-exemple de l'événement rare est montré ({pct(ctr['couverture'])}). "
        f"*Réserve* : fluctuation binomiale de ±{fr(sigma_pt, 1)} point, et couverture mesurée avec peu de chemins par répétition, pas à la précision des tables de prix.",
        f"- **Risque : cinq méthodes de VaR/ES backtestées sur {espace(T)} jours.** {resultat} "
        f"*Réserve* : portefeuille de 7 positions longues linéaires (le Monte Carlo y simule la loi jointe des rendements, il ne réévalue pas d'options), P&L en devise de cotation sans conversion EUR ; "
        f"avec {T * 0.01:.0f} exceptions attendues, un backtest d'un an ne détecte un modèle à 2 % d'exceptions que {pct(q7, 0)} du temps (Q7) et une p-value proche de 0,05 n'est pas décisive.",
    ])


def tableau_validation(val):
    cas = {c["id"]: c for c in val["cas"]}
    lignes = ["| Cas | Méthode | Référence | Estimation | Erreur-type | z | Verdict |", "|---|---|---|---|---|---|---|"]
    for i in CAS_TETE:
        c = cas[i]
        verdict = "échec attendu (démo)" if c["attendu_echec"] else ("OK" if c["verdict_ok"] else "ÉCHEC")
        lignes.append(f"| {c['theme']} | {c['methode']} | {f(c['reference'], 4)} | {f(c['estimation'], 4)} | {f(c['erreur_type'], 4)} | "
                      f"{c['z']:+.2f} | {verdict} |")
    tous, demos, valides, stricts, toleres = comptes_validation(val)
    pire = max(stricts, key=lambda c: abs(c["z"]))
    lignes += ["", f"***{len(valides)} cas validés + {len(demos)} contre-exemples volontaires** (sur {len(tous)} comparaisons exécutées ; MC naïf sur barrière discrète, pathwise sur "
                   f"digitale). Parmi les {len(valides)} cas validés, {len(stricts)} n'ont aucun biais de discrétisation toléré (pire |z| = {fr(abs(pire['z']), 2)} : {pire['theme']}, "
                   f"{pire['methode']}) et {len(toleres)} ont un biais documenté, dont la borne haute duale du put américain, jugée par un encadrement et non par une égalité. "
                   f"Un |z| compatible établit la cohérence à l'erreur-type près, pas l'exactitude. Toutes les lignes : [docs/validation.md](docs/validation.md). "
                   f"Frais : aucun (le labo ne trade pas).*"]
    return "\n".join(lignes)


def tableau_risque(risque):
    meta = risque["meta"]
    T = meta["n_jours_backtest"]
    lignes = [f"Backtest de la VaR 99 % à 1 jour sur {espace(T)} jours ({meta['periode_backtest'][0]} → {meta['periode_backtest'][1]}), "
              f"portefeuille de {meta['notionnel_eur'] / 1e6:.0f} M€, **{fr(T * 0.01, 1)} exceptions attendues**. p-values : rejet à 5 % si p < 0,05 (en gras).",
              "", "| Méthode | Exceptions | Kupiec (p) | Christoffersen indép. (p) | Christoffersen cc (p) | ES 97,5 % Z2 (p) | Verdict |",
              "|---|---|---|---|---|---|---|"]

    def cell(p):
        return f"**{p:.3f}**" if p < 0.05 else f"{p:.3f}"

    for k, m in risque["methodes"].items():
        b, e = m["backtest_var"], m["backtest_es"]
        ps = [b["kupiec_p"], b["christoffersen"]["p_ind"], b["christoffersen"]["p_cc"], e["p_value"]]
        rejets = [n for n, p in zip(("Kupiec", "indépendance", "cc", "ES"), ps) if p < 0.05]
        verdict = "passe les 4 tests" if not rejets else "rejetée : " + ", ".join(rejets)
        lignes.append(f"| {NOMS_VAR[k]} | {b['n_exceptions']} (attendu {T * 0.01:.0f}) | "
                      f"{cell(ps[0])} | {cell(ps[1])} | {cell(ps[2])} | {cell(ps[3])} | {verdict} |")
    return "\n".join(lignes)


def lecture_overfit(val, risque, appro):
    """Section « Overfit ou pas ? » : ce qui est fixé avant, ce qui n'est pas sélectionné, ce que les p-values valent."""
    meta = risque["meta"]
    ps = [p for m in risque["methodes"].values() for p in (m["backtest_var"]["kupiec_p"], m["backtest_var"]["christoffersen"]["p_ind"],
                                                         m["backtest_var"]["christoffersen"]["p_cc"], m["backtest_es"]["p_value"])]
    proches = sum(0.03 <= p <= 0.10 for p in ps)
    tous, demos, valides, _, _ = comptes_validation(val)
    faux_positifs = len(valides) * 2 * stats.norm.sf(3)
    q7 = appro["blocs"]["Q7"]["chiffres"]["kupiec"]["points"]["250"]["puissance_exacte"]
    return "\n".join([
        f"- **Rien n'est réglé sur le backtest du risque** : fenêtre {meta['fenetre_jours']} j, lambda = {fr(meta['lambda_ewma'], 2)}, refit GARCH {meta['refit_garch_jours']} j, "
        f"{espace(meta['n_simulations_mc'])} tirages sont fixés avant les calculs. Il n'y a pas de découpage TRAIN / VAL / TEST parce que rien n'est optimisé : le backtest est "
        "glissant et causal (la prévision du jour t n'utilise que les rendements jusqu'à t-1, ce qu'un test de non-fuite vérifie).",
        f"- **Aucune sélection a posteriori** : les 5 méthodes sont publiées, rejetées comprises. 5 méthodes x 4 tests = {len(ps)} p-values sans correction pour tests multiples, "
        f"dont {proches} tombent entre 0,03 et 0,10 : elles ne sont pas décisives. Un backtest d'un an ne détecte un modèle à 2 % d'exceptions que {pct(q7, 0)} du temps (Q7).",
        f"- **Seuil de pricing fixé avant** : un cas est validé si |z| < 3. Sous H0, {len(valides)} comparaisons produisent environ {fr(faux_positifs, 1)} dépassement par hasard : "
        "un écart isolé de 2 à 2,5 sigma n'est pas un signal, et les résultats sont déterministes à graine donnée.",
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
            verdict = "contient un contre-exemple volontaire : " + ", ".join(c["methode"] for c in cs if c["attendu_echec"])
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


def blocs(val, risque, appro):
    return (("tete_glance", bloc_glance(val, risque)), ("tete_cles", bloc_cles(val, risque, appro)),
            ("tete_validation", tableau_validation(val)), ("tete_risque", tableau_risque(risque)),
            ("tete_lecture", lecture_overfit(val, risque, appro)), ("variantes_pricing", variantes_pricing(val)),
            ("variantes_vr", variantes_vr(val, appro)), ("variantes_var", variantes_var(risque)))


def main():
    val, risque, appro = charger("validation.json"), charger("risque_backtest.json"), charger("approfondissement.json")
    sauver_hero(val, risque)
    p = RACINE / "README.md"
    md = p.read_text(encoding="utf-8")
    for cle, contenu in blocs(val, risque, appro):
        md = injecter(md, cle, contenu)
    p.write_text(md, encoding="utf-8")
    print("README (tête) et figure héro régénérés")


if __name__ == "__main__":
    main()
