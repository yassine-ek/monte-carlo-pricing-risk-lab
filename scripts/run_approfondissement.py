"""Relance tout l'approfondissement : results/approfondissement.json + figures/appro_*.png + section du README.

Usage : python scripts/run_approfondissement.py [--only Q1,Q4] [--sans-readme]     (environ 7 minutes pour tout)
        python scripts/run_approfondissement.py --readme-seul                       (réécrit la section du README depuis le JSON, sans calcul)
Toutes les graines sont fixées : deux exécutions donnent les mêmes nombres (hors temps de calcul et chronométrages de Q6).
Le JSON contient un bloc par question : question, théorème, ce qu'on attend, chiffres, conclusion, durée.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "scripts"))

import numpy as np  # noqa: E402

import figures_appro as fa  # noqa: E402
from mclab import experiences  # noqa: E402

RESULTATS, FIGURES = RACINE / "results" / "approfondissement.json", RACINE / "figures"
MARQUEUR_DEBUT, MARQUEUR_FIN = "<!-- AUTO:approfondissement:DEBUT -->", "<!-- AUTO:approfondissement:FIN -->"


def f(x, n=2):
    return f"{x:.{n}f}".replace(".", ",")


def ic(v, n=2):
    return f"[{f(v[0], n)} ; {f(v[1], n)}]"


def lignes_readme(cle, b):
    """3 à 6 lignes de réponse, lues dans le bloc JSON `b` (aucun nombre n'est recopié à la main)."""
    c = b["chiffres"]
    if cle == "Q1":
        g, ci, h = c["gbm"], c["cir"], c["heston"]["feller_viole"]
        return [
            f"GBM, erreur forte (même brownien) : Euler en dt^{f(g['euler']['fort_l1']['pente'], 2)} {ic(g['euler']['fort_l1']['ic95'], 2)} (théorie 1/2) ; Milstein en dt^{f(g['milstein']['fort_l2']['pente'], 2)} en L2 "
            f"et dt^{f(g['milstein']['fort_l1']['pente'], 2)} en L1 (théorie 1 ; la pente locale L1 vaut {f(c['pentes_locales_gbm']['milstein']['fort_l1'][-1], 2)} aux pas les plus fins).",
            f"GBM, erreur faible : Euler {f(g['euler']['faible_call']['wls']['p'], 2)} ± {f(1.96 * g['euler']['faible_call']['wls']['se_p'], 2)} (call, bruité), Milstein {f(g['milstein']['faible_call']['pente'], 2)} ; "
            f"sans bruit (E[S^2] exact) {f(g['euler']['faible_x2_exact']['pente'], 3)} et {f(g['milstein']['faible_x2_exact']['pente'], 3)} : ordre faible 1 pour les deux.",
            f"CIR : les ordres du théorème exigent Feller. Feller respecté (F = {f(ci['feller_respecte']['ratio_feller'], 1)}) : Euler {f(ci['feller_respecte']['euler']['pente'])}, Milstein {f(ci['feller_respecte']['milstein']['pente'])} ; "
            f"Feller violé (F = {f(ci['feller_viole']['ratio_feller'], 2)}, paramètres du projet) : Euler {f(ci['feller_viole']['euler']['pente'])}, Milstein {f(ci['feller_viole']['milstein']['pente'])}.",
            f"Heston, biais faible du prix : Euler full-truncation {f(h['euler_ft']['biais'][0], 3)} à 4 pas -> {f(h['euler_ft']['biais'][3], 3)} à 32 pas -> {f(h['euler_ft']['biais'][-1], 3)} ± {f(h['euler_ft']['se'][-1], 3)} à 128 pas ; "
            f"QE : {f(h['qe']['biais'][0], 3)} à 4 pas, compatible avec 0 (chi2 p = {f(h['qe']['p_chi2_zero'], 2)}), erreur-type {f(h['qe']['se'][0], 3)}.",
        ]
    if cle == "Q2":
        g, hs = c["gbm"], c["heston"]
        hd, ha = hs["D_les_deux (xi=0,3, rho=-0,999)"], hs["A_projet (Feller violé, rho=-0,7)"]
        return [
            f"beta (variance par niveau, V_l ~ 2^(-beta l)) : Euler {f(g['euler']['beta'])} {ic(g['euler']['ic_beta'])}, Milstein {f(g['milstein']['beta'])} {ic(g['milstein']['ic_beta'])} (théorie 1 et 2) ; "
            f"alpha (ordre faible) : Euler {f(g['euler']['alpha'])} {ic(g['euler']['ic_alpha'])}, Milstein {f(g['milstein']['alpha'])} {ic(g['milstein']['ic_alpha'])} ; gamma = 1 (pas fins).",
            f"Giles : Euler beta ~ gamma donne eps^-2 (log eps)^2 ; Milstein beta > gamma donne eps^-2. Coût mesuré en eps^-x : x = {f(g['euler']['cout']['pente_cout'])} {ic(g['euler']['cout']['ic_pente_cout'])} (Euler), "
            f"{f(g['milstein']['cout']['pente_cout'])} {ic(g['milstein']['cout']['ic_pente_cout'])} (Milstein), {f(g['euler']['cout']['pente_cout_standard'])} pour le MC standard.",
            f"Gain de coût de Milstein sur Euler (pas fins) : x{f(c['gain_cout_milstein_sur_euler'][0], 1)} à eps = {g['euler']['cout']['epsilons'][0]} -> x{f(c['gain_cout_milstein_sur_euler'][-1], 1)} à eps = {g['euler']['cout']['epsilons'][-1]} "
            f"(coût x eps^2 : Euler {f(g['euler']['cout']['cout_x_eps2'][-1], 0)}, Milstein {f(g['milstein']['cout']['cout_x_eps2'][-1], 0)}) ; la croissance en log^2 d'Euler est masquée par le niveau 0 sur cette plage.",
            f"Heston : beta passe de {f(ha['euler']['beta'])} (Euler) à {f(ha['milstein']['beta'])} (Milstein) sur le cas du projet, mais atteint {f(hd['milstein']['beta'])} {ic(hd['milstein']['ic_beta'])} seulement si Feller est respecté ET si l'aire de Lévy est négligeable (rho -> -1).",
        ]
    if cle == "Q3":
        bl, d_max = c["blocs"], str(max(int(d) for d in c["blocs"]))
        b16, bm = bl["16"], bl[d_max]

        def s(b, k, kk):
            return 100 * b[k]["indice"][b[k]["ks"].index(kk)]

        return [
            f"Variance expliquée par la 1re coordonnée (d = 16) : {f(s(b16, 'incrementale', 1), 1)} % (incrémentale), {f(s(b16, 'pont', 1), 1)} % (pont), {f(s(b16, 'acp', 1), 1)} % (ACP) ; "
            f"par les 4 premières : {f(s(b16, 'incrementale', 4), 1)} / {f(s(b16, 'pont', 4), 1)} / {f(s(b16, 'acp', 4), 1)} %.",
            f"Dimension effective de troncature à 99 % : {b16['incrementale']['dim_troncature_99']} / {b16['pont']['dim_troncature_99']} / {b16['acp']['dim_troncature_99']} à d = 16, "
            f"{bm['incrementale']['dim_troncature_99']} / {bm['pont']['dim_troncature_99']} / {bm['acp']['dim_troncature_99']} à d = {d_max} : seule la construction incrémentale en dépend (résolution 8 au-delà de k = 16).",
            f"RMSE à N = 2^14 (d = 16) : MC {b16['mc']['rmse'][-1]:.1e}, incrémentale {b16['incrementale']['rmse']['rmse'][-1]:.1e}, pont {b16['pont']['rmse']['rmse'][-1]:.1e}, ACP {b16['acp']['rmse']['rmse'][-1]:.1e} "
            f"(pentes {f(b16['mc']['pente'])} / {f(b16['incrementale']['rmse']['pente'])} / {f(b16['pont']['rmse']['pente'])} / {f(b16['acp']['rmse']['pente'])}).",
            f"Koksma-Hlawka : l'erreur suit la discrépance de la projection qui porte la variance ; gain de l'ACP sur le pont : x{f(b16['pont']['rmse']['rmse'][-1] / b16['acp']['rmse']['rmse'][-1], 1)} à d = 16, "
            f"x{f(bm['pont']['rmse']['rmse'][-1] / bm['acp']['rmse']['rmse'][-1], 1)} à d = {d_max}.",
        ]
    if cle == "Q4":
        st, lh = c["strates"], c["lhs"]
        e1, e2, a = st["europeen_K100"]["64"], st["europeen_K140"]["64"], st["asiatique_16_pont"]["64"]
        return [
            f"Stratification de la normale terminale, H = 64 (n = {c['n']}), européen ATM : x{f(e1['mesure']['facteur_prop'], 0)} mesuré (proportionnelle) contre x{f(e1['theorie']['facteur_prop'], 0)} prédit par sigma^2 / somme(p_h sigma_h^2) ; "
            f"Neyman avec sigma_h connus x{f(e1['mesure']['facteur_neyman_oracle'], 0)} (théorie x{f(e1['theorie']['facteur_neyman_oracle'], 0)}). Précision de chaque facteur mesuré : ~{f(100 * c['precision_relative_facteur'] / 2, 0)} %.",
            f"Call OTM K = 140 (sigma_h très inégaux) : proportionnelle x{f(e2['mesure']['facteur_prop'], 1)}, Neyman x{f(e2['mesure']['facteur_neyman_oracle'], 0)} (oracle) et x{f(e2['mesure']['facteur_neyman_pilote'], 0)} (pilote n/8). "
            f"Asiatique 16 dates (Z_1 du pont stratifiée) : x{f(a['mesure']['facteur_prop'], 1)} / x{f(a['mesure']['facteur_neyman_pilote'], 1)} (pilote) / x{f(a['mesure']['facteur_neyman_oracle'], 1)} (oracle), théorie {f(a['theorie']['facteur_prop'], 1)} / {f(a['theorie']['facteur_neyman_oracle'], 1)} (proportionnelle / oracle).",
            f"LHS (Stein) : somme des indices de Sobol d'ordre 1 = {f(lh['asiatique_16_incrementale']['somme_S1'], 3)} (incrémentale), {f(lh['asiatique_16_pont']['somme_S1'], 3)} (pont) -> gain asymptotique prédit "
            f"x{f(lh['asiatique_16_incrementale']['gain_asymptotique_theorique'], 1)} / x{f(lh['asiatique_16_pont']['gain_asymptotique_theorique'], 1)} ; mesuré à n = 4096 : x{f(lh['asiatique_16_incrementale']['n']['4096']['gain_mesure'], 1)} / x{f(lh['asiatique_16_pont']['n']['4096']['gain_mesure'], 1)}.",
            f"Dimension 1 : le LHS est une stratification à n strates, variance en n^{f(lh['europeen_d1']['pente_var_lhs'], 2)} contre n^{f(lh['europeen_d1']['pente_var_mc'], 2)} pour le MC.",
        ]
    if cle == "Q5":
        k, cr, le = c["cas"]["call_K180"], c["cas"]["credit"], c["efficacite_logarithmique"]
        p, q = k["points"], cr["points"]
        ecart = max(abs(v["variance_empirique"] / v["variance_theorique"] - 1) for cas in (p, q) for lib, v in cas.items() if lib != "naif")
        return [
            f"Call K = 180 : le second moment M(mu) est convexe ; mu* = {f(p['mu_star']['mu'], 3)} (variance divisée par {f(p['mu_star']['facteur_variance'], 0)}), contre mu = z_K = {f(p['strike']['mu'], 3)} (x{f(p['strike']['facteur_variance'], 0)}, choix de v1) "
            f"et mu_CE = {f(p['mu_ce']['mu'], 3)} (x{f(p['mu_ce']['facteur_variance'], 0)}). L'entropie croisée par 8 itérations de 20 000 tirages donne mu = {f(np.mean(k['ce_mu_final']), 3)} ± {f(np.std(k['ce_mu_final'], ddof=1), 3)} (10 graines).",
            f"Défaut rare (au moins {c['k_defauts']} défauts sur 100, P = {cr['valeur_exacte']:.2e}) : mu* = {f(q['mu_star']['mu'], 3)}, mu_CE = {f(q['mu_ce']['mu'], 3)}, variance divisée par {f(q['mu_star']['facteur_variance'], 0)} (CE : {f(q['mu_ce']['facteur_variance'], 0)}).",
            f"La variance empirique (n = {c['n_empirique']}) retrouve M(mu) - I^2 à {f(100 * ecart, 1)} % près sur les dérives d'importance.",
            f"Grandes déviations : ln M(mu*) / (2 ln P) = {f(le[0]['rapport_log'], 3)} (a = 2) -> {f(le[2]['rapport_log'], 3)} (a = 6) -> {f(le[-1]['rapport_log'], 3)} (a = 20) ; l'erreur relative de l'IS ne vaut que {f(le[-1]['erreur_relative_is'], 1)} à a = 20 (MC naïf : {le[-1]['erreur_relative_naive']:.1e}).",
        ]
    if cle == "Q6":
        ex, bs, geo, co = c["exactitude"], c["vs_black_scholes_europeen"], c["vs_geometrique_ferme"], c["cout"]
        pire = max(max(v["max_abs_aad_vs_pathwise"].values()) for v in ex.values())
        d = co[-1]
        return [
            f"Exactitude : sur les mêmes chemins (n = {c['n_verif']}, 12 dates), l'AAD en mode inverse égale le pathwise écrit à la main à {pire:.0e} près en valeur absolue (grecques de l'ordre de 10 à 500) ; boucle et version vectorisée coïncident.",
            f"Contre des références indépendantes : européen vs Black-Scholes z = {f(bs['delta']['z'])} / {f(bs['vega']['z'])} / {f(bs['rho']['z'])} (delta / vega / rho) ; asiatique géométrique vs différences finies de la formule fermée z = {f(geo['delta']['z'])} / {f(geo['vega']['z'])} / {f(geo['rho']['z'])}.",
            f"Coût relatif au prix (asiatique, {d['d']} dates, {d['m_grecques']} grecques dont {d['d']} vegas par date) : AAD {f(d['aad_boucle'], 1)}x (boucle) et {f(d['aad_vectorise'], 1)}x (vectorisé), contre {f(d['bump_seaux'], 0)}x pour le seul bump-and-revalue des vegas ({d['reevaluations_bump_seaux']} réévaluations).",
            f"Le rapport AAD/prix reste dans {f(min(x['aad_boucle'] for x in co), 1)}-{f(max(x['aad_boucle'] for x in co), 1)}x (boucle) quand m passe de {co[0]['m_grecques']} à {d['m_grecques']} (Griewank : 3 à 4 en comptage d'opérations) ; "
            f"accélération sur le bump : x{f(d['acceleration_aad_boucle_vs_bump_seaux'], 0)} (boucle) à x{f(d['acceleration_aad_vectorise_vs_bump_seaux'], 0)} (vectorisé). Temps numpy : dépendent de la machine.",
        ]
    if cle == "Q7":
        k, ch_, es = c["kupiec"], c["christoffersen"], c["es_euler"]
        pk, cc = k["points"], ch_["groupées 1 %, pi11 = 0,20"]
        out = [
            f"Kupiec, modèle à 2 % au lieu de 1 % : puissance {f(100 * pk['250']['puissance_exacte'], 0)} % à 1 an, {f(100 * pk['500']['puissance_exacte'], 0)} % à 2 ans, {f(100 * pk['1250']['puissance_exacte'], 0)} % à 5 ans, "
            f"{f(100 * pk['2500']['puissance_exacte'], 0)} % à 10 ans ; 80 % de puissance durable après {k['T_80']} jours ({f(k['T_80'] / 250, 1)} ans ; formule normale {f(k['T_80_formule_normale'] / 250, 1)} ans). La taille réelle du test vaut {f(100 * pk['250']['taille_reelle'], 1)} % à 1 an (pas 5 %).",
            f"Bâle : un modèle à 2 % est en zone verte {f(100 * k['zones_bale_250j_taux_2pct']['vert'], 0)} % du temps sur 250 jours. Exceptions groupées de taux correct 1 % (pi11 = 0,20) : Christoffersen (indépendance) rejette "
            f"{f(100 * cc['250']['ind'], 0)} / {f(100 * cc['500']['ind'], 0)} / {f(100 * cc['1250']['ind'], 0)} / {f(100 * cc['2500']['ind'], 0)} % à 1/2/5/10 ans, Kupiec seulement {f(100 * cc['2500']['kupiec'], 0)} % à 10 ans.",
        ]
        if "donnees_absentes" not in es:
            e9, em = es["jour_2020_03_09"], es["moyenne"]
            es_txt = f"{e9['es']:,.0f}".replace(",", " ")
            out += [
                f"ES FHS 97,5 % au 9 mars 2020 : {es_txt} EUR ; somme des contributions d'Euler = ES à {e9['ecart_relatif_somme']:.0e} près ; dérivées numériques w_i dES/dw_i = contributions à {e9['ecart_relatif_euler_derivees']:.0e} ; ES(2w)/ES(w) = {e9['rapport_homogeneite']:.6f}.",
                f"Parts de l'ES (poids du capital) : SPX {f(100 * e9['parts']['SPXUSD'], 0)} % ({f(100 * e9['poids']['SPXUSD'], 0)} %), NSX {f(100 * e9['parts']['NSXUSD'], 0)} % ({f(100 * e9['poids']['NSXUSD'], 0)} %), BCO {f(100 * e9['parts']['BCOUSD'], 0)} % ({f(100 * e9['poids']['BCOUSD'], 0)} %), "
                f"XAU {f(100 * e9['parts']['XAUUSD'], 0)} % ({f(100 * e9['poids']['XAUUSD'], 0)} %) ; moyenne sur {em['n_dates']} dates : SPX {f(100 * em['parts']['SPXUSD'], 0)} %, NSX {f(100 * em['parts']['NSXUSD'], 0)} %, BCO {f(100 * em['parts']['BCOUSD'], 0)} %, XAU {f(100 * em['parts']['XAUUSD'], 0)} %.",
            ]
        return out
    if cle == "Q8":
        bl = c["blocs"]
        return [
            "Pente log-log de la part des pas tronqués en dt : " + " ; ".join(f"F = {f(v['F'])} -> {f(v['pente'])}" for v in bl.values()) + " (la pente croît avec F = 2 kappa theta / xi^2 et reste sous F : régime non asymptotique).",
            f"À 16 pas, la troncature touche {f(100 * bl['0.25']['fraction_tronquee'][1], 0)} % des pas pour F = 0,25, {f(100 * bl['0.5']['fraction_tronquee'][1], 0)} % pour F = 0,5 (cas du projet : 0,48), {f(100 * bl['1.0']['fraction_tronquee'][1], 1)} % pour F = 1, "
            f"{f(100 * bl['4.0']['fraction_tronquee'][1], 3)} % pour F = 4.",
            f"Biais relatif de E[v_T^+] à 16 pas : {f(100 * bl['0.5']['biais_moyenne']['16']['biais_relatif'], 1)} % (F = 0,5) contre {f(100 * bl['4.0']['biais_moyenne']['16']['biais_relatif'], 2)} % (F = 4).",
        ]
    raise KeyError(cle)


LECTURE_FIGURE = {
    "Q1": "Quatre panneaux en log-log où la pente est l'ordre de convergence : erreur forte du GBM, erreur faible du GBM (barres = IC95), erreur forte du CIR (traits pleins = Feller respecté, pointillés = violé) et biais du prix Heston (Euler full-truncation contre QE). Une courbe qui s'écarte de la droite de référence contredit le théorème ou sort de son domaine de validité.",
    "Q2": "Quatre panneaux : variance par niveau (pente = -beta), espérance par niveau (pente = -alpha), coût multiplié par eps^2 (une courbe plate est un coût en eps^-2 pur ; Milstein doit être plat et sous Euler) et beta de Heston selon Feller et l'aire de Lévy.",
    "Q3": "À gauche, part de variance expliquée par les k premières coordonnées pour les trois constructions (plus la courbe monte vite, plus la dimension effective est petite) ; à droite, RMSE du QMC randomisé contre N pour d = 16, 32, 64 : ACP sous le pont, lui-même sous l'incrémental, MC en dernier.",
    "Q4": "Facteur de variance à H = 64 strates (barre = mesuré, trait = théorie : une barre sur son trait veut dire que la formule prévoit le gain), gain de variance du LHS sur l'asiatique, puis variance contre n en dimension 1 où le LHS suit n^-2.",
    "Q5": "Variance par tirage en fonction de la dérive mu pour chaque cas : une cuvette dont le fond est mu* (les repères montrent le choix naïf et l'entropie croisée) ; à droite, le rapport logarithmique qui tend vers 1 quand l'événement devient rare.",
    "Q6": "À gauche, coût de toutes les grecques rapporté au coût du prix quand le nombre de grecques m augmente : l'AAD reste plat, le bump-and-revalue croît avec m (les temps dépendent de la machine) ; à droite, écart de l'AAD à des références indépendantes en erreurs-types (dans la bande verte +/-3 : compatible).",
    "Q7": "Puissance du test de Kupiec contre l'horizon de backtest (courbe en dents de scie : loi discrète), taux de rejet de Christoffersen (trait plein) contre Kupiec (pointillé) sur des exceptions groupées, et contributions d'Euler de chaque ligne à l'ES (une barre négative est une ligne qui couvre).",
    "Q8": "À gauche, fraction de pas tronqués contre dt en log-log pour plusieurs F = 2 kappa theta / xi^2 ; à droite, la pente de ces droites contre F : elle croît avec F et reste sous la bissectrice tant que dt n'est pas assez petit.",
}


def section_readme(blocs):
    """Texte markdown de la section « ## Approfondissement » (titres de questions + lignes lues dans le JSON)."""
    sortie = ["## Approfondissement", "",
              "Huit questions précises, chacune avec un théorème énoncé, une expérience qui le vérifie ou le met en défaut et une conclusion chiffrée. "
              "Tout est lu dans `results/approfondissement.json` (régénéré par `scripts/run_approfondissement.py`, graines fixées) ; "
              "le détail pas à pas est dans `notebooks/approfondissement.ipynb`. Les chronométrages de Q6 et les durées dépendent de la machine.", ""]
    for cle, b in blocs.items():
        sortie.append(f"### {cle} {b['question']}")
        sortie.append("")
        sortie.append(f"*Théorème* : {b['theoreme']}")
        sortie.append("")
        sortie += [f"- {ligne}" for ligne in lignes_readme(cle, b)]
        sortie.append("")
        sortie.append(f"![{cle}](figures/{fa.FIGURES[cle][1]})")
        sortie.append("")
        sortie.append(f"**Comment lire ce graphique.** {LECTURE_FIGURE[cle]}")
        sortie.append("")
    return "\n".join(sortie)


def injecter_readme(texte):
    p = RACINE / "README.md"
    md = p.read_text(encoding="utf-8")
    bloc = f"{MARQUEUR_DEBUT}\n{texte}\n{MARQUEUR_FIN}"
    if MARQUEUR_DEBUT in md:
        avant, reste = md.split(MARQUEUR_DEBUT, 1)
        _, apres = reste.split(MARQUEUR_FIN, 1)
        md = avant + bloc + apres
    else:  # première insertion : juste avant « ## Limites honnêtes »
        ancre = "## Limites honnêtes"
        assert ancre in md
        md = md.replace(ancre, bloc + "\n\n" + ancre, 1)
    p.write_text(md, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(experiences.BLOCS))
    ap.add_argument("--sans-readme", action="store_true")
    ap.add_argument("--readme-seul", action="store_true", help="réécrit la section du README depuis le JSON existant, sans calcul")
    args = ap.parse_args()
    if args.readme_seul:
        injecter_readme(section_readme(json.loads(RESULTATS.read_text(encoding="utf-8"))["blocs"]))
        print("README mis à jour (sans calcul)")
        return
    cles = [c.strip().upper() for c in args.only.split(",")]
    existant = json.loads(RESULTATS.read_text(encoding="utf-8")) if RESULTATS.exists() else {"blocs": {}}
    blocs = existant.get("blocs", {})
    t0 = time.time()
    for cle in cles:
        print(f"[{cle}] ...", flush=True)
        b = experiences.BLOCS[cle]()
        fa.sauver(fa.FIGURES[cle][0](b["chiffres"]), FIGURES, fa.FIGURES[cle][1])
        b["figure"] = f"figures/{fa.FIGURES[cle][1]}"
        blocs[cle] = b
        print(f"[{cle}] {b['duree_s']:.0f} s\n  {b['conclusion'][:300]}...", flush=True)
        blocs = {k: blocs[k] for k in experiences.BLOCS if k in blocs}
        meta = {"python": platform.python_version(), "numpy": np.__version__, "cpu_logiques": os.cpu_count(),
                "duree_derniere_execution_s": time.time() - t0}
        RESULTATS.write_text(json.dumps({"meta": meta, "blocs": blocs}, ensure_ascii=False, indent=1), encoding="utf-8")
    if not args.sans_readme and set(blocs) == set(experiences.BLOCS):
        injecter_readme(section_readme(blocs))
        print("README mis à jour")
    print(f"terminé en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
