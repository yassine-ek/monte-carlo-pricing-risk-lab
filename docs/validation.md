# Résultats de validation détaillés

Tables complètes de `results/validation.json` (**42 cas validés + 3 contre-exemples volontaires**), couverture des IC95, gains de réduction de variance, encadrement du put américain, MLMC, numba, galerie de figures et lectures. Les blocs entre marqueurs `AUTO` sont réécrits par `scripts/run_all.py` : ne pas les éditer à la main. Retour au [README](../README.md).

## Règle du verdict

**OK si |estimation - référence| < 3 erreurs-types** (plus, quand c'est écrit, un biais de discrétisation documenté : 5 cas, dont la
borne haute duale du put américain, jugée par un encadrement et non par une égalité). Les cas ci-dessous (**42 validés + 3 contre-exemples
volontaires**) sont **lus dans `results/validation.json`** (régénéré par `scripts/run_all.py`, graines fixées) ; les tableaux sont réinjectés
automatiquement dans ce fichier. Lignes marquées « échec attendu » : contre-exemples volontaires (MC naïf sur barrière discrète, pathwise sur une digitale).
Un |z| < 3 établit la cohérence à l'erreur-type près, pas l'exactitude : avec une erreur-type de 0,0138 sur Heston QE, un biais de 0,5 % du prix y reste invisible.

## Prix

<!-- AUTO:prix:DEBUT -->
| Cas | Méthode | Estimation ± SE | IC95 | Référence | z | Verdict |
|---|---|---|---|---|---|---|
| Call européen BS | MC naïf | 8.8668 ± 0.0194 | [8.8287 ; 8.9049] | 8.9160 | -2.53 | OK |
| Call européen BS | MC antithétique | 8.8968 ± 0.0149 | [8.8676 ; 8.9260] | 8.9160 | -1.29 | OK |
| Call européen BS | VC sur S_T | 8.9155 ± 0.0083 | [8.8992 ; 8.9317] | 8.9160 | -0.07 | OK |
| Call européen BS | QMC Sobol randomisé (16 x 2^14) | 8.9159 ± 0.0001 | [8.9157 ; 8.9161] | 8.9160 | -1.98 | OK |
| Call Heston | Euler full-truncation (100 pas) | 8.2357 ± 0.0218 | [8.1931 ; 8.2784] | 8.1950 | +1.87 | OK |
| Call Heston | QE d'Andersen (20 pas) | 8.2135 ± 0.0138 | [8.1866 ; 8.2405] | 8.1950 | +1.35 | OK |
| Call Heston | QE numba parallèle (50 pas) | 8.1876 ± 0.0048 | [8.1781 ; 8.1971] | 8.1950 | -1.53 | OK |
| Call Merton | schéma exact (Poisson) | 10.1423 ± 0.0221 | [10.0989 ; 10.1856] | 10.1272 | +0.68 | OK |
| Asiatique géométrique (12 dates) | MC naïf | 5.2016 ± 0.0111 | [5.1798 ; 5.2234] | 5.1909 | +0.97 | OK |
| Asiatique géométrique (12 dates) | QMC pont (16 x 2^12) | 5.1907 ± 0.0008 | [5.1891 ; 5.1924] | 5.1909 | -0.18 | OK |
| Asiatique arithmétique (12 dates) | MC naïf | 5.3659 ± 0.0181 | [5.3304 ; 5.4014] | 5.3782 | -0.68 | OK |
| Asiatique arithmétique (12 dates) | MC antithétique | 5.3674 ± 0.0136 | [5.3408 ; 5.3940] | 5.3782 | -0.80 | OK |
| Asiatique arithmétique (12 dates) | VC asiatique géométrique | 5.3783 ± 0.0005 | [5.3774 ; 5.3793] | 5.3782 | +0.18 | OK |
| Asiatique arithmétique (12 dates) | QMC pont (16 x 2^12) | 5.3776 ± 0.0009 | [5.3755 ; 5.3796] | 5.3782 | -0.69 | OK |
| Call très OTM (K=180) | MC naïf | 0.0158 ± 0.0012 | [0.0134 ; 0.0182] | 0.0177 | -1.51 | OK |
| Call très OTM (K=180) | importance sampling (mu=2.94) | 0.0177 ± 0.0001 | [0.0176 ; 0.0178] | 0.0177 | +0.35 | OK |
| Down-and-out call (H=90, 50 dates) | MC naïf (grille) | 7.7682 ± 0.0194 | [7.7302 ; 7.8062] | 7.3004 | +24.14 | échec attendu (démo) |
| Down-and-out call (H=90, 50 dates) | décalage BGK | 7.3134 ± 0.0192 | [7.2758 ; 7.3510] | 7.3004 | +0.68 | OK |
| Down-and-out call (H=90, 50 dates) | pont brownien | 7.3131 ± 0.0190 | [7.2758 ; 7.3504] | 7.3004 | +0.67 | OK |
| Put américain LSM (50 dates) | Longstaff-Schwartz, Laguerre deg. 3, évaluation indépendante | 5.7703 ± 0.0096 | [5.7515 ; 5.7891] | 5.7849 | -1.53 | OK |
| Put américain LSM, cas Longstaff-Schwartz (S0=36, K=40) | Longstaff-Schwartz, 50 dates | 4.4712 ± 0.0061 | [4.4593 ; 4.4831] | 4.4780 | -1.13 | OK |
| Put américain LSM, cas Longstaff-Schwartz (S0=40, K=40) | Longstaff-Schwartz, 50 dates | 2.3058 ± 0.0054 | [2.2951 ; 2.3165] | 2.3140 | -1.50 | OK |
| Put américain LSM, cas Longstaff-Schwartz (S0=44, K=40) | Longstaff-Schwartz, 50 dates | 1.1012 ± 0.0055 | [1.0904 ; 1.1121] | 1.1101 | -1.60 | OK |
| Put américain LSM (10 dates) | Longstaff-Schwartz | 5.7205 ± 0.0097 | [5.7015 ; 5.7396] | 5.7318 | -1.16 | OK |
| Put américain, borne duale (10 dates) | Rogers / Haugh-Kogan (valeur LSM deg. 7, quadrature) | 5.7650 ± 0.0029 | [5.7593 ; 5.7707] | 5.7318 | +11.34 | borne valide |
| MLMC GBM (Euler) | MLMC de Giles, eps=0.01 | 8.9169 ± 0.0071 | [8.9031 ; 8.9308] | 8.9160 | +0.13 | OK |
| MLMC Heston (Euler) | MLMC de Giles, eps=0.05 | 8.2639 ± 0.0354 | [8.1946 ; 8.3333] | 8.1950 | +1.95 | OK |
<!-- AUTO:prix:FIN -->

## Grecques (call et digitale, S0=K=100, T=1, r=2 %, sigma=20 %)

<!-- AUTO:grecques:DEBUT -->
| Cas | Méthode | Estimation ± SE | IC95 | Référence | z | Verdict |
|---|---|---|---|---|---|---|
| Grecque delta (call) | pathwise | 0.5792 ± 0.0008 | [0.5776 ; 0.5808] | 0.5793 | -0.08 | OK |
| Grecque delta (call) | lr | 0.5773 ± 0.0020 | [0.5734 ; 0.5811] | 0.5793 | -1.00 | OK |
| Grecque delta (call) | bump | 0.5792 ± 0.0008 | [0.5776 ; 0.5808] | 0.5793 | -0.11 | OK |
| Grecque gamma (call) | lr | 0.0194 ± 0.0002 | [0.0190 ; 0.0197] | 0.0196 | -0.93 | OK |
| Grecque gamma (call) | bump | 0.0193 ± 0.0002 | [0.0190 ; 0.0196] | 0.0196 | -1.35 | OK |
| Grecque vega (call) | pathwise | 39.0102 ± 0.1053 | [38.8039 ; 39.2165] | 39.1043 | -0.89 | OK |
| Grecque vega (call) | lr | 38.7572 ± 0.3734 | [38.0252 ; 39.4891] | 39.1043 | -0.93 | OK |
| Grecque vega (call) | bump | 39.0091 ± 0.1053 | [38.8028 ; 39.2154] | 39.1043 | -0.90 | OK |
| Grecque delta (digitale) | pathwise | 0.0000 ± 0.0000 | [0.0000 ; 0.0000] | 0.0196 | +inf | échec attendu (démo) |
| Grecque delta (digitale) | lr | 0.0195 ± 0.0000 | [0.0194 ; 0.0196] | 0.0196 | -0.69 | OK |
| Grecque delta (digitale) | bump | 0.0195 ± 0.0001 | [0.0192 ; 0.0198] | 0.0196 | -0.48 | OK |
| Grecque vega (digitale) | pathwise | 0.0000 ± 0.0000 | [0.0000 ; 0.0000] | -0.3910 | +inf | échec attendu (démo) |
| Grecque vega (digitale) | lr | -0.3986 ± 0.0064 | [-0.4111 ; -0.3861] | -0.3910 | -1.18 | OK |
| Grecque vega (digitale) | bump | -0.3823 ± 0.0061 | [-0.3942 ; -0.3703] | -0.3910 | +1.44 | OK |
<!-- AUTO:grecques:FIN -->

## Parité call-put sur les prix MC

<!-- AUTO:parite:DEBUT -->
| Cas | Méthode | Estimation ± SE | IC95 | Référence | z | Verdict |
|---|---|---|---|---|---|---|
| Parité call-put sur prix MC | GBM exact | 2.0122 ± 0.0286 | [1.9561 ; 2.0682] | 1.9801 | +1.12 | OK |
| Parité call-put sur prix MC | Heston QE | 1.9737 ± 0.0259 | [1.9230 ; 2.0245] | 1.9801 | -0.25 | OK |
| Parité call-put sur prix MC | Heston Euler | 1.9671 ± 0.0409 | [1.8870 ; 2.0472] | 1.9801 | -0.32 | OK |
| Parité call-put sur prix MC | Merton exact | 2.0152 ± 0.0330 | [1.9504 ; 2.0800] | 1.9801 | +1.06 | OK |
<!-- AUTO:parite:FIN -->

## Les IC95 sont-ils honnêtes ? Couverture sur répétitions indépendantes

Chaque estimateur est répété 1 000 fois avec des graines indépendantes (nombre de chemins petit, 2 000) ; on compte la
fraction d'IC95 contenant la vraie valeur. Attendu : 95 % à la fluctuation binomiale près.

<!-- AUTO:couverture:DEBUT -->
| Estimateur | Couverture observée | Attendu | Écart (en sigmas binomiaux) |
|---|---|---|---|
| MC naïf | 94.3 % (943/1000) | 95 % ± 1.4 pt | -1.0 |
| antithétique | 95.2 % (952/1000) | 95 % ± 1.4 pt | +0.3 |
| variable de contrôle | 95.6 % (956/1000) | 95 % ± 1.4 pt | +0.9 |
| importance sampling (call K=180) | 95.1 % (951/1000) | 95 % ± 1.4 pt | +0.1 |
| QMC randomisé (asiatique géo., 16 x 2^6) | 95.2 % (952/1000) | 95 % ± 1.4 pt | +0.3 |
| MC naïf sur call K=180 (événement rare, contre-exemple) | 75.6 % (756/1000) | 95 % ± 1.4 pt | -28.1 |
<!-- AUTO:couverture:FIN -->

## Réduction de variance : gains mesurés

<!-- AUTO:efficacite:DEBUT -->
| Technique | Facteur de variance | Facteur d'efficacité (variance x temps) |
|---|---|---|
| Européen : antithétique | 1.7x | 1.8x |
| Européen : VC sur S_T | 5.5x | 5.0x |
| Asiatique arith. : antithétique | 1.8x | 1.8x |
| Asiatique arith. : VC géométrique | 1383.3x | 936.4x |
| Call OTM K=180 : importance sampling | 584.3x | 714.4x |
<!-- AUTO:efficacite:FIN -->

Le **facteur de variance** est déterministe (graines fixées) ; le **facteur d'efficacité** inclut les temps de calcul
mesurés sur la machine et fluctue d'une exécution à l'autre.

## Convergence MC vs QMC

<!-- AUTO:convergence:DEBUT -->
pente MC = -0.49 ; QMC incréments = -0.69 ; QMC pont brownien = -0.85
<!-- AUTO:convergence:FIN -->

## Put américain : encadrement par borne basse (LSM) et borne haute (dualité)

<!-- AUTO:encadrement:DEBUT -->
| Dates d'exercice | LSM (borne basse) | Sur l'apprentissage | CRR bermudéen | Borne haute duale | Écart borne haute |
|---|---|---|---|---|---|
| 10 | 5.7205 ± 0.0097 | 5.7213 | 5.7318 | 5.7650 ± 0.0029 | +0.6 % |
| 50 | 5.7703 ± 0.0096 | 5.7968 | 5.7849 | 5.9591 ± 0.0081 | +3.0 % |
<!-- AUTO:encadrement:FIN -->

## Barrière discrète : erreur (prix MC - formule continue)

<!-- AUTO:barriere:DEBUT -->
| Dates m | MC naïf | Décalage BGK | Pont brownien |
|---|---|---|---|
| 5 | +1.0271 ± 0.0307 | +0.0985 ± 0.0303 | -0.0443 ± 0.0292 |
| 10 | +0.8427 ± 0.0307 | -0.0218 ± 0.0303 | -0.0170 ± 0.0297 |
| 25 | +0.6221 ± 0.0307 | +0.0061 ± 0.0303 | +0.0112 ± 0.0299 |
| 50 | +0.4414 ± 0.0306 | -0.0178 ± 0.0303 | -0.0166 ± 0.0300 |
| 100 | +0.3311 ± 0.0305 | -0.0121 ± 0.0303 | -0.0114 ± 0.0301 |
| 250 | +0.1930 ± 0.0302 | -0.0251 ± 0.0301 | -0.0241 ± 0.0300 |
<!-- AUTO:barriere:FIN -->

## MLMC : coût vs précision

<!-- AUTO:mlmc:DEBUT -->
| Modèle | eps | niveaux | coût MLMC | coût MC standard (estimé) | rapport MLMC / standard (< 1 : MLMC moins cher) |
|---|---|---|---|---|---|
| GBM (Euler) | 0.1 | 3 | 1.12e+05 | 3.34e+04 | 3.35 |
| GBM (Euler) | 0.05 | 3 | 2.82e+05 | 2.93e+05 | 0.96 |
| GBM (Euler) | 0.02 | 4 | 2.15e+06 | 3.85e+06 | 0.56 |
| GBM (Euler) | 0.01 | 4 | 8.57e+06 | 3.05e+07 | 0.28 |
| GBM (Euler) | 0.005 | 4 | 3.42e+07 | 2.39e+08 | 0.14 |
| Heston (Euler) | 0.2 | 5 | 5.07e+05 | 7.46e+04 | 6.80 |
| Heston (Euler) | 0.1 | 6 | 2.95e+06 | 5.98e+05 | 4.94 |
| Heston (Euler) | 0.05 | 6 | 1.18e+07 | 4.85e+06 | 2.44 |
| Heston (Euler) | 0.03 | 7 | 4.66e+07 | 1.34e+07 | 3.49 |
| Heston (Euler) | 0.02 | 7 | 1.05e+08 | 6.03e+07 | 1.74 |

GBM (Euler) : exposant du coût en 1/eps = 1.96 (MLMC) contre 2.95 (MC standard), variance par niveau V_l ~ 2^(-0.88 l) ; Heston (Euler) : exposant du coût en 1/eps = 2.31 (MLMC) contre 2.84 (MC standard), variance par niveau V_l ~ 2^(-0.74 l).
<!-- AUTO:mlmc:FIN -->

## Numba

<!-- AUTO:benchmark:DEBUT -->
numba parallèle : 2 106 920 chemins/s  soit x23 face à numpy (S_T seul) et x25 face à numpy (trajectoires complètes)
<!-- AUTO:benchmark:FIN -->

## Figures (`figures/`, dpi 130)

Chaque image est produite par un script (liste complète dans [figures/README.md](../figures/README.md)). Les figures de l'approfondissement (`appro_q1` à `appro_q8`) sont dans [approfondissement.md](approfondissement.md), avec leur question.

![Erreur RMSE du prix en fonction de N : le Monte Carlo suit la pente -1/2, le QMC avec pont brownien s'en détache nettement](../figures/convergence_mc_vs_qmc.png)

**Comment lire ce graphique.** Erreur RMSE sur le prix en fonction du nombre de chemins N, en échelle log-log : une droite de pente −1/2 est le Monte Carlo ordinaire, une droite plus raide est un gain réel. L'ordre « incréments » du QMC est moins bon que le pont brownien, qui concentre la variance dans les premières coordonnées (voir Q3).

![Smile de volatilité implicite de Heston : les prix Monte Carlo QE collent à la courbe semi-fermée, Euler reste décalé](../figures/smile_heston.png)

**Comment lire ce graphique.** Volatilité implicite en fonction du strike : la courbe noire est le prix semi-fermé de Heston (la vérité), les points avec barres sont les prix Monte Carlo de QE et d'Euler convertis en vol implicite. Plus les points collent à la courbe, meilleur est le schéma ; Euler à 50 pas reste décalé de quelques points de base.

![Frontière d'exercice du put américain apprise par Longstaff-Schwartz, comparée à celle de l'arbre CRR](../figures/frontiere_exercice_lsm.png)

**Comment lire ce graphique.** Frontière d'exercice du put américain : en dessous de la courbe, il vaut mieux exercer. La frontière apprise par Longstaff-Schwartz doit épouser celle de l'arbre CRR ; l'encadrement borne basse / CRR / borne haute montre à quel point le prix est serré.

![Rapport estimation sur valeur exacte pour trois méthodes de grecques : le pathwise tombe à zéro sur la digitale](../figures/grecques_mc_vs_exact.png)

**Comment lire ce graphique.** Rapport estimation / valeur exacte pour trois méthodes de grecques (pathwise, rapport de vraisemblance, bump avec nombres aléatoires communs) : 1 est la bonne réponse. **Le pathwise tombe à 0 sur la digitale** (confiant et faux), le rapport de vraisemblance et le bump s'en sortent.

![MLMC : variance et espérance par niveau, et coût fois epsilon carré contre epsilon pour GBM et Heston](../figures/mlmc.png)

**Comment lire ce graphique.** À gauche, variance et espérance de la correction par niveau : leur décroissance en 2^(−β l) et 2^(−α l) fait le gain du MLMC. À droite, le coût multiplié par ε² en fonction de ε : une courbe qui descend sous celle du Monte Carlo standard est un gain réel (ici sur GBM, pas sur Heston dans la plage testée).

![Erreur du prix de la barrière discrète selon le nombre de dates : le MC naïf reste biaisé, BGK et le pont brownien non](../figures/biais_barriere.png)

**Comment lire ce graphique.** Erreur du prix de la barrière discrète par rapport à la formule continue selon le nombre de dates de surveillance : le MC naïf reste biaisé de plusieurs dizaines d'erreurs-types, tandis que le décalage BGK et le pont brownien retombent sur la référence même avec peu de dates.

![Chemins simulés par seconde : numpy en trajectoires complètes, numpy pour S_T seul, numba parallèle](../figures/benchmark_numba.png)

**Comment lire ce graphique.** Chemins simulés par seconde : numpy en trajectoires complètes, numpy pour S_T seul, numba parallèle. Les vitesses dépendent de la machine ; seul le rapport (environ ×23) est à retenir.

![Couverture observée des IC95 sur 1 000 répétitions : cinq estimateurs dans la bande, le contre-exemple de l'événement rare très en dessous](../figures/couverture_ic95.png)

**Comment lire ce graphique.** Part des intervalles de confiance à 95 % qui contiennent la vraie valeur sur 1 000 répétitions ; la bande est la fluctuation binomiale attendue autour de 95 %. Les cinq estimateurs sont dans la bande ; le contre-exemple (MC naïf sur un événement rare) est très en dessous, preuve que l'IC normal peut mentir quand il y a trop peu de tirages utiles.

**Volet risque.**

![P&L quotidien du portefeuille et -VaR 99 % par méthode, avec les exceptions marquées](../figures/risque_pnl_vs_var.png)

**Comment lire ce graphique.** Un panneau par méthode : P&L quotidien du portefeuille de 10 M€ (gris), −VaR 99 % (couleur) et exceptions (points noirs, là où la perte dépasse la VaR). La VaR doit remonter quand le marché s'agite (mars 2020) et redescendre ensuite ; un amas de points noirs sous la courbe est un modèle trop optimiste.

![Zoom sur mars 2020 : P&L et -VaR 99 % des cinq méthodes autour de la perte du 9 mars](../figures/risque_zoom_covid.png)

**Comment lire ce graphique.** Zoom sur mars 2020 : barres grises = P&L, courbes = −VaR 99 % des cinq méthodes, marqueurs = exceptions (un symbole par méthode). La VaR n'est utile que si elle s'ajuste *avant* la perte du 9 mars, pas après.

![Tableau récapitulatif des tests de backtest par méthode : vert si non rejeté à 5 %, rouge si rejeté](../figures/risque_tableau_recap.png)

**Comment lire ce graphique.** Une ligne par méthode : exceptions observées (attendu 1 %), p-values de Kupiec, de Christoffersen, de l'ES, feux de Bâle. Case verte = test non rejeté à 5 %, rouge = rejeté. Les autres figures du volet risque (`risque_zoom_2022.png`) sont décrites dans [docs/risque.md](risque.md).

## Ce qu'on retient (lectures)

- **Les erreurs-types sont calibrées** : 94 à 96 % de couverture pour le MC naïf, l'antithétique, la variable de contrôle
  (beta pilote), l'importance sampling et le QMC randomisé. **Contre-exemple instructif** : le MC naïf sur un call très
  hors de la monnaie (événement rare, 2 000 chemins) ne couvre que ~76 % des cas : l'IC normal est trop étroit ;
  l'importance sampling corrige à la fois la précision et la couverture.
- **Variable de contrôle sur l'asiatique arithmétique** : facteur de variance supérieur à 1 000 avec la géométrique
  (corrélation ~ 1), sans biais. L'antithétique ne gagne que ~1,7x sur un call : c'est le résultat théorique attendu.
- **QMC** : pente mesurée proche de -0,5 pour MC, -0,69 pour Sobol en ordre naïf, **-0,85** avec le pont brownien. On
  n'atteint pas -1 sur ce payoff (coude en max(., 0) et 16 dimensions) ; on rapporte la valeur mesurée, pas la valeur de manuel.
- **Heston** : les paramètres de référence **violent la condition de Feller** (2 kappa theta = 0,12 < xi^2 = 0,25). Euler
  full-truncation a un biais visible à 10 pas (plusieurs erreurs-types), qui se résorbe vers 100 pas ; le schéma QE est
  compatible avec le semi-fermé dès 10 à 20 pas.
- **Barrière** : le MC naïf est biaisé haut de plusieurs dizaines d'erreurs-types, biais qui décroît lentement avec le nombre
  de dates ; le pont brownien (et le décalage BGK) retombent sur la formule continue même à 10 dates. Ce biais est
  *invisible à l'erreur-type* : seule la comparaison à la référence l'expose.
- **LSM** : estimateur hors échantillon biaisé bas, encadré par une borne haute duale (écart de la borne haute au CRR
  bermudéen dans le tableau ci-dessus : faible à 10 dates, plus lâche à 50 dates).
- **Grecques** : le pathwise est le plus précis sur le call, mais **inapplicable à la digitale** (il renvoie 0 avec une
  erreur-type nulle : confiant et faux) ; LR et bump (nombres aléatoires communs) fonctionnent.
- **MLMC** : variance par niveau en ~2^(-0,9 l) sur GBM (2^(-0,7 l) sur Heston), exposant du coût en 1/eps de 1,96 (GBM) et
  2,31 (Heston) contre ~2,9 pour le MC standard. Sur GBM le MLMC devient moins cher que le MC standard vers eps = 0,05
  (rapport 0,96) et l'est 7 fois plus à eps = 0,005 ; sur Heston il reste plus cher dans la plage testée (voir la limite ci-dessous).
