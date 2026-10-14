# Laboratoire Monte Carlo de pricing et de risque, validé

Un moteur Monte Carlo pour options (GBM, Heston QE, Merton, asiatiques, barrières, put américain, grecques, MLMC) et un volet risque (VaR 99 % et ES 97,5 % par cinq méthodes, toutes backtestées) où **chaque nombre a une barre d'erreur, chaque prix est confronté à une référence indépendante et chaque technique d'amélioration a un gain chiffré**.

<!-- AUTO:tete_glance:DEBUT -->
> **At a glance.** A Monte Carlo option-pricing and market-risk lab (Python, numpy, scipy, numba): 42 validated cases against independent references plus 3 deliberate failure demos, and a 1,000-repetition coverage test of the 95% intervals.  
> Five VaR/ES methods backtested over 3,511 days (Kupiec, Christoffersen, Basel, Acerbi-Székely): only filtered historical simulation (GARCH) passes all four tests, on a linear 7-asset portfolio.  
> No trading strategy and no transaction costs: every number is regenerated from fixed seeds (`python scripts/run_all.py`) and checked by `pytest` and CI.
<!-- AUTO:tete_glance:FIN -->

[![tests](https://github.com/yassine-ek/monte-carlo-pricing-risk-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/yassine-ek/monte-carlo-pricing-risk-lab/actions/workflows/tests.yml) ![python](https://img.shields.io/badge/python-3.12-blue) ![licence](https://img.shields.io/badge/licence-MIT-green)

## Résultats clés

<!-- AUTO:tete_cles:DEBUT -->
- **Pricing : 42 cas validés + 3 contre-exemples volontaires.** Chaque prix Monte Carlo est à moins de 3 erreurs-types d'une référence indépendante (37 cas sans biais toléré, pire |z| = 2,53 ; 5 avec un biais de discrétisation documenté). *Réserve* : un |z| compatible établit la cohérence à l'erreur-type près, pas l'exactitude ; sur Heston QE (z = +1,35, erreur-type 0,0138) un biais de 0,5 % du prix resterait invisible.
- **Barres d'erreur honnêtes.** L'IC95 contient la vraie valeur 94,3 % à 95,6 % du temps sur 1 000 répétitions (cinq estimateurs) ; le contre-exemple de l'événement rare est montré (75,6 %). *Réserve* : fluctuation binomiale de ±1,4 point, et couverture mesurée avec peu de chemins par répétition, pas à la précision des tables de prix.
- **Risque : cinq méthodes de VaR/ES backtestées sur 3 511 jours.** Une seule méthode passe les quatre tests (FHS (GARCH) : 38 exceptions pour 35 attendues), les quatre autres sont rejetées et dites telles. *Réserve* : portefeuille de 7 positions longues linéaires (le Monte Carlo y simule la loi jointe des rendements, il ne réévalue pas d'options), P&L en devise de cotation sans conversion EUR ; avec 35 exceptions attendues, un backtest d'un an ne détecte un modèle à 2 % d'exceptions que 24 % du temps (Q7) et une p-value proche de 0,05 n'est pas décisive.
<!-- AUTO:tete_cles:FIN -->

![Figure héro à deux panneaux : à gauche l'erreur RMSE du prix selon N pour le Monte Carlo et deux variantes de QMC, à droite le nombre d'exceptions de la VaR 99 % par méthode contre la bande compatible avec 1 %](figures/hero_convergence_et_var.png)

**Lecture** ([voir en grand](figures/hero_convergence_et_var.png)). À gauche, plus la courbe descend vite quand N augmente, plus la méthode est précise : le Monte Carlo suit la pente −1/2, le QMC avec pont brownien s'en détache nettement. À droite, chaque barre est le nombre de fois où la perte a dépassé la VaR 99 % ; la bande verte est ce qu'un bon modèle à 1 % produirait par hasard, et une barre marquée * est rejetée par le test de Kupiec.

**Aller plus loin** : [visite guidée (notebook)](notebooks/visite_guidee.ipynb) · [approfondissement Q1 à Q8](docs/approfondissement.md) · [tables de validation complètes](docs/validation.md) · [méthodologie du volet risque](docs/risque.md)

## Frais et hypothèses

- **Frais et coûts** : aucun. Ce dépôt ne contient **aucune stratégie de trading** (ni courbe d'equity) : le labo valorise des produits et mesure des risques, il n'exécute pas d'ordres. La VaR et l'ES portent sur la variation de valeur, brute de frais, d'un portefeuille réévalué en totalité chaque jour.
- **Cas de référence (pricing)** : call européen S0 = K = 100, T = 1 an, r = 2 %, σ = 20 % ; Heston v0 = θ = 0,04, κ = 1,5, ξ = 0,5, ρ = −0,7 (viole la condition de Feller : 2κθ = 0,12 < ξ² = 0,25) ; Merton λ = 0,5, μ_J = −0,1, δ_J = 0,15.
- **Règle de verdict** : un cas est **OK** si |estimation − référence| < 3 erreurs-types (plus, quand c'est écrit, un biais de discrétisation documenté). Références : formules fermées ou semi-fermées, arbre CRR à 2 000 pas, Monte Carlo haute précision. Graines fixes (`np.random.default_rng`) : deux exécutions donnent les mêmes nombres, hors chronométrages.
- **Données du volet risque** : 7 lignes longues (EURUSD 20 %, USDJPY 15 %, GBPUSD 15 %, XAUUSD 15 %, BCOUSD 10 %, SPXUSD 15 %, NSXUSD 10 %), notionnel 10 M€, clôture quotidienne tirée de barres 1 h ; 4 011 rendements du 2010-11-16 au 2026-06-26, P&L en devise de cotation sans conversion en EUR (le notionnel est un habillage). **Données de marché non redistribuées** : voir « Obtenir les données » en bas de page ; sans elles, les résultats du dépôt (`results/*.json`) restent consultables.
- **TRAIN / VAL / TEST et délai d'exécution** : sans objet, car rien n'est optimisé et aucun ordre n'est passé. Backtest glissant et causal du 2012-10-25 au 2026-06-26 (3 511 jours) : la prévision du jour t n'utilise que les rendements jusqu'à t−1, ce qu'un test de non-fuite vérifie.

## Métriques

**Pricing** : chaque ligne compare une estimation Monte Carlo à une référence indépendante (z = (estimation − référence) / erreur-type).

<!-- AUTO:tete_validation:DEBUT -->
| Cas | Méthode | Référence | Estimation | Erreur-type | z | Verdict |
|---|---|---|---|---|---|---|
| Call européen BS | MC naïf | 8.9160 | 8.8668 | 0.0194 | -2.53 | OK |
| Call européen BS | QMC Sobol randomisé (16 x 2^14) | 8.9160 | 8.9159 | 0.0001 | -1.98 | OK |
| Call Heston | QE d'Andersen (20 pas) | 8.1950 | 8.2135 | 0.0138 | +1.35 | OK |
| Call Heston | QE numba parallèle (50 pas) | 8.1950 | 8.1876 | 0.0048 | -1.53 | OK |
| Call Merton | schéma exact (Poisson) | 10.1272 | 10.1423 | 0.0221 | +0.68 | OK |
| Asiatique géométrique (12 dates) | QMC pont (16 x 2^12) | 5.1909 | 5.1907 | 0.0008 | -0.18 | OK |
| Asiatique arithmétique (12 dates) | VC asiatique géométrique | 5.3782 | 5.3783 | 0.0005 | +0.18 | OK |
| Call très OTM (K=180) | importance sampling (mu=2.94) | 0.0177 | 0.0177 | 0.0001 | +0.35 | OK |
| Down-and-out call (H=90, 50 dates) | pont brownien | 7.3004 | 7.3131 | 0.0190 | +0.67 | OK |
| Down-and-out call (H=90, 50 dates) | MC naïf (grille) | 7.3004 | 7.7682 | 0.0194 | +24.14 | échec attendu (démo) |
| Put américain LSM (50 dates) | Longstaff-Schwartz, Laguerre deg. 3, évaluation indépendante | 5.7849 | 5.7703 | 0.0096 | -1.53 | OK |
| MLMC GBM (Euler) | MLMC de Giles, eps=0.01 | 8.9160 | 8.9169 | 0.0071 | +0.13 | OK |
| Grecque delta (call) | pathwise | 0.5793 | 0.5792 | 0.0008 | -0.08 | OK |

***42 cas validés + 3 contre-exemples volontaires** (sur 45 comparaisons exécutées ; MC naïf sur barrière discrète, pathwise sur digitale). Parmi les 42 cas validés, 37 n'ont aucun biais de discrétisation toléré (pire |z| = 2,53 : Call européen BS, MC naïf) et 5 ont un biais documenté, dont la borne haute duale du put américain, jugée par un encadrement et non par une égalité. Un |z| compatible établit la cohérence à l'erreur-type près, pas l'exactitude. Toutes les lignes : [docs/validation.md](docs/validation.md). Frais : aucun (le labo ne trade pas).*
<!-- AUTO:tete_validation:FIN -->

**Risque** : cinq méthodes de VaR/ES, jugées comme le ferait une fonction Contrôle des risques.

<!-- AUTO:tete_risque:DEBUT -->
Backtest de la VaR 99 % à 1 jour sur 3 511 jours (2012-10-25 → 2026-06-26), portefeuille de 10 M€, **35,1 exceptions attendues**. p-values : rejet à 5 % si p < 0,05 (en gras).

| Méthode | Exceptions | Kupiec (p) | Christoffersen indép. (p) | Christoffersen cc (p) | ES 97,5 % Z2 (p) | Verdict |
|---|---|---|---|---|---|---|
| Simulation historique | 47 (attendu 35) | 0.055 | **0.000** | **0.000** | **0.004** | rejetée : indépendance, cc, ES |
| Normale EWMA | 73 (attendu 35) | **0.000** | **0.021** | **0.000** | **0.000** | rejetée : Kupiec, indépendance, cc, ES |
| Monte Carlo Student-t | 59 (attendu 35) | **0.000** | **0.019** | **0.000** | **0.000** | rejetée : Kupiec, indépendance, cc, ES |
| FHS (GARCH) | 38 (attendu 35) | 0.629 | 0.070 | 0.172 | 0.378 | passe les 4 tests |
| Copule t + GARCH | 52 (attendu 35) | **0.008** | **0.048** | **0.004** | **0.011** | rejetée : Kupiec, indépendance, cc, ES |
<!-- AUTO:tete_risque:FIN -->

**Pourquoi un Monte Carlo sur un portefeuille linéaire ?** Les 7 positions sont longues et linéaires : la réévaluation complète `L = −N Σ wᵢ (exp(rᵢ) − 1)` est une fonction simple des rendements, donc le Monte Carlo ne sert pas ici à repricer des options. Il sert à simuler la **loi jointe, non gaussienne,** des 7 rendements (queues épaisses, asymétrie de baisse, dépendance de queue : Student multivarié, bootstrap conjoint de la FHS, copule t + GARCH), dont le quantile de la perte n'a pas de formule fermée ; la VaR paramétrique normale n'en est que le cas gaussien, et c'est l'une des plus rejetées. Que la FHS l'emporte est un résultat attendu pour ce type de portefeuille : l'apport est la méthode de backtest (cinq modèles jugés par quatre tests), pas une découverte. Ce que ce volet ne montre pas : une réévaluation complète où delta-gamma et réévaluation divergent, c'est-à-dire un portefeuille d'options (extension non faite).

## Overfit ou pas ?

<!-- AUTO:tete_lecture:DEBUT -->
- **Rien n'est réglé sur le backtest du risque** : fenêtre 500 j, lambda = 0,94, refit GARCH 20 j, 10 000 tirages sont fixés avant les calculs. Il n'y a pas de découpage TRAIN / VAL / TEST parce que rien n'est optimisé : le backtest est glissant et causal (la prévision du jour t n'utilise que les rendements jusqu'à t-1, ce qu'un test de non-fuite vérifie).
- **Aucune sélection a posteriori** : les 5 méthodes sont publiées, rejetées comprises. 5 méthodes x 4 tests = 20 p-values sans correction pour tests multiples, dont 3 tombent entre 0,03 et 0,10 : elles ne sont pas décisives. Un backtest d'un an ne détecte un modèle à 2 % d'exceptions que 24 % du temps (Q7).
- **Seuil de pricing fixé avant** : un cas est validé si |z| < 3. Sous H0, 42 comparaisons produisent environ 0,1 dépassement par hasard : un écart isolé de 2 à 2,5 sigma n'est pas un signal, et les résultats sont déterministes à graine donnée.
<!-- AUTO:tete_lecture:FIN -->

## Toutes les variantes

Rien n'est caché : toutes les méthodes testées sont listées, y compris celles qui échouent.

### Pricing et réduction de variance

![Facteurs de variance et d'efficacité par technique de réduction de variance, en échelle logarithmique](figures/efficacite_vr.png)

**Lecture.** Chaque barre est le nombre de fois où une technique divise la variance (échelle logarithmique) par rapport au Monte Carlo naïf. L'antithétique ne gagne qu'un facteur 2 environ ; la variable de contrôle géométrique sur l'asiatique et l'importance sampling sur l'option très hors de la monnaie gagnent plusieurs centaines à un millier : tout dépend de la corrélation avec une quantité connue ou de la rareté de l'événement.

<details>
<summary>Tableaux : toutes les méthodes de pricing par thème, gain de chaque technique de réduction de variance</summary>

**Pricing et grecques, résumé par thème** (toutes les lignes : 42 cas validés et 3 contre-exemples volontaires, dans [docs/validation.md](docs/validation.md)).

<!-- AUTO:variantes_pricing:DEBUT -->
| Thème | Méthodes comparées | Référence | Pire z | Verdict |
|---|---|---|---|---|
| Call européen BS | MC naïf ; MC antithétique ; VC sur S_T ; QMC Sobol randomisé (16 x 2^14) | formule fermée | -2.53 | OK |
| Call Heston | Euler full-truncation (100 pas) ; QE d'Andersen (20 pas) ; QE numba parallèle (50 pas) | semi-fermé (Fourier) | +1.87 | OK |
| Call Merton | schéma exact (Poisson) | série de Merton | +0.68 | OK |
| Asiatique géométrique (12 dates) | MC naïf ; QMC pont (16 x 2^12) | formule fermée | +0.97 | OK |
| Asiatique arithmétique (12 dates) | MC naïf ; MC antithétique ; VC asiatique géométrique ; QMC pont (16 x 2^12) | MC haute précision (VC, 2 M chemins, SE=1.5e-04) | -0.80 | OK |
| Call très OTM (K=180) | MC naïf ; importance sampling (mu=2.94) | formule fermée | -1.51 | OK |
| Down-and-out call (H=90, 50 dates) | MC naïf (grille) ; décalage BGK ; pont brownien | formule continue (Reiner-Rubinstein) | +24.14 | contient un contre-exemple volontaire : MC naïf (grille) |
| Put américain LSM (50 dates) | Longstaff-Schwartz, Laguerre deg. 3, évaluation indépendante | arbre CRR bermudéen (N=2000, 50 dates) | -1.53 | OK |
| Put américain LSM, cas Longstaff-Schwartz (S0=36, K=40) | Longstaff-Schwartz, 50 dates | arbre CRR bermudéen (N=2000, 50 dates) | -1.13 | OK |
| Put américain LSM, cas Longstaff-Schwartz (S0=40, K=40) | Longstaff-Schwartz, 50 dates | arbre CRR bermudéen (N=2000, 50 dates) | -1.50 | OK |
| Put américain LSM, cas Longstaff-Schwartz (S0=44, K=40) | Longstaff-Schwartz, 50 dates | arbre CRR bermudéen (N=2000, 50 dates) | -1.60 | OK |
| Put américain LSM (10 dates) | Longstaff-Schwartz | arbre CRR bermudéen (N=2000, 10 dates) | -1.16 | OK |
| Put américain, borne duale (10 dates) | Rogers / Haugh-Kogan (valeur LSM deg. 7, quadrature) | arbre CRR bermudéen (N=2000, 10 dates) | +11.34 | OK, grâce à un biais toléré et documenté (borne haute de la dualité) |
| Grecque delta (call) | pathwise ; lr ; bump | formule fermée BS | -1.00 | OK |
| Grecque gamma (call) | lr ; bump | formule fermée BS | -1.35 | OK |
| Grecque vega (call) | pathwise ; lr ; bump | formule fermée BS | -0.93 | OK |
| Grecque delta (digitale) | pathwise ; lr ; bump | formule fermée BS | -0.69 | contient un contre-exemple volontaire : pathwise |
| Grecque vega (digitale) | pathwise ; lr ; bump | formule fermée BS | +1.44 | contient un contre-exemple volontaire : pathwise |
| MLMC GBM (Euler) | MLMC de Giles, eps=0.01 | formule fermée BS | +0.13 | OK |
| MLMC Heston (Euler) | MLMC de Giles, eps=0.05 | semi-fermé (Fourier) | +1.95 | OK |
| Parité call-put sur prix MC | GBM exact ; Heston QE ; Heston Euler ; Merton exact | S0 e^{-qT} - K e^{-rT} | +1.12 | OK |
<!-- AUTO:variantes_pricing:FIN -->

**Réduction de variance, QMC et importance sampling : gain de chaque technique.**

<!-- AUTO:variantes_vr:DEBUT -->
| Technique | Cas | Facteur de variance | Facteur d'efficacité (variance × temps) |
|---|---|---|---|
| antithétique | Européen | ×1.7 | ×1.8 |
| VC sur S_T | Européen | ×5.5 | ×5.0 |
| antithétique | Asiatique arith. | ×1.8 | ×1.8 |
| VC géométrique | Asiatique arith. | ×1383.3 | ×936.4 |
| importance sampling | Call OTM K=180 | ×584.3 | ×714.4 |
| Stratification proportionnelle (H = 64) | Européen ATM, brownien terminal | ×92 | non chronométré |
| Stratification de Neyman, σ_h connus (H = 64) | Européen ATM | ×1110 | non chronométré |
| Stratification de Neyman, pilote n/8 (H = 64) | Call OTM K = 140 | ×180 | non chronométré |
| Hypercube latin (n = 4096) | Asiatique 16 dates, pont | ×10 | non chronométré |
| Importance sampling, dérive optimale μ* | Call OTM K = 180 | ×882 | non chronométré |
| Importance sampling, entropie croisée | Call OTM K = 180 | ×873 | non chronométré |
| QMC Sobol + pont brownien (N = 2^14, rapport de RMSE²) | Asiatique 16 dates | ×1901 | non chronométré |
| QMC Sobol + ACP (N = 2^14, rapport de RMSE²) | Asiatique 16 dates | ×12007 | non chronométré |

*Les cinq premières lignes viennent de `results/validation.json` (efficacité = variance × temps mesuré). Les huit suivantes viennent de `results/approfondissement.json` (Q3, Q4, Q5) : leur coût par tirage est voisin de celui du MC naïf, donc l'efficacité vaut à peu près le facteur de variance, mais elle n'est pas chronométrée.*
<!-- AUTO:variantes_vr:FIN -->

</details>

### Risque

![Exceptions de VaR cumulées par méthode, avec l'attendu à 1 % et la bande binomiale à 95 %](figures/risque_exceptions_cumulees.png)

**Lecture.** Chaque courbe compte les exceptions cumulées d'une méthode ; la droite en pointillé est ce qu'on attend à 1 % et la bande grise l'intervalle binomial à 95 % d'un modèle exact. Une courbe qui sort de la bande par le haut sous-estime le risque ; un saut brutal marque une crise (mars 2020). Le détail méthode par méthode est dans [docs/risque.md](docs/risque.md).

<details>
<summary>Tableau : les cinq méthodes de VaR/ES (principe, VaR et ES moyennes, zone verte de Bâle, mars 2020)</summary>

<!-- AUTO:variantes_var:DEBUT -->
| Méthode | Principe | VaR 99 % moy. | ES 97,5 % moy. | Exceptions | Zone verte Bâle | COVID 9 mars 2020 : perte / VaR |
|---|---|---|---|---|---|---|
| Simulation historique | quantile des 500 dernières pertes | 138 281 € | 140 469 € | 47 / 3511 | 69 % | ×3.56 |
| Normale EWMA | gaussienne, covariance EWMA (lambda = 0,94) | 108 302 € | 108 836 € | 73 / 3511 | 44 % | ×2.50 |
| Monte Carlo Student-t | Student multivarié, covariance EWMA | 114 771 € | 117 565 € | 59 / 3511 | 59 % | ×2.37 |
| FHS (GARCH) | GARCH(1,1) + bootstrap de résidus standardisés | 130 556 € | 134 577 € | 38 / 3511 | 80 % | ×1.91 |
| Copule t + GARCH | marges GARCH + copule t | 117 039 € | 120 067 € | 52 / 3511 | 65 % | ×2.10 |

*Pertes sur un portefeuille de 10 M€, réévaluation complète. « Zone verte » : part des fenêtres glissantes de 250 jours avec au plus 4 exceptions. La dernière colonne vaut plus de 1 quand la perte réalisée dépasse la VaR du jour.*
<!-- AUTO:variantes_var:FIN -->

</details>

## Pourquoi ce projet

Un desk de trading ou une équipe de pricing ne se fie à un prix ou à une VaR que si on sait **à quel point on peut s'y fier** : erreur-type, référence indépendante, gain mesuré de chaque technique, test du modèle de risque. C'est le travail d'un quant de support au trading : valider un pricer, accélérer une simulation, justifier un chiffre devant le risque. Métiers visés : market making, exécution, plateformes de trading FX.

## Méthode en bref

- **Estimateur et erreur-type.** Un prix est une espérance, $C = e^{-rT}\,\mathbb{E}[(S_T-K)^+]$ ; on la remplace par la moyenne de $n$ payoffs simulés, avec $\mathrm{SE} = s/\sqrt n$ et $\mathrm{IC}_{95} = \hat C \pm 1{,}96\,\mathrm{SE}$. Le score $z = (\hat C - C_{\text{réf}})/\mathrm{SE}$ dit si l'écart à la référence est du bruit (|z| < 3) ou un biais.
- **Réduire la variance.** Antithétique (tirer $Z$ et $-Z$), variable de contrôle (soustraire une quantité corrélée d'espérance connue), importance sampling (tirer là où le payoff vit, corriger par le poids de Girsanov), stratification, hypercube latin ; le gain est le rapport des variances, et le facteur d'**efficacité** le multiplie par le rapport des temps de calcul.
- **Quasi-Monte Carlo.** Suite de Sobol brouillée plus pont brownien : l'erreur décroît plus vite que $N^{-1/2}$ parce que la variance est portée par les premières coordonnées (voir Q3).
- **Discrétiser.** Euler, Milstein, QE d'Andersen pour Heston ; MLMC de Giles : $\mathbb{E}[P_L] = \mathbb{E}[P_0] + \sum_l \mathbb{E}[P_l - P_{l-1}]$, avec le même brownien sur deux grilles consécutives.
- **Risque.** VaR 99 % = quantile de la perte à 1 jour, ES 97,5 % = moyenne de la perte au-delà ; cinq méthodes, chacune jugée par Kupiec (nombre d'exceptions), Christoffersen (leur indépendance), les feux de Bâle et le test de l'ES d'Acerbi-Székely, avec une prévision du jour $t$ qui n'utilise que les rendements jusqu'à $t-1$.

<!-- AUTO:approfondissement:DEBUT -->
## Approfondissement

Huit théorèmes (convergence d'Euler et de Milstein, MLMC, dimension effective du QMC, stratification, importance sampling, AAD, puissance des backtests, Feller) confrontés à la mesure, avec les cas où la théorie et les données divergent. Détail, formules et figures : [docs/approfondissement.md](docs/approfondissement.md) et [notebook](notebooks/approfondissement.ipynb).

- **Q1 Euler et Milstein** : ordre fort mesuré en L1 0,49 (Euler) contre 0,94 (Milstein), ordre faible 1 pour les deux ; Feller violé dégrade le CIR (Euler 0,40, Milstein 0,68).
- **Q2 MLMC et Milstein** : variance par niveau en 2^(-1,95 l) avec Milstein contre 2^(-0,95 l) avec Euler ; coût en eps^-1,94 contre eps^-2,20 ; Milstein x3,6 moins cher à eps = 0,002.
- **Q3 Pont brownien et ACP** : la 1re coordonnée porte 12 % / 73 % / 98 % de la variance (incréments / pont / ACP) ; RMSE du QMC divisée par 2,5 entre pont et ACP (d = 16).
- **Q4 Stratification et hypercube latin** : variance divisée par 92 mesuré contre 89 prédit (H = 64, européen ATM) ; Neyman x1110 ; LHS x9,7 sur l'asiatique (pont).
- **Q5 Importance sampling** : dérive optimale mu* = 3,39 (variance divisée par 882) ; l'entropie croisée la retrouve (mu = 3,49, x873).
- **Q6 Grecques par AAD** : 67 grecques pour 3,3x le coût du prix contre 153x en bump-and-revalue ; exactes à 6e-13 près contre le pathwise.
- **Q7 Puissance des backtests** : un modèle à 2 % d'exceptions n'est détecté que 24 % du temps après 1 an (80 % après 1130 jours) ; les contributions d'Euler à l'ES somment à l'ES.
- **Q8 Condition de Feller** : à 16 pas, 14 % des pas sont tronqués pour F = 0,5 (cas du projet) contre 0,013 % pour F = 4.
<!-- AUTO:approfondissement:FIN -->

## Limites honnêtes

- **Volet risque : un portefeuille linéaire, un seul jeu de données.** 7 positions longues (deux indices corrélés à 0,93), P&L en devise de cotation sans conversion EUR, poids constants sans coûts ni portage. La réévaluation complète est ici l'identité d'une fonction simple : le Monte Carlo y apporte la loi jointe, pas le repricing d'options. Avec 35 exceptions attendues, la puissance des tests est limitée (24 % pour un modèle à 2 % d'exceptions sur un an) et aucune correction pour tests multiples n'est appliquée.
- **z compatible ne veut pas dire exact.** Un |z| < 3 établit la cohérence à l'erreur-type près : sur Heston QE (erreur-type 0,0138), un biais de 0,5 % du prix passerait inaperçu. Le z de la borne duale (+11,3) est un test d'inégalité (la borne doit dépasser la valeur), pas un écart à une égalité.
- **MLMC sur Heston : aucun gain mesuré** dans la plage testée (rapport MLMC / standard de 6,8 à eps = 0,2 jusqu'à 1,7 à
  eps = 0,02). Les exposants (≈2,3 contre ≈2,8) montrent l'avantage asymptotique, mais la variance par niveau décroît
  lentement avec Euler full-truncation (~2^(-0,7 l)) ; le croisement, extrapolé vers eps ~ 7e-3, n'a pas été mesuré
  (trop coûteux). Sur GBM, croisement vers eps ~ 0,05. Le coût du MC standard est une **estimation à la Giles** (variance
  du niveau fin x nombre de pas imposé par le biais), pas une exécution.
- **Borne haute duale** : valide mais pas serrée à 50 dates. Une variante par **simulation emboîtée (Andersen-Broadie)** a été
  prototypée puis abandonnée : le bruit des sous-simulations donnait des bornes très lâches (de l'ordre de +8 % à 10 dates et
  +40 % à 50 dates, mesure de développement non rejouée par `run_all.py`). On utilise à la place une espérance conditionnelle
  par **quadrature** (le pas GBM est lognormal) avec une valeur approchée de degré 7 : sans bruit, mais dépendante de la
  qualité de cette approximation.
- **LSM** : estimateur biaisé bas par construction. Écart relatif au CRR **américain** (N=2000) dans `validation.json` :
  -0,49 % (put ATM, 50 dates), -0,35 % / -0,59 % / **-1,07 %** pour S0 = 36 / 40 / 44 (cas de Longstaff-Schwartz 2001,
  K=40) : le put le plus hors de la monnaie **dépasse 1 %** (dont ~0,3 % viennent de la seule discrétisation bermudéenne).
  Les estimations portent sur un put **bermudéen** à 10 ou 50 dates, comparé à un arbre bermudéen sur les mêmes dates ;
  le CRR américain N=2000 n'est pas exact non plus (erreur ~ 1e-3, oscillante en N).
- **Références non fermées** : l'asiatique arithmétique est comparée à un **MC haute précision** (8 x 250 000 chemins avec
  variable de contrôle), pas à une formule ; son z combine les deux erreurs-types.
- **Schéma QE** : version standard (K0 sans correction de martingale) ; le biais résiduel est invisible à nos erreurs-types
  (voir la parité call-put) mais existe en principe.
- **QMC randomisé** : la convergence est étudiée sur l'asiatique géométrique (référence exacte). Sur un payoff 1-D avec peu
  de brouillages (K=8) l'IC de Student sous-couvre (~92 % en développement) ; la couverture rapportée est celle de la
  configuration utilisée (K=16, 16 dimensions). Le z du QMC randomisé du tableau de tête est jugé avec le seuil normal 3, alors que 16 brouillages relèveraient d'un Student à 15 degrés de liberté, et sa référence est imprimée à 4 décimales pour une erreur-type de 1e-4.
- **numba** : générateur SplitMix64 par chemin (pas PCG64) pour que le résultat ne dépende pas du nombre de threads ;
  l'équivalence avec la version numpy est **statistique** (même prix à l'erreur-type près), pas bit à bit. Le noyau ne
  renvoie que S_T. Les vitesses dépendent de la machine (nombre de coeurs logiques indiqué dans `validation.json`).
- **Verdicts à 3 sigma** : 42 comparaisons validées (hors 3 contre-exemples volontaires) à graines fixées ; sous H0 une comparaison sur 370 dépasserait 3 sigma par
  hasard, donc un écart isolé de 2 à 2,5 sigma n'est pas un signal. Les résultats sont déterministes à graine donnée.
- **Approfondissement : limites propres.** (i) L'AAD est codée à la main pour un GBM discret à volatilité par date ; ce n'est pas un outil de différentiation automatique générique. (ii) Les chronométrages (Q6, `benchmark_numba`) dépendent de la machine et de sa charge : une première exécution sur machine saturée avait donné un coût AAD inférieur au prix, ce qui est impossible ; on mesure donc le minimum de plusieurs répétitions contre un prix de référence minimal, et on ne retient que les rapports. (iii) Q7 (contributions d'Euler à l'ES réel) exige les données de marché (`UNIVERS_DIR`) ; sans elles ce bloc est ignoré. (iv) Les asymptotiques de Q1, Q2, Q8 sont mesurées sur des pas finis : une pente sous la valeur théorique peut venir du régime non asymptotique (Q8) plutôt que d'un défaut du schéma.
- **Pas fait** : GPU, calibration au marché, vol locale ou rugueuse, mini-xVA, Milstein avec aire de Lévy (le Milstein de Q2 sur Heston omet ce terme, d'où β < 2 hors du cas favorable), portefeuille d'options dans le backtest de VaR.

<details>
<summary>Structure du dépôt : modules de <code>mclab/</code> et arborescence</summary>

| Module | Contenu | Validé contre |
|---|---|---|
| `utils.py` | `ResultatMC`, résumé i.i.d. / antithétique, paramètres Heston et Merton, payoffs, vol implicite (brentq) | aller-retour prix -> vol -> prix |
| `analytique.py` | Black-Scholes + grecques (avec dividende q), **Heston** semi-fermé (fonction caractéristique « little trap », `quad`), **Merton** (série de Black), asiatique **géométrique** discrète, **down-and-out** continu, digitale, put américain **CRR** (N=2000) | limites connues (Heston -> BS quand la vol de vol tend vers 0, Merton sans saut, parités, différences finies) |
| `modeles.py` | GBM exact ; Heston **Euler full-truncation** et **QE d'Andersen (2008)** ; Merton exact (Poisson + sauts lognormaux) ; tirages fournis (QMC) et antithétique | Heston semi-fermé, série de Merton, BS |
| `reduction_variance.py` | antithétique ; **variable de contrôle** (S_T pour l'européenne, asiatique géométrique pour l'arithmétique, beta sur un **pilote indépendant**) ; **importance sampling** par translation de drift (call très OTM) ; facteurs de variance et d'efficacité | BS, MC haute précision |
| `qmc.py` | **Sobol brouillé** + normale inverse + **pont brownien** ; QMC **randomisé** (K brouillages -> erreur-type, IC de Student) ; étude de convergence RMSE vs N | asiatique géométrique (formule exacte) |
| `barriere.py` | down-and-out discret : **naïf** vs **pont brownien** vs décalage BGK | formule continue de Reiner-Rubinstein |
| `americaine.py` | **Longstaff-Schwartz** (Laguerre deg. 3, règle apprise sur un jeu, prix évalué sur un **autre jeu indépendant**), frontière d'exercice, **borne haute duale** | arbre CRR (mêmes dates d'exercice) |
| `grecques.py` | **pathwise**, **rapport de vraisemblance**, **bump-and-revalue avec nombres aléatoires communs** ; cas **digital** | Black-Scholes, digitale fermée |
| `mlmc.py` | **MLMC de Giles** (Euler ou Milstein, raffinement x2, niveaux adaptatifs), call GBM puis Heston, exposants alpha/beta/gamma mesurés, coût vs eps | BS, Heston semi-fermé |
| `accel.py` | simulation **Heston QE en numba** (`@njit(parallel=True)`), benchmark numpy vs numba | Heston semi-fermé |
| `validation.py` | **couverture des IC95** par répétitions, **parité call-put** sur prix MC | (contrôle des estimateurs eux-mêmes) |
| `risque.py` | **VaR 99 % / ES 97,5 %** par 5 méthodes (historique, EWMA normale, MC Student-t, FHS-GARCH, copule t), tests de Kupiec, Christoffersen, feux de Bâle, Acerbi-Székely | backtest causal sur 3 511 jours (voir [docs/risque.md](docs/risque.md)) |
| `schemas.py` | **Euler, Milstein** (GBM, CIR), Heston full-truncation et QE, erreurs forte et faible, régression des ordres | solution exacte du GBM, Heston semi-fermé |
| `anova.py` | **dimension effective** : indices de Sobol fermés (pick-freeze), indices d'ordre 1 (Hermite), constructions incrémentale / pont / ACP | exponentielle linéaire (indice exact connu) |
| `stratification.py` | **stratification** proportionnelle et de Neyman, **hypercube latin** | variances théoriques par quadrature |
| `importance.py` | **importance sampling** gaussien : second moment M(mu), dérive optimale, entropie croisée, grandes déviations | quadrature adaptative |
| `aad.py` | **différentiation adjointe (AAD)** codée à la main pour un GBM discret, contre pathwise, bump-and-revalue | Black-Scholes, asiatique géométrique fermée |
| `puissance.py` | **puissance de Kupiec / Christoffersen** (énumération binomiale exacte et simulation), zones de Bâle | formules exactes |
| `contributions_es.py` | **contributions d'Euler à l'ES** (somme = ES), vérifiées par dérivées numériques | théorème d'Euler (homogénéité de degré 1) |
| `experiences.py` | les huit expériences Q1 à Q8 de l'approfondissement (une fonction par question, blocs JSON) | (appelle les modules ci-dessus) |

```
.
  mclab/            20 modules : pricing (utils, analytique, modeles, reduction_variance, qmc, barriere, americaine, grecques,
                    mlmc, accel, validation), risque (risque), approfondissement (schemas, anova, stratification, importance,
                    aad, puissance, contributions_es, experiences) ; README.md : une ligne par module
  tests/            un fichier par module + tests d'approfondissement et d'hygiène (pytest, graines fixées, tolérances en erreurs-types)
  scripts/          run_all.py, run_risque.py, run_approfondissement.py, figures*.py, build_*_01.py (notebook, tête du README)
  notebooks/        visite_guidee.ipynb, approfondissement.ipynb (intuition -> formule -> attente -> expérience -> lecture)
  results/          validation.json, risque_backtest.json, approfondissement.json, tableau_validation.md
  figures/          hero, 9 figures de pricing, 5 du risque, 8 de l'approfondissement ; README.md : une ligne par image
  docs/             risque.md, validation.md, approfondissement.md, positionnement.md
  .github/workflows/tests.yml   CI : pytest sur Python 3.12
```

Positionnement face à ~70 dépôts publics de Monte Carlo en finance : [docs/positionnement.md](docs/positionnement.md).

</details>

## Reproduire

```bash
python -m venv .venv                             # PowerShell : .venv\Scripts\Activate.ps1 ; bash : source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                              # tests (environ 1 min ; les tests du volet risque réel sont ignorés sans données)
python scripts/run_all.py                        # validation.json + figures + tableaux de docs/validation.md (~2,5 min)
python scripts/run_approfondissement.py          # approfondissement.json + figures appro_*.png + docs/approfondissement.md + résumé du README (~7 min)
python scripts/build_readme_tete_01.py           # figure héro + blocs de tête du README, lus dans results/*.json
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/visite_guidee.ipynb          # notebook de visite (~1 min)
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/approfondissement.ipynb      # notebook d'approfondissement (~10 min)
```

Les scripts réécrivent les fichiers versionnés de `results/` et `figures/` : durées, versions de numpy et bruit à la 15e décimale changent d'une machine à l'autre (`git diff` pour le voir). L'intégration continue (`.github/workflows/tests.yml`, Ubuntu, Python 3.12) lance `pip install -r requirements.txt` puis `pytest -q` ; sans données de marché, un test (contributions d'Euler à l'ES sur les vrais rendements) est ignoré.

### Obtenir les données (volet risque seulement)

Le pricing, l'approfondissement et la quasi-totalité des tests n'ont besoin d'aucune donnée. Le backtest de VaR/ES lit des barres 1 h de 7 actifs, **non redistribuées** : télécharger les barres 1 minute (EURUSD, USDJPY, GBPUSD, XAUUSD, BCOUSD, SPXUSD, NSXUSD) sur <https://www.histdata.com>, puis les rééchantillonner en 1 h (dernier close de l'heure, sans interpolation, heure EST de HistData). Arborescence attendue, dans un dossier désigné par la variable `UNIVERS_DIR` :

```
univers/
  EURUSD_1h.csv   colonnes : datetime,close   (une ligne par heure)
  USDJPY_1h.csv   GBPUSD_1h.csv   XAUUSD_1h.csv   BCOUSD_1h.csv   SPXUSD_1h.csv   NSXUSD_1h.csv
```

Puis `export UNIVERS_DIR=/chemin/vers/univers` (PowerShell : `$env:UNIVERS_DIR = "C:\chemin\vers\univers"`) et `python scripts/run_risque.py` (~2,5 min), qui écrit `results/risque_backtest.json` et les figures `risque_*.png`. Sans cette variable, le script s'arrête avec un message d'une ligne et tout le reste fonctionne.
