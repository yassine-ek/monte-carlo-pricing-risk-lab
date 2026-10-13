# Risque de marché Monte Carlo : VaR 99 % et ES 97,5 % à 1 jour

Code : `mclab/risque.py` · Exécution : `python scripts/run_risque.py` (≈ 2,5 min) ·
Tests : `python -m pytest -q tests/test_risque.py` · Résultats bruts : `results/risque_backtest.json`

## 1. Objectif

Mesurer le risque de marché à 1 jour d'un portefeuille multi-actifs réel de 10 M€ avec **cinq
méthodes** de VaR / ES, puis **juger chaque méthode par backtest** comme le ferait une fonction
Risques / Contrôle : couverture (Kupiec), indépendance (Christoffersen), feux tricolores de Bâle,
backtest de l'ES (Acerbi–Székely), comportement en stress (mars 2020, 2022).

Le point de vue est celui d'un contrôleur : on ne cherche pas la « meilleure » VaR (aucun
paramètre n'est réglé sur le backtest : fenêtre 500 j, λ = 0,94, refit GARCH 20 j, N = 10 000 sont
fixés *avant* les calculs) mais à savoir **laquelle est défendable** devant un régulateur.

Niveaux : VaR à 99 % (backtest réglementaire) et ES à 97,5 % (mesure de capital FRTB, qui
remplace la VaR 99 %). Convention : on raisonne sur la **perte** `L = −P&L`, positive quand on perd.

**Pourquoi un Monte Carlo sur un portefeuille linéaire ?** Les 7 positions sont longues et linéaires
(delta-1) : la réévaluation complète `L = −N Σ wᵢ (exp(rᵢ) − 1)` est une fonction simple des rendements,
donc le Monte Carlo ne sert pas ici à repricer des options. Il sert à simuler la **loi jointe, non
gaussienne,** des 7 rendements (queues épaisses, asymétrie de baisse, dépendance de queue) : Student
multivarié, bootstrap conjoint de lignes entières (FHS), copule t + marges GARCH. Le quantile de la
perte sous ces lois n'a pas de formule fermée ; la VaR paramétrique normale (méthode b) n'en est que le
cas gaussien, et c'est l'une des plus rejetées. Que la FHS l'emporte est un résultat attendu pour ce type
de portefeuille : l'apport du volet est la procédure de backtest (cinq modèles, quatre tests, puissance
discutée en limites), pas une découverte de modèle. Ce que le volet ne montre pas : une réévaluation
complète où delta-gamma et réévaluation divergent, c'est-à-dire un portefeuille d'options (extension non
faite).

## 2. Portefeuille et données

Poids déclarés **avant tout calcul**, notionnel 10 M€, 7 lignes **toutes longues**, rebalancées
chaque jour aux poids constants :

| Ligne | EURUSD | USDJPY | GBPUSD | XAUUSD | BCOUSD (Brent) | SPXUSD | NSXUSD |
|---|---|---|---|---|---|---|---|
| Poids | 20 % | 15 % | 15 % | 15 % | 10 % | 15 % | 10 % |

Actifs morts exclus (fonction `cloture_quotidienne` lève une erreur) : WTIUSD, ETXEUR, XAUAUD,
XAUCHF, XAUEUR, XAUGBP, UDXUSD, GRXEUR. Indices choisis : SPXUSD et NSXUSD, cotés quasi 24 h comme
le FX, dont la dernière barre du jour est à 16 h New York dans 92 à 98 % des jours, comme pour les
autres lignes. Les indices à couverture horaire partielle (FRXEUR, JPXJPY, UKXGBP…) sont écartés
pour éviter des clôtures asynchrones.

Source : `<ACTIF>_1h.csv` (barres 1 h) dans le dossier désigné par la variable d'environnement `UNIVERS_DIR`, par défaut
`data/processed/univers` ; ces données ne sont pas redistribuées avec le dépôt (sans elles, `scripts/run_risque.py` s'arrête avec
un message clair et les tests qui en dépendent sont ignorés). **Clôture quotidienne** = dernière barre 1h du jour de trading. Les horodatages sont à
l'heure de New York, le jour de trading va de 17 h à 17 h : on décale de +7 h avant de prendre la
date (la réouverture du dimanche tombe sur le lundi) ; les jours à moins de 12 barres (stubs de
dimanche, jours fériés) sont écartés et le rendement les enjambe. Aucun remplissage vers l'avant.
On garde les dates communes aux 7 lignes.

Données : 4 011 rendements log quotidiens, 2010-11-16 → 2026-06-26. Le Brent et les indices
démarrent en novembre 2010 (donc pas de 2010 complet) et la fenêtre de 500 jours décale la première
prévision : **backtest du 2012-10-25 au 2026-06-26, 3 511 jours**, soit **35 exceptions attendues**
à 99 %. Pertes valorisées en réévaluation complète : `L = −N · Σ wᵢ (exp(rᵢ) − 1)`.

## 3. Méthodes

Toutes utilisent une fenêtre glissante de 500 jours (`R[t−500 : t]`), à moyenne nulle à 1 jour
(convention RiskMetrics) sauf la simulation historique.

**(a) Simulation historique.** *Intuition* : le futur ressemble aux 500 derniers jours.
VaR = quantile 99 % des 500 pertes passées du portefeuille ; ES = moyenne de la queue ≥ 97,5 %.
Aucune hypothèse de loi, mais mémoire lente (un krach reste 500 jours dans la fenêtre).

**(b) Paramétrique normale, EWMA RiskMetrics.** *Intuition* : pertes gaussiennes, volatilité qui
s'adapte vite. `Σ_t = Σ_s λ^(âge) rₛrₛ' / Σ λ^(âge)` avec λ = 0,94 ; `σ_p = √(w'Σw)` ;
`VaR = N · z₀,₉₉ · σ_p` (z = 2,3263) ; `ES = N · σ_p · φ(z₀,₉₇₅) / 0,025` (= 2,3378 σ_p N).
Formules fermées, vérifiées dans les tests.

**(c) Monte Carlo Student-t multivarié.** *Intuition* : même covariance EWMA que (b) mais des
queues épaisses et une dépendance de queue (un même facteur de chocs commun aux actifs).
Tirage `r = (L·g) · √((ν−2)/χ²_ν)` avec `g ~ N(0, I)`, `L L' = Σ_EWMA` (covariance exactement Σ) ;
ν estimé par maximum de vraisemblance sur les rendements du portefeuille standardisés par sa
volatilité EWMA passée (Student de variance 1). 10 000 tirages par jour.

**(d) Filtered Historical Simulation (FHS).** *Intuition* : on garde la forme empirique des
chocs mais on les remet à l'échelle de la volatilité du jour. GARCH(1,1) par actif
(`σ²_t = ω + α r²_{t−1} + β σ²_{t−1}`, innovations Student, package `arch`), réestimé tous les
20 jours sur le seul passé ; résidus standardisés `z = r/σ` ; **bootstrap de lignes entières**
de `z` (les 7 actifs d'un même jour sont tirés ensemble : la dépendance et l'asymétrie sont
conservées) ; `r_sim = σ_t · z*`. 10 000 tirages par jour.

**(e) Monte Carlo copule de Student + marges GARCH.** *Intuition* : on sépare les marges
(GARCH(1,1)-t par actif, loi de ddl νᵢ) de la dépendance (copule t de matrice R et ddl ν_c).
`x = g/√(χ²_{ν_c}/ν_c)` avec `g ~ N(0, R)` ; `uᵢ = F_{t,ν_c}(xᵢ)` ; `rᵢ = σᵢ,t · F⁻¹_{t,νᵢ}(uᵢ)`
(Student de variance 1). R : rho de Spearman des résidus (`R = 2 sin(π ρ/6)`), recalculé chaque
jour ; ν_c : maximum de vraisemblance de la copule sur une grille {3…100}, recalculé tous les 20 jours.
La transformation `F⁻¹_{t,νᵢ}(F_{t,ν_c}(x))` est interpolée sur une grille (erreur relative
< 2·10⁻⁵ en test) car l'inversion exacte coûtait 0,2 s par jour.

## 4. Protocole causal

* La prévision du jour `t` n'utilise **que** `R[t−500 : t]` (rendements ≤ t−1). Le rendement du
  jour t n'est utilisé que pour juger la prévision.
* GARCH : paramètres estimés sur le passé à la date de refit, puis **figés 20 jours** ; la variance
  du jour t est obtenue par le filtre sur la fenêtre passée (amorce = moyenne des 50 premiers r² de
  la fenêtre). En cas d'échec d'ajustement, on reprend les paramètres précédents (0 échec dans
  ce run ; 110 des 1 232 ajustements (8,9 %) tombent sur la frontière α+β = 1 à 10⁻⁶ près, dont 82 avec α+β ≥ 1 exactement : IGARCH, accepté ; compte refait indépendamment).
* Monte Carlo : graine propre à chaque jour (`default_rng([20260101, t])`), partagée par les
  méthodes (c), (d), (e) → reproductible et comparaison à bruit commun. Les 5 méthodes sont évaluées
  **tous les jours** (pas de sous-échantillonnage : 140 s au total).
* **Test de non-fuite** (`tests/test_risque.py::test_aucune_fuite_temporelle`) : on remplace les
  rendements à partir de t* (y compris un krach de −30 % en t*) et on vérifie que les 5 méthodes
  donnent des prévisions **strictement identiques** jusqu'en t* ; un contrôle positif vérifie que la
  prévision de t*+1 change bien.

## 5. Backtests

* **Kupiec (POF)** : `LR = −2 ln[(1−p)^{T−x} p^x] + 2 ln[(1−x/T)^{T−x}(x/T)^x] ~ χ²(1)`, p = 1 %.
  Teste le *nombre* d'exceptions.
* **Christoffersen** : indépendance des exceptions (chaîne de Markov d'ordre 1), `LR_ind ~ χ²(1)`,
  couverture conditionnelle `LR_cc = LR_pof + LR_ind ~ χ²(2)`. Teste leur *regroupement*.
* **Feux de Bâle** : exceptions sur 250 jours glissants, vert 0–4, orange 5–9, rouge ≥ 10.
* **ES (Acerbi–Székely Z2)** : `Z2 = Σ_t L_t·1{L_t > VaR₉₇,₅,t} / (T · 0,025 · ES_t) − 1`, `E[Z2] = 0`
  si l'ES est correct, `Z2 > 0` s'il est sous-estimé. P-value par simulation (5 000 tirages) sous H0 :
  chaque jour, la perte est tirée dans la queue prédictive du modèle (200 quantiles stockés).
  Statistique complémentaire : perte moyenne / ES moyen prévu, les jours où `L > VaR 97,5 %`.

Figures : `figures/risque_pnl_vs_var.png` (P&L vs bandes VaR, exceptions marquées),
`risque_exceptions_cumulees.png`, `risque_zoom_covid.png`, `risque_zoom_2022.png`,
`risque_tableau_recap.png`.

## 6. Résultats (lus dans `results/risque_backtest.json`)

### VaR 99 % : couverture et indépendance

| Méthode | VaR 99 % moy. (k€ ; % notionnel) | Exceptions (taux ; attendu 1 %) | Kupiec p | Christoffersen indép. p | Cond. cov. p |
|---|---|---|---|---|---|
| Simulation historique | 138 ; 1,38 % | 47 (1,34 %) | 0,0552 | 0,0000 | 0,0000 |
| Normale EWMA | 108 ; 1,08 % | 73 (2,08 %) | 0,0000 | 0,0213 | 0,0000 |
| MC Student-t | 115 ; 1,15 % | 59 (1,68 %) | 0,0002 | 0,0195 | 0,0001 |
| **FHS (GARCH)** | 131 ; 1,31 % | **38 (1,08 %)** | **0,6286** | 0,0699 | **0,1721** |
| Copule t + GARCH | 117 ; 1,17 % | 52 (1,48 %) | 0,0075 | 0,0484 | 0,0040 |

### Feux tricolores de Bâle (fenêtres glissantes de 250 jours)

| Méthode | % fenêtres vertes / orange / rouges | Max exceptions sur 250 j | Dernière fenêtre |
|---|---|---|---|
| Simulation historique | 69 / 25 / 6 | 11 (rouge) | 3 (vert) |
| Normale EWMA | 44 / 51 / 5 | 11 (rouge) | 5 (orange) |
| MC Student-t | 59 / 41 / 0 | 10 (rouge) | 4 (vert) |
| FHS (GARCH) | 80 / 18 / 2 | 11 (rouge) | 3 (vert) |
| Copule t + GARCH | 65 / 30 / 5 | 11 (rouge) | 3 (vert) |

### ES 97,5 % (Acerbi–Székely)

| Méthode | ES 97,5 % moy. (k€) | Exceptions VaR 97,5 % (taux ; attendu 2,5 %) | Z2 | p-value | Perte moy. / ES moy. si > VaR 97,5 % |
|---|---|---|---|---|---|
| Simulation historique | 140 | 105 (2,99 %) | +0,309 | 0,0040 | 1,08 |
| Normale EWMA | 109 | 126 (3,59 %) | +0,667 | 0,0002 | 1,16 |
| MC Student-t | 118 | 128 (3,65 %) | +0,560 | 0,0002 | 1,07 |
| **FHS (GARCH)** | 135 | **88 (2,51 %)** | **+0,032** | **0,3781** | **1,03** |
| Copule t + GARCH | 120 | 102 (2,91 %) | +0,259 | 0,0110 | 1,09 |

L'écart-type de Z2 simulé sous H0 vaut ≈ 0,11 : le Z2 de la normale (+0,667) est à ≈ 6 écarts-types.

### Stress (perte réalisée vs prévisions du même jour)

La pire perte de l'échantillon (442 k€, 4,4 % du notionnel) tombe le **2020-03-09** : Brent −26,6 %
(266 k€ de perte à lui seul), S&P 500 −7,2 %, Nasdaq −6,5 %.

| Méthode | Exceptions mars 2020 (21 j) | VaR 99 % le 2020-03-09 (k€) | Perte / VaR | Perte / ES | Exceptions mars 2022 (23 j) | Exceptions année 2022 (258 j) |
|---|---|---|---|---|---|---|
| Simulation historique | 6 | 124 | 3,56 | 3,73 | 0 | 6 |
| Normale EWMA | 2 | 177 | 2,50 | 2,49 | 0 | 6 |
| MC Student-t | 2 | 186 | 2,37 | 2,29 | 0 | 4 |
| FHS (GARCH) | 2 | 232 | 1,91 | 1,86 | 0 | 2 |
| Copule t + GARCH | 2 | 210 | 2,10 | 1,98 | 0 | 5 |

Mars 2022 (invasion de l'Ukraine) **n'est pas un stress pour ce portefeuille** : long pétrole et or,
sa perte moyenne du mois est négative (gain moyen de 12 k€/jour) et sa pire perte (96 k€) reste sous
toutes les VaR. Le stress de 2022 est étalé sur l'année (hausse des taux, 2 à 6 exceptions selon la
méthode, pire perte 194 k€ le 2022-05-18). Détail complet dans le JSON (`stress`).

### Recoupement des exceptions

Sur 3 511 jours, 3 420 ne déclenchent aucune exception ; **23 jours en déclenchent chez les 5 méthodes
à la fois** (chocs de gap qu'aucun modèle à 1 jour ne voit venir), 21 chez une seule, 21 chez deux,
13 chez trois, 13 chez quatre.

### Diagnostic FHS vs copule (22 dates, une tous les 160 jours)

| | FHS | Copule t + GARCH |
|---|---|---|
| Asymétrie du rendement simulé du portefeuille | −0,52 | +0,07 |
| Kurtosis en excès | 2,43 | 4,93 |
| VaR 99 % simulée (copule / FHS) | 1 | 0,92 |
| ddl moyen de la copule | | 11,5 |

## 7. Lecture critique

**Seule la FHS passe tous les tests au seuil de 5 %** : 38 exceptions pour 35 attendues, Kupiec
p = 0,63, couverture conditionnelle p = 0,17, Z2 p = 0,38 et perte/ES = 1,03. C'est aussi la seule
méthode dont la proportion de fenêtres vertes (80 %) est élevée. Deux raisons : (i) un GARCH très
persistant (la volatilité reste élevée après un choc et décroît lentement) corrige le *clustering de
volatilité* mieux qu'une variance à poids égaux ou qu'un EWMA à mémoire courte ; (ii) le bootstrap de
lignes entières conserve la **forme réelle des chocs**, en particulier l'asymétrie de baisse
(skewness −0,52) et le fait que les actifs risqués chutent ensemble. Son indépendance est limite
(p = 0,07) : le gap du 9 mars 2020 et les enchaînements de 2024–2025 forment des paquets.

**La normale EWMA échoue le plus nettement** : 73 exceptions (2,08 %, soit deux fois le taux nominal),
VaR moyenne la plus basse (1,08 %), pertes de queue 16 % au-dessus de l'ES prévu (perte/ES = 1,16). C'est le défaut classique :
queues trop fines, donc VaR 99 % trop optimiste, et pas de dépendance de queue. Le 9 mars 2020,
la perte vaut 2,5 fois sa VaR.

**Le Student-t multivarié n'est qu'à moitié suffisant.** Il corrige une partie du biais (59 exceptions,
1,68 %, VaR moyenne +6 % par rapport à la normale) mais reste rejeté par Kupiec (p = 0,0002) et par Z2.
Il épaissit les queues *inconditionnelles* avec la **même covariance EWMA** : l'erreur est surtout
dans le niveau de volatilité conditionnelle (mémoire de ≈ 32 jours, un seul ddl pour tout le
portefeuille), pas seulement dans la forme de la loi.

**La simulation historique est un cas d'école** : Kupiec ne la rejette pas (p = 0,055, très
près du seuil) mais Christoffersen la rejette sans appel (p < 10⁻⁴, 7 exceptions suivies le lendemain d'une
autre, 211 fenêtres rouges d'avril 2020 à février 2021). Sa VaR est constante par paliers :
trop basse avant le choc de mars 2020 (124 k€ le 9 mars, perte/VaR = 3,56), puis montée par paliers à
≈ 240 k€ et figée à ce niveau jusqu'à ce que mars 2020 sorte de la fenêtre de 500 jours (début
2022 : 241 k€ fin janvier, 149 k€ fin février). Le bon *nombre* d'exceptions
peut cacher un **mauvais calendrier** : c'est exactement ce que le test d'indépendance sert à voir.

**La copule t + GARCH déçoit malgré sa sophistication** : 52 exceptions, Kupiec rejeté (p = 0,0075).
Le diagnostic en donne la cause : une copule de Student est **symétrique** (asymétrie simulée
+0,07 contre −0,52 pour le bootstrap empirique). Pour un portefeuille long actions, pétrole et
dollar, le risque est dans la queue de *baisse*, que la copule symétrique sous-estime ; sa VaR 99 %
est ≈ 8 % sous celle de la FHS, malgré un kurtosis deux fois plus grand (les queues très lointaines
comptent peu à 99 %). Moralité : plus de paramètres ne veut pas dire plus de réalisme.

**Bâle.** Toutes les méthodes touchent la zone rouge au moins une fois (≥ 10 exceptions sur 250 j),
la FHS seulement de début avril à début juillet 2025 (11 exceptions entre fin avril 2024 et début
avril 2025) ; la simulation historique en 2020-2021 (211 fenêtres) ; la normale (156 fenêtres) et la
copule (149 fenêtres) sur plusieurs épisodes de 2020 à 2025. La proportion de fenêtres rouges
(0 à 6 %) est faible, mais un seul passage en rouge est déjà un signal de contrôle qui, à la date
d'évaluation, conduirait à la majoration maximale du multiplicateur de capital.

**Message de fond** : les 23 jours où les 5 méthodes échouent ensemble montrent qu'une partie
du risque n'est pas modélisable à 1 jour (gaps de marché). La VaR ne dit pas tout : d'où l'ES, et
d'où le suivi de la perte / ES (1,03 à 1,16) plutôt que du seul nombre d'exceptions.

## 8. Limites

* **Un seul portefeuille** de 7 lignes, long seulement, deux indices corrélés à 0,93 : résultats
  illustratifs, pas une loi générale. Le Brent (−27 % en un jour) pèse beaucoup dans les chocs.
* **P&L hypothétique** : poids constants rebalancés chaque jour, pas de coûts, de portage, de
  marge ni de trading intrajournalier ; rendements dans la devise de cotation sans conversion
  EUR (le notionnel de 10 M€ est un habillage). Les pertes en dollars ne sont pas converties.
* Clôtures horaires des CFD de la base de données (17 h New York), jours fériés enjambés.
* **Taille d'échantillon** : 35 exceptions attendues, donc les tests ont une puissance limitée ;
  des p-values proches de 0,05 (Kupiec HS 0,055 ; Christoffersen copule 0,048, FHS 0,070) ne sont
  pas décisives. Aucune correction pour tests multiples (5 méthodes × 4 tests).
* **Bruit Monte Carlo** : 10 000 tirages, soit ≈ 100 points au-delà du quantile 99 % ; erreur
  standard d'environ 1,6 % sur la VaR pour une loi normale (ES 97,5 % : 250 points).
* **Choix fixés, non optimisés** : λ = 0,94, fenêtre 500 j, refit 20 j, grille de ν_c grossière,
  un seul ν pour (c), GARCH(1,1) sans levier (pas de GJR), moyenne nulle. ≈ 9 % des ajustements
  GARCH sont à la frontière α+β = 1 (à 10⁻⁶ près).
* Christoffersen ne teste que la dépendance d'ordre 1 ; le test Z2 suppose la queue prédictive
  fidèle à l'échantillon simulé ; la normale n'a pas de queue « vraie » au-delà de la formule.
* Le backtest FRTB réglementaire se fait au niveau desk (VaR 99 % et 97,5 %, P&L propre et
  hypothétique) : ici, un seul portefeuille et le seul P&L hypothétique.
