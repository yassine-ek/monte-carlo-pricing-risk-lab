# Approfondissement : huit questions, huit théorèmes confrontés à la mesure

Huit questions précises, chacune avec un théorème énoncé, une expérience qui le vérifie ou le met en défaut et une conclusion chiffrée. Tout est lu dans `results/approfondissement.json` (régénéré par `scripts/run_approfondissement.py`, graines fixées) ; le détail pas à pas est dans [le notebook](../notebooks/approfondissement.ipynb). Les chronométrages de Q6 et les durées dépendent de la machine. Ce fichier est généré : ne pas l'éditer à la main. Retour au [README](../README.md).

## Q1 À quelle vitesse l'erreur de discrétisation tend-elle vers 0 pour Euler et Milstein (forte et faible), et ces ordres survivent-ils à Heston ?

*Théorème* : Euler : ordre fort 1/2, faible 1. Milstein (terme b b' (dW^2 - dt)/2) : ordre fort 1, faible 1 (Kloeden-Platen), pour des coefficients réguliers.

- GBM, erreur forte (même brownien) : Euler en dt^0,49 [0,49 ; 0,49] (théorie 1/2) ; Milstein en dt^1,00 en L2 et dt^0,94 en L1 (théorie 1 ; la pente locale L1 vaut 0,99 aux pas les plus fins).
- GBM, erreur faible : Euler 0,72 ± 0,09 (call, bruité), Milstein 0,98 ; sans bruit (E[S^2] exact) 0,997 et 0,998 : ordre faible 1 pour les deux.
- CIR : les ordres du théorème exigent Feller. Feller respecté (F = 3,0) : Euler 0,65, Milstein 1,07 ; Feller violé (F = 0,48, paramètres du projet) : Euler 0,40, Milstein 0,68.
- Heston, biais faible du prix : Euler full-truncation 0,522 à 4 pas -> 0,028 à 32 pas -> 0,004 ± 0,002 à 128 pas ; QE : -0,004 à 4 pas, compatible avec 0 (chi2 p = 0,30), erreur-type 0,002.

![Q1 : pentes log-log des erreurs forte et faible d'Euler et de Milstein sur GBM, CIR et Heston](../figures/appro_q1_ordres_convergence.png)

**Comment lire ce graphique.** Quatre panneaux en log-log où la pente est l'ordre de convergence : erreur forte du GBM, erreur faible du GBM (barres = IC95), erreur forte du CIR (traits pleins = Feller respecté, pointillés = violé) et biais du prix Heston (Euler full-truncation contre QE). Une courbe qui s'écarte de la droite de référence contredit le théorème ou sort de son domaine de validité.

## Q2 Milstein rend-il le MLMC en O(eps^-2) : que valent alpha, beta, gamma mesurés, et quel gain de coût ?

*Théorème* : Giles (2008) : si |E[P_l - P]| <= c 2^(-alpha l) avec alpha >= min(beta, gamma)/2, V_l <= c 2^(-beta l) et C_l <= c 2^(gamma l), le coût pour une RMSE eps est O(eps^-2) si beta > gamma, O(eps^-2 (log eps)^2) si beta = gamma, O(eps^-(2+(gamma-beta)/alpha)) si beta < gamma.

- beta (variance par niveau, V_l ~ 2^(-beta l)) : Euler 0,95 [0,93 ; 0,98], Milstein 1,95 [1,92 ; 1,98] (théorie 1 et 2) ; alpha (ordre faible) : Euler 0,62 [0,32 ; 0,93], Milstein 0,98 [0,98 ; 0,98] ; gamma = 1 (pas fins).
- Giles : Euler beta ~ gamma donne eps^-2 (log eps)^2 ; Milstein beta > gamma donne eps^-2. Coût mesuré en eps^-x : x = 2,20 [1,98 ; 2,43] (Euler), 1,94 [1,82 ; 2,06] (Milstein), 3,08 pour le MC standard.
- Gain de coût de Milstein sur Euler (pas fins) : x1,3 à eps = 0.05 -> x3,6 à eps = 0.002 (coût x eps^2 : Euler 1526, Milstein 429) ; la croissance en log^2 d'Euler est masquée par le niveau 0 sur cette plage.
- Heston : beta passe de 0,81 (Euler) à 0,96 (Milstein) sur le cas du projet, mais atteint 2,12 [1,99 ; 2,24] seulement si Feller est respecté ET si l'aire de Lévy est négligeable (rho -> -1).

![Q2 : variance et espérance par niveau, coût fois epsilon carré, bêta de Heston selon Feller](../figures/appro_q2_mlmc_milstein.png)

**Comment lire ce graphique.** Quatre panneaux : variance par niveau (pente = -beta), espérance par niveau (pente = -alpha), coût multiplié par eps^2 (une courbe plate est un coût en eps^-2 pur ; Milstein doit être plat et sous Euler) et beta de Heston selon Feller et l'aire de Lévy.

## Q3 Pourquoi le pont brownien (et l'ACP) rendent-ils le QMC plus précis sur l'asiatique : que mesure la dimension effective ?

*Théorème* : Décomposition ANOVA f = somme_u f_u ; variance expliquée par {1..k} = Var(E[f | Z_1..Z_k]) / Var f (Sobol fermé). Koksma-Hlawka : |I_N - I| <= V_HK(f) D*_N, avec D*_N ~ (log N)^s / N en dimension s : une fonction presque entièrement portée par ses s premières coordonnées (s petit) est intégrée à la vitesse de la projection de dimension s.

- Variance expliquée par la 1re coordonnée (d = 16) : 12,1 % (incrémentale), 72,6 % (pont), 98,4 % (ACP) ; par les 4 premières : 47,4 / 98,2 / 100,0 %.
- Dimension effective de troncature à 99 % : 14 / 7 / 2 à d = 16, 56 / 7 / 2 à d = 64 : seule la construction incrémentale en dépend (résolution 8 au-delà de k = 16).
- RMSE à N = 2^14 (d = 16) : MC 5.4e-02, incrémentale 6.2e-03, pont 1.2e-03, ACP 4.9e-04 (pentes -0,50 / -0,73 / -0,89 / -0,97).
- Koksma-Hlawka : l'erreur suit la discrépance de la projection qui porte la variance ; gain de l'ACP sur le pont : x2,5 à d = 16, x4,6 à d = 64.

![Q3 : variance expliquée par les premières coordonnées et RMSE du QMC pour trois constructions du brownien](../figures/appro_q3_pont_acp_dimension_effective.png)

**Comment lire ce graphique.** À gauche, part de variance expliquée par les k premières coordonnées pour les trois constructions (plus la courbe monte vite, plus la dimension effective est petite) ; à droite, RMSE du QMC randomisé contre N pour d = 16, 32, 64 : ACP sous le pont, lui-même sous l'incrémental, MC en dernier.

## Q4 Stratifier le brownien terminal (proportionnel vs Neyman) ou utiliser un hypercube latin : combien de variance gagne-t-on, et la théorie le prédit-elle ?

*Théorème* : Stratification : Var = somme p_h^2 sigma_h^2 / n_h ; proportionnelle -> (1/n) somme p_h sigma_h^2 <= sigma^2/n (variance totale) ; Neyman n_h ~ p_h sigma_h -> (1/n) (somme p_h sigma_h)^2 <= proportionnelle (Cauchy-Schwarz). LHS (Stein 1987) : variance asymptotique (1/n)(sigma^2 - somme_j sigma_j^2), il ne retire que la partie additive.

- Stratification de la normale terminale, H = 64 (n = 4096), européen ATM : x92 mesuré (proportionnelle) contre x89 prédit par sigma^2 / somme(p_h sigma_h^2) ; Neyman avec sigma_h connus x1110 (théorie x1125). Précision de chaque facteur mesuré : ~4 %.
- Call OTM K = 140 (sigma_h très inégaux) : proportionnelle x6,7, Neyman x237 (oracle) et x180 (pilote n/8). Asiatique 16 dates (Z_1 du pont stratifiée) : x3,4 / x2,6 (pilote) / x5,7 (oracle), théorie 3,6 / 5,2 (proportionnelle / oracle).
- LHS (Stein) : somme des indices de Sobol d'ordre 1 = 0,749 (incrémentale), 0,896 (pont) -> gain asymptotique prédit x4,0 / x9,6 ; mesuré à n = 4096 : x3,8 / x9,7.
- Dimension 1 : le LHS est une stratification à n strates, variance en n^-2,02 contre n^-0,99 pour le MC.

![Q4 : facteurs de variance mesurés et théoriques de la stratification et de l'hypercube latin](../figures/appro_q4_stratification_lhs.png)

**Comment lire ce graphique.** Facteur de variance à H = 64 strates (barre = mesuré, trait = théorie : une barre sur son trait veut dire que la formule prévoit le gain), gain de variance du LHS sur l'asiatique, puis variance contre n en dimension 1 où le LHS suit n^-2.

## Q5 Quelle dérive de Girsanov minimise la variance d'un estimateur d'événement rare, et l'entropie croisée la retrouve-t-elle ?

*Théorème* : Pour Y ~ N(mu,1) et w = exp(-mu Y + mu^2/2), Var = M(mu) - I^2, M(mu) = E[f(Z)^2 exp(-mu Z + mu^2/2)] convexe : minimum unique mu*. Entropie croisée : mu_CE = E[f(Z) Z] / E[f(Z)]. Grandes déviations : pour {Z > a}, M(mu) = e^{mu^2} P(Z > a + mu), mu* ~ a, ln M(mu*) ~ 2 ln P (efficacité logarithmique).

- Call K = 180 : le second moment M(mu) est convexe ; mu* = 3,386 (variance divisée par 882), contre mu = z_K = 2,939 (x738, dérive centrée sur le strike) et mu_CE = 3,491 (x873). L'entropie croisée par 8 itérations de 20 000 tirages donne mu = 3,492 ± 0,003 (10 graines).
- Défaut rare (au moins 20 défauts sur 100, P = 4.04e-04) : mu* = -3,436, mu_CE = -3,465, variance divisée par 1098 (CE : 1097).
- La variance empirique (n = 400000) retrouve M(mu) - I^2 à 0,3 % près sur les dérives d'importance.
- Grandes déviations : ln M(mu*) / (2 ln P) = 0,844 (a = 2) -> 0,950 (a = 6) -> 0,992 (a = 20) ; l'erreur relative de l'IS ne vaut que 4,9 à a = 20 (MC naïf : 1.9e+44).

![Q5 : variance par tirage en fonction de la dérive d'importance, avec optimum et entropie croisée](../figures/appro_q5_importance_sampling.png)

**Comment lire ce graphique.** Variance par tirage en fonction de la dérive mu pour chaque cas : une cuvette dont le fond est mu* (les repères montrent le choix naïf et l'entropie croisée) ; à droite, le rapport logarithmique qui tend vers 1 quand l'événement devient rare.

## Q6 Peut-on obtenir toutes les grecques d'un payoff pour ~3 à 4 fois le coût du prix, et sont-elles exactes ?

*Théorème* : Principe du gradient bon marché (Griewank) : TEMPS{F, grad F} <= omega TEMPS{F}, omega ~ 3-4, indépendant du nombre m de paramètres, par propagation d'adjoints bar x = dP/dx en sens inverse du graphe. Le bump-and-revalue coûte 2 m réévaluations.

- Exactitude : sur les mêmes chemins (n = 400000, 12 dates), l'AAD en mode inverse égale le pathwise écrit à la main à 6e-13 près en valeur absolue (grecques de l'ordre de 10 à 500) ; boucle et version vectorisée coïncident.
- Contre des références indépendantes : européen vs Black-Scholes z = 0,03 / 1,56 / -0,34 (delta / vega / rho) ; asiatique géométrique vs différences finies de la formule fermée z = 0,53 / 0,15 / 0,59.
- Coût relatif au prix (asiatique, 64 dates, 67 grecques dont 64 vegas par date) : AAD 3,3x (boucle) et 2,5x (vectorisé), contre 153x pour le seul bump-and-revalue des vegas (129 réévaluations).
- Le rapport AAD/prix reste dans 2,3-3,3x (boucle) quand m passe de 11 à 67 (Griewank : 3 à 4 en comptage d'opérations) ; accélération sur le bump : x46 (boucle) à x62 (vectorisé). Temps numpy : dépendent de la machine.

![Q6 : coût des grecques par AAD contre bump-and-revalue, et écart aux références indépendantes](../figures/appro_q6_aad.png)

**Comment lire ce graphique.** À gauche, coût de toutes les grecques rapporté au coût du prix quand le nombre de grecques m augmente : l'AAD reste plat, le bump-and-revalue croît avec m (les temps dépendent de la machine) ; à droite, écart de l'AAD à des références indépendantes en erreurs-types (dans la bande verte +/-3 : compatible).

## Q7 Combien d'années de backtest faut-il pour détecter un modèle de VaR à 2 % d'exceptions au lieu de 1 %, et d'où vient l'ES du portefeuille ligne par ligne ?

*Théorème* : Puissance = P(LR > chi2_{1-a} | H1) ; sous H1 le nombre d'exceptions est Binomial(T, p1). Théorème d'Euler : F homogène de degré 1 => F(w) = somme_i w_i dF/dw_i ; pour l'ES, w_i dES/dw_i = E[perte de la ligne i | L >= VaR] et la somme des contributions vaut l'ES.

- Kupiec, modèle à 2 % au lieu de 1 % : puissance 24 % à 1 an, 54 % à 2 ans, 87 % à 5 ans, 98 % à 10 ans ; 80 % de puissance durable après 1130 jours (4,5 ans ; formule normale 3,9 ans). La taille réelle du test vaut 9,5 % à 1 an (pas 5 %).
- Bâle : un modèle à 2 % est en zone verte 44 % du temps sur 250 jours. Exceptions groupées de taux correct 1 % (pi11 = 0,20) : Christoffersen (indépendance) rejette 28 / 43 / 72 / 92 % à 1/2/5/10 ans, Kupiec seulement 9 % à 10 ans.
- ES FHS 97,5 % au 9 mars 2020 : 247 659 EUR ; somme des contributions d'Euler = ES à 2e-16 près ; dérivées numériques w_i dES/dw_i = contributions à 1e-10 ; ES(2w)/ES(w) = 2.000000.
- Parts de l'ES (poids du capital) : SPX 50 % (15 %), NSX 29 % (10 %), BCO 21 % (10 %), XAU -6 % (15 %) ; moyenne sur 59 dates : SPX 22 %, NSX 17 %, BCO 28 %, XAU 13 %.

![Q7 : puissance des tests de Kupiec et de Christoffersen, et contributions d'Euler à l'ES](../figures/appro_q7_puissance_backtests_es.png)

**Comment lire ce graphique.** Puissance du test de Kupiec contre l'horizon de backtest (courbe en dents de scie : loi discrète), taux de rejet de Christoffersen (trait plein) contre Kupiec (pointillé) sur des exceptions groupées, et contributions d'Euler de chaque ligne à l'ES (une barre négative est une ligne qui couvre).

## Q8 Que vaut la condition de Feller pour la discrétisation : quelle fraction de pas est tronquée selon 2 kappa theta / xi^2 ?

*Théorème* : CIR : le bord 0 est inatteignable ssi F = 2 kappa theta / xi^2 >= 1 (Feller 1951). La loi stationnaire est Gamma(F, xi^2/(2 kappa)) : P(v < eps) ~ eps^F. L'Euler full-truncation tronque quand v_n + xi sqrt(v_n dt) Z < 0, donc avec probabilité ~ dt^F.

- Pente log-log de la part des pas tronqués en dt : F = 0,25 -> 0,19 ; F = 0,50 -> 0,43 ; F = 1,00 -> 0,89 ; F = 2,00 -> 1,74 ; F = 4,00 -> 3,19 (la pente croît avec F = 2 kappa theta / xi^2 et reste sous F : régime non asymptotique).
- À 16 pas, la troncature touche 27 % des pas pour F = 0,25, 14 % pour F = 0,5 (cas du projet : 0,48), 4,2 % pour F = 1, 0,013 % pour F = 4.
- Biais relatif de E[v_T^+] à 16 pas : 1,3 % (F = 0,5) contre 0,23 % (F = 4).

![Q8 : fraction de pas tronqués selon le ratio de Feller et pente log-log correspondante](../figures/appro_q8_feller.png)

**Comment lire ce graphique.** À gauche, fraction de pas tronqués contre dt en log-log pour plusieurs F = 2 kappa theta / xi^2 ; à droite, la pente de ces droites contre F : elle croît avec F et reste sous la bissectrice tant que dt n'est pas assez petit.

