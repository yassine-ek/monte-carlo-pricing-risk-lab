# Positionnement face aux dépôts GitHub

À partir d'une recherche de ~70 dépôts publics sur le Monte Carlo en finance (README et racines balayés le 2026-10-13, code non exécuté ;
un notebook peut contenir ce que le README ne cite pas, donc ordres de grandeur) :
**seulement 4 README sur 70** parlent d'intervalle de confiance ou d'erreur-type, **5 sur 70** de convergence,
**2 sur 70** d'antithétique, **2 sur 70** de variable de contrôle, **0 sur 70** de stratification (traitée ici : proportionnelle, Neyman, hypercube latin), **1 sur 70** de pont brownien.

| Dépôt (étoiles) | Ce qu'il fait bien | Ce que ce labo ajoute |
|---|---|---|
| [cantaro86/Financial-Models-Numerical-Methods](https://github.com/cantaro86/Financial-Models-Numerical-Methods) (7 492) | notebooks pédagogiques : Heston, Merton, Lévy, EDP, américain | Monte Carlo au centre : IC95 partout, tests pytest, antithétique / contrôle / Sobol + pont, étude de convergence |
| [ArturSepp/StochVolModels](https://github.com/ArturSepp/StochVolModels) (239) | bon réflexe « MC validé contre Fourier », noyaux numba | réduction de variance, QMC, courbes erreur-vs-N, tests de couverture ; ici aussi Heston QE vs semi-fermé, mais avec tolérance en erreurs-types |
| [QMCSoftware/qmcpy](https://github.com/QMCSoftware/qmcpy) (93) | référence QMC : Sobol, lattices, critères d'arrêt adaptatifs | **application de bout en bout** : asiatique avec Sobol brouillé + pont brownien, erreur-type par brouillages, pente mesurée vs MC (qmcpy reste plus complet comme bibliothèque, ici seulement ce qu'il faut au pricing) |
| [luphord/longstaff_schwartz](https://github.com/luphord/longstaff_schwartz) (52) | paquet LSM propre, CI, documentation | **borne haute duale**, évaluation hors échantillon explicite (biais bas vs biais de prévoyance), comparaison à un arbre bermudéen sur les mêmes dates |
| [Yunfan-Wang/CMU15418-High-performance-Longstaff-Schwartz-Solver](https://github.com/Yunfan-Wang/CMU15418-High-performance-Longstaff-Schwartz-Solver) (54) | LSM CPU + CUDA (13,9x), limites écrites honnêtement : « pas d'erreur contre un oracle exact » | l'oracle exact (arbre CRR 2 000 pas, bermudéen sur les mêmes dates), les IC et l'encadrement (pas de GPU ici) |
| [auto-differentiation/xad](https://github.com/auto-differentiation/xad) (432) | AAD adjointe pour le MC (C++) | comparaison **pathwise vs LR vs bump (CRN)** avec erreur-type, y compris sur un payoff discontinu où le pathwise échoue ; AAD en mode inverse **codée à la main** sur un GBM discret, exacte à 1e-12 près contre le pathwise (pas de bibliothèque d'AD générique) |
| [lballabio/QuantLib](https://github.com/lballabio/QuantLib) (7 666) | référence mondiale, moteurs MC et formules fermées | autre échelle : QuantLib est la vérité terrain ; ce labo est lisible de bout en bout et fournit les études (convergence, couverture) qu'il ne propose pas prêtes à l'emploi |
| [google/tf-quant-finance](https://github.com/google/tf-quant-finance) (5 523) | MC sur GPU/TPU, Sobol, Heston, SABR | archivé (non maintenu) ; ici le chemin est inverse : peu de vitesse brute, mais chaque prix est validé et chaque gain chiffré |

Ce qui est **en plus** dans ce labo : prix MC toujours livrés avec erreur-type ; tolérance exprimée en erreurs-types ;
**test de couverture** des IC95 (1 000 répétitions, avec le contre-exemple de l'événement rare) ; parité call-put sur les prix
MC ; tableau « facteur de variance et efficacité » par technique ; pente de convergence mesurée ; Greeks par trois méthodes
dont une qui échoue volontairement sur la digitale ; LSM encadré par le haut ; MLMC avec exposants de coût (Euler et Milstein) ; stratification, hypercube latin, importance sampling optimal et AAD avec la théorie énoncée puis confrontée à la mesure ; volet risque à cinq méthodes backtestées ; figures et JSON
régénérés par une seule commande. Ce qu'ils font **mieux ou que nous ne faisons pas** : couverture produit (QuantLib), AAD générique (xad),
GPU (tf-quant-finance, CMU15418), cadre QMC adaptatif (qmcpy), pédagogie sur EDP/Fourier (cantaro86).

Retour au [README](../README.md).
