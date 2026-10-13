# `mclab/` : le paquet Python

Le laboratoire Monte Carlo : estimateurs avec erreur-type et IC95, modèles de simulation, références de validation, volet risque et modules de
l'approfondissement. Chaque module est testé par un fichier de `tests/`. Import : `from mclab import ...` depuis la racine du projet.

| Module | Ce qu'il fait |
|---|---|
| `__init__.py` | déclare le paquet |
| `utils.py` | briques communes : `ResultatMC` (prix, erreur-type, IC95), paramètres Heston et Merton, payoffs, volatilité implicite |
| `analytique.py` | prix de référence : Black-Scholes et grecques, Heston semi-fermé, Merton (série), asiatique géométrique, barrière continue, arbre CRR américain |
| `modeles.py` | simulation de trajectoires : GBM exact, Heston (Euler full-truncation et QE d'Andersen), Merton exact |
| `reduction_variance.py` | antithétique, variable de contrôle, importance sampling par translation de dérive, facteurs de variance et d'efficacité |
| `qmc.py` | quasi-Monte Carlo : Sobol brouillé, normale inverse, pont brownien (et ACP), QMC randomisé, étude de convergence |
| `barriere.py` | option à barrière discrète : MC naïf, décalage BGK, pont brownien |
| `americaine.py` | put américain par Longstaff-Schwartz hors échantillon, frontière d'exercice, borne haute duale |
| `grecques.py` | grecques par pathwise, rapport de vraisemblance et bump-and-revalue (nombres aléatoires communs) |
| `mlmc.py` | Monte Carlo multi-niveaux de Giles : Euler et Milstein, niveaux adaptatifs, exposants alpha/beta/gamma |
| `accel.py` | simulation Heston QE accélérée par numba (parallèle) et benchmark contre numpy |
| `validation.py` | contrôles des estimateurs eux-mêmes : couverture des IC95 sur répétitions, parité call-put sur prix MC |
| `risque.py` | volet risque : VaR 99 % et ES 97,5 % par cinq méthodes, backtests de Kupiec, Christoffersen, Bâle, Acerbi-Székely ; données lues via `UNIVERS_DIR` |
| `schemas.py` | schémas d'Euler et de Milstein (GBM, CIR), biais faible de Heston, régression des ordres de convergence |
| `anova.py` | dimension effective : indices de Sobol fermés (pick-freeze) et d'ordre 1, constructions incrémentale / pont / ACP, RMSE du QMC |
| `stratification.py` | stratification (allocation proportionnelle et de Neyman), hypercube latin |
| `importance.py` | importance sampling gaussien optimal : second moment, dérive optimale, entropie croisée, grandes déviations |
| `aad.py` | différentiation adjointe (AAD) codée à la main pour un GBM discret, contre pathwise et bump-and-revalue |
| `puissance.py` | puissance des backtests de VaR (Kupiec exact et simulé, Christoffersen sur exceptions groupées), zones de Bâle |
| `contributions_es.py` | contributions d'Euler à l'Expected Shortfall (somme des contributions = ES) |
| `experiences.py` | les huit expériences Q1 à Q8 de l'approfondissement, chacune renvoie un bloc JSON (appelées par le script et le notebook) |
