# `docs/` : documentation détaillée

Le `README.md` à la racine donne les résultats clés ; ces fichiers portent le détail.

| Fichier | Contenu |
|---|---|
| `risque.md` | volet risque en entier : objectif, pourquoi un Monte Carlo sur un portefeuille linéaire, portefeuille et données, cinq méthodes de VaR/ES, protocole causal, backtests, résultats lus dans `results/risque_backtest.json`, lecture critique et limites |
| `validation.md` | tables complètes de `results/validation.json` (42 cas validés + 3 contre-exemples volontaires), couverture des IC95, gains de réduction de variance, encadrement du put américain, MLMC, numba, galerie de figures et lectures ; tableaux réécrits par `scripts/run_all.py` |
| `approfondissement.md` | les huit questions Q1 à Q8 (théorème, chiffres, figure) ; généré par `scripts/run_approfondissement.py` depuis `results/approfondissement.json` |
| `positionnement.md` | comparaison avec environ 70 dépôts publics de Monte Carlo en finance : ce qu'ils font bien, ce que ce labo ajoute, ce qu'il ne fait pas |
