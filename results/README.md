# `results/` : sorties chiffrées

Tous les nombres du `README.md` sont lus dans ces fichiers (jamais tapés à la main). Aucun fichier n'est lourd (moins de 1 Mo) ni ne contient de donnée de marché brute.

| Fichier | Contenu | Produit par |
|---|---|---|
| `validation.json` | 42 cas validés + 3 contre-exemples volontaires du pricing (estimation, erreur-type, référence, z), couverture des IC95, convergence MC/QMC, efficacité, barrière, MLMC, smile, benchmark numba | `scripts/run_all.py` |
| `tableau_validation.md` | le même tableau de validation en Markdown | `scripts/run_all.py` |
| `risque_backtest.json` | volet risque : cinq méthodes, backtests (Kupiec, Christoffersen, Bâle, Z2), stress, recoupement des exceptions | `scripts/run_risque.py` |
| `approfondissement.json` | un bloc par question Q1 à Q8 : question, théorème, attendu, chiffres, conclusion, durée | `scripts/run_approfondissement.py` |
