# `tests/` : la suite pytest

Lancement : `python -m pytest -q` depuis la racine du projet (environ 1 minute). Graines fixées ; les tolérances sont exprimées en erreurs-types.
Les tests qui exigent les données de marché sont ignorés (`skipif`) quand la variable `UNIVERS_DIR` ne pointe pas vers les CSV 1 h.

| Fichier | Ce qu'il vérifie |
|---|---|
| `conftest.py` | marqueur `sans_donnees` (ignore un test si les CSV 1 h sont absents) |
| `test_analytique.py` | cohérence des formules de référence entre elles (limites connues, parités, différences finies) |
| `test_references_independantes.py` | chaque formule comparée à une méthode différente et exacte (~1e-8) |
| `test_modeles.py` | chaque modèle de simulation contre son prix de référence ; la graine fixe tout |
| `test_reduction_variance.py` | chaque technique reste sans biais et gagne en précision |
| `test_qmc.py` | pont brownien correct, QMC randomisé calibré, pente meilleure que MC |
| `test_barriere.py` | le MC naïf est biaisé haut, le pont brownien ramène à la formule continue |
| `test_americaine.py` | LSM proche de CRR (< 1 %), biaisé bas hors échantillon, encadré par la borne duale |
| `test_grecques.py` | chaque estimateur applicable à moins de 3 erreurs-types de la valeur exacte |
| `test_mlmc.py` | MLMC : tolérance respectée, variance décroissante, moins cher que le MC standard |
| `test_accel.py` | numba : même prix à l'erreur-type près, reproductible, plus rapide |
| `test_validation.py` | couverture des IC95 et parité call-put |
| `test_risque.py` | volet risque : formules fermées, backtests, absence de fuite temporelle |
| `test_requirements.py` | tout paquet tiers importé par `mclab` figure dans `requirements.txt` |
| `test_approfondissement_schemas.py` | Q1 et Q8 : ordres d'Euler et de Milstein, biais faible de Heston, Feller |
| `test_approfondissement_mlmc.py` | Q2 : MLMC avec Milstein, exposants de Giles |
| `test_approfondissement_anova.py` | Q3 : constructions du brownien, ANOVA, dimension effective, RMSE du QMC |
| `test_approfondissement_stratification.py` | Q4 : stratification et hypercube latin contre la théorie |
| `test_approfondissement_importance.py` | Q5 : second moment, dérive optimale, entropie croisée, grandes déviations |
| `test_approfondissement_aad.py` | Q6 : AAD contre pathwise, analytique et bump |
| `test_approfondissement_risque.py` | Q7 : puissance de Kupiec et Christoffersen, contributions d'Euler à l'ES |
| `test_approfondissement_json.py` | `results/approfondissement.json` complet et cohérent avec `docs/approfondissement.md` et le résumé du README |
| `test_readme_tete.py` | README : sections imposées dans l'ordre, premier écran, 400 lignes au plus, tableaux égaux à `results/*.json`, figure héro à deux panneaux et police lisible |
| `test_hygiene_depot.py` | aucune trace interne ni chemin de machine (le `.venv` est ignoré), liens relatifs des docs résolus, `.gitattributes`, erreur guidée quand les données de marché manquent |
| `test_dossiers_documentes.py` | chaque dossier du projet a un README qui cite tous ses fichiers |
