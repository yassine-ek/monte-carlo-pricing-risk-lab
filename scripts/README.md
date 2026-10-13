# `scripts/` : relancer les résultats

À lancer depuis la racine du projet. Les scripts écrivent dans `results/` et `figures/` et réinjectent les tableaux du `README.md`.

| Script | Ce qu'il produit | Durée |
|---|---|---|
| `run_all.py` | validation du pricing : `results/validation.json`, `results/tableau_validation.md`, 9 figures de pricing, tableaux du README | ~2,5 min |
| `figures.py` | tracés des 9 figures de pricing (appelé par `run_all.py`) | |
| `run_risque.py` | backtest VaR/ES de cinq méthodes : `results/risque_backtest.json`, 5 figures `risque_*.png` ; exige `UNIVERS_DIR` (données non fournies) | ~2,5 min |
| `run_approfondissement.py` | huit expériences Q1 à Q8 : `results/approfondissement.json`, 8 figures `appro_*.png`, section du README (`--only Q1,Q4`, `--readme-seul`) | ~7 min |
| `figures_appro.py` | tracés des figures de l'approfondissement (appelé par le script et par le notebook) | |
| `build_nb_appro_01.py` | génère `notebooks/approfondissement.ipynb` (un chapitre par thème) | quelques secondes |
| `build_readme_tete_01.py` | figure héro `figures/hero_convergence_et_var.png` et tableaux de tête du README, lus dans `results/*.json` (`--tests N` met à jour le badge) | quelques secondes |
