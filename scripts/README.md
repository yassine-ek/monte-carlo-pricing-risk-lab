# `scripts/` : relancer les résultats

À lancer depuis la racine du projet. Les scripts écrivent dans `results/`, `figures/` et `docs/` et réinjectent les blocs générés du `README.md`.

| Script | Ce qu'il produit | Durée |
|---|---|---|
| `run_all.py` | validation du pricing : `results/validation.json`, `results/tableau_validation.md`, 9 figures de pricing, tableaux de `docs/validation.md` | ~2,5 min |
| `figures.py` | tracés des 9 figures de pricing (appelé par `run_all.py`) | |
| `run_risque.py` | backtest VaR/ES de cinq méthodes : `results/risque_backtest.json`, 5 figures `risque_*.png` ; exige `UNIVERS_DIR` (données non fournies) | ~2,5 min |
| `run_approfondissement.py` | huit expériences Q1 à Q8 : `results/approfondissement.json`, 8 figures `appro_*.png`, `docs/approfondissement.md` et résumé de 8 lignes dans le README (`--only Q1,Q4`, `--readme-seul`) | ~7 min |
| `figures_appro.py` | tracés des figures de l'approfondissement (appelé par le script et par le notebook) | |
| `build_nb_appro_01.py` | génère `notebooks/approfondissement.ipynb` (un chapitre par thème) | quelques secondes |
| `build_readme_tete_01.py` | figure héro `figures/hero_convergence_et_var.png` (2 panneaux, police ≥ 12,5 pt) et blocs de tête du README (résultats clés, tableaux, variantes), lus dans `results/*.json` | quelques secondes |
