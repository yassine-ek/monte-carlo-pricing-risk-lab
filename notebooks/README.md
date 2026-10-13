# `notebooks/` : lecture guidée

Chaque notebook suit le fil intuition, formule, ce qu'on attend, expérience, lecture du résultat. Exécution :
`python -m jupyter nbconvert --to notebook --execute --inplace notebooks/<nom>.ipynb` depuis la racine du projet.

| Fichier | Contenu |
|---|---|
| `visite_guidee.ipynb` | visite des modules du pricing : estimateur et erreur-type, Heston et Merton, réduction de variance, QMC, barrière, LSM, grecques, MLMC (~1 min) |
| `approfondissement.ipynb` | huit chapitres Q1 à Q8 : question, intuition, théorème en LaTeX, attente, expérience, résultat, conclusion vérifiée par `assert` (~10 min) ; généré par `scripts/build_nb_appro_01.py` |
