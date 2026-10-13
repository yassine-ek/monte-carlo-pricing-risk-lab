# `figures/` : une image, un script

Toutes les images sont régénérées par les scripts de `scripts/` (graines fixées). La colonne « Script » dit lequel.

| Image | Ce qu'elle montre | Script |
|---|---|---|
| `hero_convergence_et_var.png` | figure héro : convergence MC contre QMC, et exceptions de VaR par méthode contre la bande de Kupiec | `build_readme_tete_01.py` |
| `convergence_mc_vs_qmc.png` | RMSE contre N en log-log : MC (pente −1/2), QMC Sobol avec incréments et avec pont brownien | `run_all.py` |
| `efficacite_vr.png` | facteurs de variance et d'efficacité par technique de réduction de variance | `run_all.py` |
| `smile_heston.png` | smile de volatilité implicite : MC (QE, Euler) contre le semi-fermé de Heston | `run_all.py` |
| `frontiere_exercice_lsm.png` | frontière d'exercice du put américain (LSM contre CRR) et encadrement | `run_all.py` |
| `grecques_mc_vs_exact.png` | pathwise, rapport de vraisemblance et bump contre la valeur exacte (échec du pathwise sur la digitale) | `run_all.py` |
| `mlmc.png` | variance et espérance par niveau, coût fois eps² contre eps (GBM et Heston) | `run_all.py` |
| `biais_barriere.png` | erreur du prix de la barrière discrète : MC naïf, BGK, pont brownien | `run_all.py` |
| `benchmark_numba.png` | chemins par seconde : numpy contre numba parallèle | `run_all.py` |
| `couverture_ic95.png` | couverture observée des IC95 avec la bande binomiale et le contre-exemple de l'événement rare | `run_all.py` |
| `risque_pnl_vs_var.png` | P&L réalisé et −VaR 99 % par méthode, exceptions marquées | `run_risque.py` |
| `risque_exceptions_cumulees.png` | exceptions cumulées par méthode contre l'attendu et la bande binomiale | `run_risque.py` |
| `risque_zoom_covid.png` | zoom sur mars 2020 : P&L, VaR des cinq méthodes, exceptions | `run_risque.py` |
| `risque_zoom_2022.png` | zoom sur 2022 | `run_risque.py` |
| `risque_tableau_recap.png` | tableau récapitulatif des tests par méthode (vert : non rejeté, rouge : rejeté) | `run_risque.py` |
| `appro_q1_ordres_convergence.png` | Q1 : ordres de convergence forts et faibles, GBM, CIR, Heston | `run_approfondissement.py` |
| `appro_q2_mlmc_milstein.png` | Q2 : variance par niveau, ordre faible, coût fois eps², bêta de Heston | `run_approfondissement.py` |
| `appro_q3_pont_acp_dimension_effective.png` | Q3 : variance expliquée par les k premières coordonnées et RMSE du QMC par construction | `run_approfondissement.py` |
| `appro_q4_stratification_lhs.png` | Q4 : facteurs de variance mesurés contre théoriques (stratification, LHS) | `run_approfondissement.py` |
| `appro_q5_importance_sampling.png` | Q5 : second moment contre dérive, entropie croisée, grandes déviations | `run_approfondissement.py` |
| `appro_q6_aad.png` | Q6 : coût de l'AAD contre le bump-and-revalue, exactitude contre références | `run_approfondissement.py` |
| `appro_q7_puissance_backtests_es.png` | Q7 : puissance de Kupiec et de Christoffersen, contributions d'Euler à l'ES | `run_approfondissement.py` |
| `appro_q8_feller.png` | Q8 : fraction de pas tronqués et pente contre le ratio de Feller | `run_approfondissement.py` |
