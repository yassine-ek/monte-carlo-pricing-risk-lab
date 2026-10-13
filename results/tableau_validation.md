## prix

| Cas | Méthode | Estimation ± SE | IC95 | Référence | z | Verdict |
|---|---|---|---|---|---|---|
| Call européen BS | MC naïf | 8.8668 ± 0.0194 | [8.8287 ; 8.9049] | 8.9160 | -2.53 | OK |
| Call européen BS | MC antithétique | 8.8968 ± 0.0149 | [8.8676 ; 8.9260] | 8.9160 | -1.29 | OK |
| Call européen BS | VC sur S_T | 8.9155 ± 0.0083 | [8.8992 ; 8.9317] | 8.9160 | -0.07 | OK |
| Call européen BS | QMC Sobol randomisé (16 x 2^14) | 8.9159 ± 0.0001 | [8.9157 ; 8.9161] | 8.9160 | -1.98 | OK |
| Call Heston | Euler full-truncation (100 pas) | 8.2357 ± 0.0218 | [8.1931 ; 8.2784] | 8.1950 | +1.87 | OK |
| Call Heston | QE d'Andersen (20 pas) | 8.2135 ± 0.0138 | [8.1866 ; 8.2405] | 8.1950 | +1.35 | OK |
| Call Heston | QE numba parallèle (50 pas) | 8.1876 ± 0.0048 | [8.1781 ; 8.1971] | 8.1950 | -1.53 | OK |
| Call Merton | schéma exact (Poisson) | 10.1423 ± 0.0221 | [10.0989 ; 10.1856] | 10.1272 | +0.68 | OK |
| Asiatique géométrique (12 dates) | MC naïf | 5.2016 ± 0.0111 | [5.1798 ; 5.2234] | 5.1909 | +0.97 | OK |
| Asiatique géométrique (12 dates) | QMC pont (16 x 2^12) | 5.1907 ± 0.0008 | [5.1891 ; 5.1924] | 5.1909 | -0.18 | OK |
| Asiatique arithmétique (12 dates) | MC naïf | 5.3659 ± 0.0181 | [5.3304 ; 5.4014] | 5.3782 | -0.68 | OK |
| Asiatique arithmétique (12 dates) | MC antithétique | 5.3674 ± 0.0136 | [5.3408 ; 5.3940] | 5.3782 | -0.80 | OK |
| Asiatique arithmétique (12 dates) | VC asiatique géométrique | 5.3783 ± 0.0005 | [5.3774 ; 5.3793] | 5.3782 | +0.18 | OK |
| Asiatique arithmétique (12 dates) | QMC pont (16 x 2^12) | 5.3776 ± 0.0009 | [5.3755 ; 5.3796] | 5.3782 | -0.69 | OK |
| Call très OTM (K=180) | MC naïf | 0.0158 ± 0.0012 | [0.0134 ; 0.0182] | 0.0177 | -1.51 | OK |
| Call très OTM (K=180) | importance sampling (mu=2.94) | 0.0177 ± 0.0001 | [0.0176 ; 0.0178] | 0.0177 | +0.35 | OK |
| Down-and-out call (H=90, 50 dates) | MC naïf (grille) | 7.7682 ± 0.0194 | [7.7302 ; 7.8062] | 7.3004 | +24.14 | échec attendu (démo) |
| Down-and-out call (H=90, 50 dates) | décalage BGK | 7.3134 ± 0.0192 | [7.2758 ; 7.3510] | 7.3004 | +0.68 | OK |
| Down-and-out call (H=90, 50 dates) | pont brownien | 7.3131 ± 0.0190 | [7.2758 ; 7.3504] | 7.3004 | +0.67 | OK |
| Put américain LSM (50 dates) | Longstaff-Schwartz, Laguerre deg. 3, évaluation indépendante | 5.7703 ± 0.0096 | [5.7515 ; 5.7891] | 5.7849 | -1.53 | OK |
| Put américain LSM, cas Longstaff-Schwartz (S0=36, K=40) | Longstaff-Schwartz, 50 dates | 4.4712 ± 0.0061 | [4.4593 ; 4.4831] | 4.4780 | -1.13 | OK |
| Put américain LSM, cas Longstaff-Schwartz (S0=40, K=40) | Longstaff-Schwartz, 50 dates | 2.3058 ± 0.0054 | [2.2951 ; 2.3165] | 2.3140 | -1.50 | OK |
| Put américain LSM, cas Longstaff-Schwartz (S0=44, K=40) | Longstaff-Schwartz, 50 dates | 1.1012 ± 0.0055 | [1.0904 ; 1.1121] | 1.1101 | -1.60 | OK |
| Put américain LSM (10 dates) | Longstaff-Schwartz | 5.7205 ± 0.0097 | [5.7015 ; 5.7396] | 5.7318 | -1.16 | OK |
| Put américain, borne duale (10 dates) | Rogers / Haugh-Kogan (valeur LSM deg. 7, quadrature) | 5.7650 ± 0.0029 | [5.7593 ; 5.7707] | 5.7318 | +11.34 | borne valide |
| MLMC GBM (Euler) | MLMC de Giles, eps=0.01 | 8.9169 ± 0.0071 | [8.9031 ; 8.9308] | 8.9160 | +0.13 | OK |
| MLMC Heston (Euler) | MLMC de Giles, eps=0.05 | 8.2639 ± 0.0354 | [8.1946 ; 8.3333] | 8.1950 | +1.95 | OK |

## grecques

| Cas | Méthode | Estimation ± SE | IC95 | Référence | z | Verdict |
|---|---|---|---|---|---|---|
| Grecque delta (call) | pathwise | 0.5792 ± 0.0008 | [0.5776 ; 0.5808] | 0.5793 | -0.08 | OK |
| Grecque delta (call) | lr | 0.5773 ± 0.0020 | [0.5734 ; 0.5811] | 0.5793 | -1.00 | OK |
| Grecque delta (call) | bump | 0.5792 ± 0.0008 | [0.5776 ; 0.5808] | 0.5793 | -0.11 | OK |
| Grecque gamma (call) | lr | 0.0194 ± 0.0002 | [0.0190 ; 0.0197] | 0.0196 | -0.93 | OK |
| Grecque gamma (call) | bump | 0.0193 ± 0.0002 | [0.0190 ; 0.0196] | 0.0196 | -1.35 | OK |
| Grecque vega (call) | pathwise | 39.0102 ± 0.1053 | [38.8039 ; 39.2165] | 39.1043 | -0.89 | OK |
| Grecque vega (call) | lr | 38.7572 ± 0.3734 | [38.0252 ; 39.4891] | 39.1043 | -0.93 | OK |
| Grecque vega (call) | bump | 39.0091 ± 0.1053 | [38.8028 ; 39.2154] | 39.1043 | -0.90 | OK |
| Grecque delta (digitale) | pathwise | 0.0000 ± 0.0000 | [0.0000 ; 0.0000] | 0.0196 | +inf | échec attendu (démo) |
| Grecque delta (digitale) | lr | 0.0195 ± 0.0000 | [0.0194 ; 0.0196] | 0.0196 | -0.69 | OK |
| Grecque delta (digitale) | bump | 0.0195 ± 0.0001 | [0.0192 ; 0.0198] | 0.0196 | -0.48 | OK |
| Grecque vega (digitale) | pathwise | 0.0000 ± 0.0000 | [0.0000 ; 0.0000] | -0.3910 | +inf | échec attendu (démo) |
| Grecque vega (digitale) | lr | -0.3986 ± 0.0064 | [-0.4111 ; -0.3861] | -0.3910 | -1.18 | OK |
| Grecque vega (digitale) | bump | -0.3823 ± 0.0061 | [-0.3942 ; -0.3703] | -0.3910 | +1.44 | OK |

## parite

| Cas | Méthode | Estimation ± SE | IC95 | Référence | z | Verdict |
|---|---|---|---|---|---|---|
| Parité call-put sur prix MC | GBM exact | 2.0122 ± 0.0286 | [1.9561 ; 2.0682] | 1.9801 | +1.12 | OK |
| Parité call-put sur prix MC | Heston QE | 1.9737 ± 0.0259 | [1.9230 ; 2.0245] | 1.9801 | -0.25 | OK |
| Parité call-put sur prix MC | Heston Euler | 1.9671 ± 0.0409 | [1.8870 ; 2.0472] | 1.9801 | -0.32 | OK |
| Parité call-put sur prix MC | Merton exact | 2.0152 ± 0.0330 | [1.9504 ; 2.0800] | 1.9801 | +1.06 | OK |

## couverture

| Estimateur | Couverture observée | Attendu | Écart (en sigmas binomiaux) |
|---|---|---|---|
| MC naïf | 94.3 % (943/1000) | 95 % ± 1.4 pt | -1.0 |
| antithétique | 95.2 % (952/1000) | 95 % ± 1.4 pt | +0.3 |
| variable de contrôle | 95.6 % (956/1000) | 95 % ± 1.4 pt | +0.9 |
| importance sampling (call K=180) | 95.1 % (951/1000) | 95 % ± 1.4 pt | +0.1 |
| QMC randomisé (asiatique géo., 16 x 2^6) | 95.2 % (952/1000) | 95 % ± 1.4 pt | +0.3 |
| MC naïf sur call K=180 (événement rare, contre-exemple) | 75.6 % (756/1000) | 95 % ± 1.4 pt | -28.1 |

## encadrement

| Dates d'exercice | LSM (borne basse) | Sur l'apprentissage | CRR bermudéen | Borne haute duale | Écart borne haute |
|---|---|---|---|---|---|
| 10 | 5.7205 ± 0.0097 | 5.7213 | 5.7318 | 5.7650 ± 0.0029 | +0.6 % |
| 50 | 5.7703 ± 0.0096 | 5.7968 | 5.7849 | 5.9591 ± 0.0081 | +3.0 % |

## convergence

pente MC = -0.49 ; QMC incréments = -0.69 ; QMC pont brownien = -0.85

## benchmark

numba parallèle : 2 106 920 chemins/s  soit x23 face à numpy (S_T seul) et x25 face à numpy (trajectoires complètes)

## efficacite

| Technique | Facteur de variance | Facteur d'efficacité (variance x temps) |
|---|---|---|
| Européen : antithétique | 1.7x | 1.8x |
| Européen : VC sur S_T | 5.5x | 5.0x |
| Asiatique arith. : antithétique | 1.8x | 1.8x |
| Asiatique arith. : VC géométrique | 1383.3x | 936.4x |
| Call OTM K=180 : importance sampling | 584.3x | 714.4x |

## mlmc

| Modèle | eps | niveaux | coût MLMC | coût MC standard (estimé) | rapport MLMC / standard (< 1 : MLMC moins cher) |
|---|---|---|---|---|---|
| GBM (Euler) | 0.1 | 3 | 1.12e+05 | 3.34e+04 | 3.35 |
| GBM (Euler) | 0.05 | 3 | 2.82e+05 | 2.93e+05 | 0.96 |
| GBM (Euler) | 0.02 | 4 | 2.15e+06 | 3.85e+06 | 0.56 |
| GBM (Euler) | 0.01 | 4 | 8.57e+06 | 3.05e+07 | 0.28 |
| GBM (Euler) | 0.005 | 4 | 3.42e+07 | 2.39e+08 | 0.14 |
| Heston (Euler) | 0.2 | 5 | 5.07e+05 | 7.46e+04 | 6.80 |
| Heston (Euler) | 0.1 | 6 | 2.95e+06 | 5.98e+05 | 4.94 |
| Heston (Euler) | 0.05 | 6 | 1.18e+07 | 4.85e+06 | 2.44 |
| Heston (Euler) | 0.03 | 7 | 4.66e+07 | 1.34e+07 | 3.49 |
| Heston (Euler) | 0.02 | 7 | 1.05e+08 | 6.03e+07 | 1.74 |

GBM (Euler) : exposant du coût en 1/eps = 1.96 (MLMC) contre 2.95 (MC standard), variance par niveau V_l ~ 2^(-0.88 l) ; Heston (Euler) : exposant du coût en 1/eps = 2.31 (MLMC) contre 2.84 (MC standard), variance par niveau V_l ~ 2^(-0.74 l).

## barriere

| Dates m | MC naïf | Décalage BGK | Pont brownien |
|---|---|---|---|
| 5 | +1.0271 ± 0.0307 | +0.0985 ± 0.0303 | -0.0443 ± 0.0292 |
| 10 | +0.8427 ± 0.0307 | -0.0218 ± 0.0303 | -0.0170 ± 0.0297 |
| 25 | +0.6221 ± 0.0307 | +0.0061 ± 0.0303 | +0.0112 ± 0.0299 |
| 50 | +0.4414 ± 0.0306 | -0.0178 ± 0.0303 | -0.0166 ± 0.0300 |
| 100 | +0.3311 ± 0.0305 | -0.0121 ± 0.0303 | -0.0114 ± 0.0301 |
| 250 | +0.1930 ± 0.0302 | -0.0251 ± 0.0301 | -0.0241 ± 0.0300 |