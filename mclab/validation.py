"""Contrôles de qualité des estimateurs eux-mêmes : couverture des IC95 et parité call-put."""
import time

import numpy as np

from mclab.utils import payoff_actualise, resumer


def couverture_ic95(estimateur, reference, repetitions=1000, graine=0):
    """Fraction des IC95 qui contiennent la vraie valeur, sur `repetitions` estimations indépendantes.

    estimateur(graine_entiere) -> ResultatMC. Si les erreurs-types sont bien calculées, la couverture
    vaut 95 % à la fluctuation binomiale près : ecart-type = sqrt(0,95 * 0,05 / repetitions).
    """
    graines = np.random.SeedSequence(graine).generate_state(repetitions)
    couverts = sum(estimateur(int(g)).contient(reference) for g in graines)
    couverture = couverts / repetitions
    sigma = np.sqrt(0.95 * 0.05 / repetitions)
    return {"couverture": couverture, "repetitions": repetitions, "sigma_binomial": float(sigma),
            "ecart_en_sigmas": float((couverture - 0.95) / sigma)}


def ecart_parite(chemins, S0, K, T, r, q=0.0):
    """Parité call-put sur les prix MC : C - P = S0 e^{-qT} - K e^{-rT}.

    Call et put sont évalués sur les MÊMES chemins ; la différence par chemin est e^{-rT}(S_T - K),
    donc ce test vérifie que le schéma de simulation conserve bien E[S_T] = S0 e^{(r-q)T}
    (propriété de martingale actualisée). Renvoie (ResultatMC de C - P, valeur exacte).
    """
    t0 = time.perf_counter()
    call = payoff_actualise("call_europeen", chemins, K, r, T)
    put = payoff_actualise("put_europeen", chemins, K, r, T)
    return resumer(call - put, t0), S0 * np.exp(-q * T) - K * np.exp(-r * T)
