"""Fixtures communes : le volet risque réel a besoin des CSV 1h (variable UNIVERS_DIR), absents du dépôt public."""
import pytest

from mclab import risque as rk


def _donnees_presentes() -> bool:
    try:
        rk.trouver_dossier_donnees()
        return True
    except FileNotFoundError:
        return False


# à poser sur tout test qui lit les vrais prix : `@sans_donnees`
sans_donnees = pytest.mark.skipif(not _donnees_presentes(), reason="données de marché absentes (définir UNIVERS_DIR)")
