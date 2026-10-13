"""results/approfondissement.json : un bloc complet par question, conclusions chiffrées, README et figures présents."""
import json
from pathlib import Path

import pytest

from mclab import experiences

RACINE = Path(__file__).resolve().parents[1]
JSON = RACINE / "results" / "approfondissement.json"


def test_conversion_json_native():
    out = experiences._l({"a": [1.5, float("nan")], "b": {"k": 3}})
    assert json.dumps(out) and out["a"][1] is None


@pytest.mark.skipif(not JSON.exists(), reason="results/approfondissement.json non généré (scripts/run_approfondissement.py)")
def test_json_complet_et_coherent_avec_le_readme():
    d = json.loads(JSON.read_text(encoding="utf-8"))
    assert list(d["blocs"]) == list(experiences.BLOCS)
    for cle, b in d["blocs"].items():
        assert {"question", "theoreme", "attendu", "chiffres", "conclusion", "figure", "duree_s"} <= set(b), cle
        assert any(ch.isdigit() for ch in b["conclusion"]) and len(b["conclusion"]) > 200
        assert (RACINE / b["figure"]).is_file(), b["figure"]
    readme = (RACINE / "README.md").read_text(encoding="utf-8")
    section = readme.split("## Approfondissement", 1)[1].split("\n## ", 1)[0]
    assert section.count("\n### Q") == 8
    for cle, b in d["blocs"].items():
        assert b["question"] in section and b["figure"] in section
