from pathlib import Path

from qc_copilot.config import PROJECT_ROOT, Settings


def test_blank_environment_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "")
    monkeypatch.setenv("QDRANT_PATH", "")
    settings = Settings(_env_file=None)
    assert settings.data_dir == PROJECT_ROOT / "data"
    assert settings.retrieval_top_k == 5
    assert settings.qdrant_path == ""


def test_data_dir_override_moves_policy_and_catalog_paths(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "/srv/qc/data")
    settings = Settings(_env_file=None)
    assert settings.policies_dir == Path("/srv/qc/data/policies")
    assert settings.catalog_dir == Path("/srv/qc/data/catalog")


def test_embedded_qdrant_path_is_opt_in(monkeypatch):
    monkeypatch.setenv("QDRANT_PATH", "/srv/qc/qdrant")
    assert Settings(_env_file=None).qdrant_path == "/srv/qc/qdrant"
