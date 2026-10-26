from pathlib import Path

from nba_rag.settings import data_dir, results_dir


def test_directories_default_to_data_and_results(monkeypatch):
    monkeypatch.delenv("NBA_RAG_DATA_DIR", raising=False)
    monkeypatch.delenv("NBA_RAG_RESULTS_DIR", raising=False)

    assert data_dir() == Path("data")
    assert results_dir() == Path("results")


def test_directories_come_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("NBA_RAG_DATA_DIR", str(tmp_path / "built"))
    monkeypatch.setenv("NBA_RAG_RESULTS_DIR", str(tmp_path / "scores"))

    assert data_dir() == tmp_path / "built"
    assert results_dir() == tmp_path / "scores"
