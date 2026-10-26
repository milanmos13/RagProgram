import json

import pytest

from fakes import FixtureArticleSource, KeywordEmbedder
from nba_rag import build as build_command
from nba_rag.store import load_built


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "version": "test",
                "articles": [
                    {"title": "2016 NBA Finals", "revision_id": 222},
                    {"title": "Stephen Curry", "revision_id": 111},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def contacts(monkeypatch):
    """Fake the network and the model; record the contact each Wikipedia source is given."""
    given: list[str] = []

    def source(contact):
        given.append(contact)
        return FixtureArticleSource()

    monkeypatch.setattr(build_command, "WikipediaArticleSource", source)
    monkeypatch.setattr(build_command, "TutorialEmbedder", lambda model_name: KeywordEmbedder())
    monkeypatch.delenv("NBA_RAG_CONTACT", raising=False)
    return given


def test_builds_the_corpus_and_embeddings_into_the_output_dir(manifest, contacts, tmp_path, capsys):
    out = tmp_path / "built"

    build_command.main(["--manifest", str(manifest), "--out", str(out), "--contact", "me@x.org"])

    corpus, matrix = load_built(out)
    assert corpus.manifest_version == "test"
    assert {c.article_title for c in corpus.chunks} == {"2016 NBA Finals", "Stephen Curry"}
    assert matrix.shape[0] == len(corpus.chunks)
    assert contacts == ["me@x.org"]
    assert f"Built {len(corpus.chunks)} Chunks from 2 Articles" in capsys.readouterr().out


def test_builds_into_the_data_dir_from_the_environment_by_default(
    manifest, contacts, tmp_path, monkeypatch
):
    monkeypatch.setenv("NBA_RAG_DATA_DIR", str(tmp_path / "from-env"))

    build_command.main(["--manifest", str(manifest), "--contact", "me@x.org"])

    load_built(tmp_path / "from-env")


def test_the_contact_can_come_from_the_environment(manifest, contacts, tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_RAG_CONTACT", "env@x.org")

    build_command.main(["--manifest", str(manifest), "--out", str(tmp_path / "built")])

    assert contacts == ["env@x.org"]


def test_a_contact_is_required_for_the_wikimedia_user_agent(manifest, contacts, tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        build_command.main(["--manifest", str(manifest), "--out", str(tmp_path / "built")])

    assert exit_info.value.code == 2
    assert "contact" in capsys.readouterr().err
    assert contacts == []
