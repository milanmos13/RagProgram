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


class Contacts(list):
    """The contact given to each Wikipedia source, plus the fake network behind them."""

    network: FixtureArticleSource


@pytest.fixture
def contacts(monkeypatch):
    """Fake the network and the model; record the contact each Wikipedia source is given."""
    given = Contacts()
    network = FixtureArticleSource()

    def source(contact):
        given.append(contact)
        return network

    monkeypatch.setattr(build_command, "WikipediaArticleSource", source)
    monkeypatch.setattr(build_command, "TutorialEmbedder", lambda model_name: KeywordEmbedder())
    monkeypatch.delenv("NBA_RAG_CONTACT", raising=False)
    given.network = network
    return given


def test_builds_the_corpus_and_embeddings_into_the_output_dir(manifest, contacts, tmp_path, capsys):
    out = tmp_path / "built"

    build_command.main(["--manifest", str(manifest), "--out", str(out), "--contact", "me@x.org"])

    corpus, matrices = load_built(out)
    assert corpus.manifest_version == "test"
    assert {c.article_title for c in corpus.chunks} == {"2016 NBA Finals", "Stephen Curry"}
    assert matrices["tutorial"].shape[0] == len(corpus.chunks)
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


def test_rebuilding_reads_articles_from_the_cache_instead_of_the_network(
    manifest, contacts, tmp_path
):
    out = tmp_path / "built"
    build_command.main(["--manifest", str(manifest), "--out", str(out), "--contact", "me@x.org"])
    fetched = list(contacts.network.requests)

    build_command.main(["--manifest", str(manifest), "--out", str(out), "--contact", "me@x.org"])

    assert fetched == [("2016 NBA Finals", 222), ("Stephen Curry", 111)]
    assert contacts.network.requests == fetched


def test_the_cache_directory_can_be_chosen(manifest, contacts, tmp_path):
    cache = tmp_path / "html"

    build_command.main(
        [
            "--manifest",
            str(manifest),
            "--out",
            str(tmp_path / "built"),
            "--contact",
            "me@x.org",
            "--cache-dir",
            str(cache),
        ]
    )

    assert len(list(cache.glob("*.html"))) == 2
