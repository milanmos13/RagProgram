import json

import pytest

from fakes import FakeCatalogue
from nba_rag import generate_manifest as command
from nba_rag.corpus import load_manifest


@pytest.fixture
def catalogue(monkeypatch):
    fake = FakeCatalogue(["Boston Celtics"], ["Stephen Curry"])
    contacts: list[str] = []

    def make(contact):
        contacts.append(contact)
        return fake

    monkeypatch.setattr(command, "WikipediaCatalogue", make)
    monkeypatch.delenv("NBA_RAG_CONTACT", raising=False)
    fake.contacts = contacts
    return fake


def test_writes_the_manifest_the_build_command_reads(catalogue, tmp_path, capsys):
    out = tmp_path / "manifest.json"

    command.main(
        ["--version", "full-1", "--last-season", "2025", "--out", str(out), "--contact", "me@x.org"]
    )

    manifest = load_manifest(out)
    assert manifest.version == "full-1"
    assert len(manifest.articles) == 80 + 80 + 1 + 1
    assert manifest.articles[0].title == "1946–47 BAA season"
    assert catalogue.contacts == ["me@x.org"]
    assert "Wrote 162 Articles" in capsys.readouterr().out


def test_regenerating_over_an_existing_manifest_keeps_its_pins(catalogue, tmp_path):
    out = tmp_path / "manifest.json"
    out.write_text(
        json.dumps(
            {"version": "old", "articles": [{"title": "Stephen Curry", "revision_id": 111}]}
        ),
        encoding="utf-8",
    )

    command.main(
        ["--version", "full-1", "--last-season", "2025", "--out", str(out), "--contact", "me@x.org"]
    )

    pinned = {a.title: a.revision_id for a in load_manifest(out).articles}
    assert pinned["Stephen Curry"] == 111
    assert "Stephen Curry" not in catalogue.revisions_requested


def test_a_contact_is_required(catalogue, tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        command.main(["--version", "v", "--last-season", "2025", "--out", str(tmp_path / "m.json")])

    assert exit_info.value.code == 2
    assert "contact" in capsys.readouterr().err
