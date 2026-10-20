import pytest

from fakes import FixtureArticleSource, count_words
from nba_rag.corpus import Corpus, Manifest, ManifestEntry, build_corpus


@pytest.fixture
def corpus() -> Corpus:
    """The 2016 NBA Finals and Stephen Curry fixtures, chunked at the default size."""
    manifest = Manifest(
        version="test",
        articles=[
            ManifestEntry(title="2016 NBA Finals", revision_id=222),
            ManifestEntry(title="Stephen Curry", revision_id=111),
        ],
    )
    return build_corpus(manifest, source=FixtureArticleSource(), count_tokens=count_words)
