import numpy as np
import pytest

from fakes import KeywordEmbedder
from nba_rag.retrieval import build_baseline, retrieve
from nba_rag.store import load_built, save_built


def test_built_corpus_and_embeddings_round_trip_without_re_embedding(corpus, tmp_path):
    built = build_baseline(corpus, KeywordEmbedder())
    save_built(tmp_path, corpus, {built.embedding_setup: built.matrix})

    loaded_corpus, loaded_matrices = load_built(tmp_path)
    embedder = KeywordEmbedder()
    reloaded = build_baseline(loaded_corpus, embedder, loaded_matrices[built.embedding_setup])

    assert loaded_corpus == corpus
    assert embedder.chunks_embedded == 0
    query = "Was Curry the unanimous MVP?"
    assert retrieve(reloaded, query, k=4) == retrieve(built, query, k=4)


def test_each_embedding_setup_keeps_its_own_matrix(corpus, tmp_path):
    first = np.arange(6, dtype=float).reshape(3, 2)
    second = np.ones((3, 4))

    save_built(tmp_path, corpus, {"tutorial": first, "instructed": second})

    _, matrices = load_built(tmp_path)
    assert sorted(matrices) == ["instructed", "tutorial"]
    np.testing.assert_array_equal(matrices["tutorial"], first)
    np.testing.assert_array_equal(matrices["instructed"], second)


def test_loading_a_directory_with_no_embeddings_raises_file_not_found(corpus, tmp_path):
    save_built(tmp_path, corpus, {})

    with pytest.raises(FileNotFoundError):
        load_built(tmp_path)
