from fakes import KeywordEmbedder
from nba_rag.retrieval import build_baseline, retrieve
from nba_rag.store import load_built, save_built


def test_built_corpus_and_embeddings_round_trip_without_re_embedding(corpus, tmp_path):
    built = build_baseline(corpus, KeywordEmbedder())
    save_built(tmp_path, corpus, built.matrix)

    loaded_corpus, loaded_matrix = load_built(tmp_path)
    embedder = KeywordEmbedder()
    reloaded = build_baseline(loaded_corpus, embedder, loaded_matrix)

    assert loaded_corpus == corpus
    assert embedder.chunks_embedded == 0
    query = "Was Curry the unanimous MVP?"
    assert retrieve(reloaded, query, k=4) == retrieve(built, query, k=4)
