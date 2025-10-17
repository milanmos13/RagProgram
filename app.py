import os
import json
import streamlit as st
from typing import Dict, List
import tempfile
import shutil
from datetime import datetime

from rag import (
    chunking,
    map_document_embeddings,
    retrieve_information,
    generate_llm_response,
    combine_chunks_for_context,
)
from transformers import AutoTokenizer, AutoModel


st.set_page_config(page_title="RAG Demo", page_icon="🔎", layout="wide")


def init_state():
    if "documents" not in st.session_state:
        st.session_state["documents"] = None
    if "vectors" not in st.session_state:
        st.session_state["vectors"] = None
    if "tokenizer" not in st.session_state:
        st.session_state["tokenizer"] = None
    if "model" not in st.session_state:
        st.session_state["model"] = None
    if "query_history" not in st.session_state:
        st.session_state["query_history"] = []
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []


def build_index(input_dir: str, model_name: str, chunk_size: int):
    documents = chunking(
        directory_path=input_dir,
        model_name=model_name,
        chunk_size=chunk_size,
        para_seperator="\n\n",
        separator=" ",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    vectors = map_document_embeddings(documents, tokenizer, model)
    st.session_state["documents"] = documents
    st.session_state["vectors"] = vectors
    st.session_state["tokenizer"] = tokenizer
    st.session_state["model"] = model


def main():
    init_state()

    st.title("Retrieval Augmented Generation (RAG) Demo")
    st.caption("Builds an index from local .txt files, retrieves top chunks, and can ask OpenAI to answer using the best chunk as context.")

    with st.sidebar:
        st.header("Settings")
        input_dir = st.text_input("Input directory", value="sample_texts")
        model_name = st.text_input("Embedding model", value="BAAI/bge-small-en-v1.5")
        chunk_size = st.number_input("Chunk size (tokens)", min_value=50, max_value=2048, value=200, step=10)
        top_k = st.number_input("Top K", min_value=1, max_value=10, value=3, step=1)
        with_llm = st.checkbox("Ask OpenAI using best chunk")
        openai_model = st.text_input("OpenAI model", value="gpt-4o-mini")
        api_key = st.text_input("OPENAI_API_KEY", value=os.getenv("OPENAI_API_KEY", ""), type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        if st.button("Build / Rebuild Index"):
            with st.spinner("Building index..."):
                build_index(input_dir, model_name, chunk_size)
            st.success("Index built.")

    st.subheader("Query")
    query = st.text_input("Ask a question")
    run = st.button("Search")

    if run:
        if not st.session_state["documents"] or not st.session_state["vectors"]:
            st.warning("Please build the index first from the sidebar.")
            return

        documents = st.session_state["documents"]
        vectors = st.session_state["vectors"]
        tokenizer = st.session_state["tokenizer"]
        model = st.session_state["model"]

        with st.spinner("Retrieving..."):
            results = retrieve_information(
                query,
                tokenizer,
                model,
                vectors,
                documents,
                top_k=int(top_k),
            )

        if not results:
            st.info("No results.")
            return

        st.write("### Top results")
        for rank, (doc_id, chunk_id, score) in enumerate(results, start=1):
            entry = documents[doc_id][chunk_id]
            text = entry["text"]
            file_name = entry.get("metadata", {}).get("file_name", "unknown")
            st.markdown(f"**{rank}. {file_name} — score={score:.4f}**")
            st.write(text)
            st.divider()

        if with_llm:
            best_doc_id, best_chunk_id, _ = results[0]
            best_entry = documents[best_doc_id][best_chunk_id]
            try:
                with st.spinner("Asking OpenAI..."):
                    answer = generate_llm_response(query, best_entry)
                st.write("### OpenAI Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"OpenAI call failed: {e}")


if __name__ == "__main__":
    main()


