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
    retrieve_hybrid_information,
    build_bm25_index,
    generate_llm_response,
    combine_chunks_for_context,
)
from transformers import AutoTokenizer, AutoModel


st.set_page_config(page_title="Advanced RAG System", layout="wide")


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
    if "bm25_index" not in st.session_state:
        st.session_state["bm25_index"] = None
    if "bm25_mapping" not in st.session_state:
        st.session_state["bm25_mapping"] = []


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
    bm25_index, bm25_mapping = build_bm25_index(documents)
    st.session_state["documents"] = documents
    st.session_state["vectors"] = vectors
    st.session_state["tokenizer"] = tokenizer
    st.session_state["model"] = model
    st.session_state["bm25_index"] = bm25_index
    st.session_state["bm25_mapping"] = bm25_mapping


def save_query_to_history(query, results, sources=None):
    """Save query and results to history"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_entry = {
        "timestamp": timestamp,
        "query": query,
        "num_results": len(results),
        "sources": sources or []
    }
    st.session_state["query_history"].insert(0, history_entry)
    # Keep only last 10 queries
    if len(st.session_state["query_history"]) > 10:
        st.session_state["query_history"] = st.session_state["query_history"][:10]


def main():
    init_state()

    # Header
    st.title("Advanced RAG System")
    st.caption("Enhanced Retrieval Augmented Generation with multi-chunk context, source citations, and file upload")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Search", "Upload Files", "Query History"])
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.header("Settings")
            model_name = st.selectbox("Embedding Model", 
                ["BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"], 
                index=0)
            chunk_size = st.slider("Chunk Size (tokens)", 50, 1000, 200)
            top_k = st.slider("Top K Results", 1, 10, 3)
            max_context_chunks = st.slider("Max Chunks for LLM Context", 1, 5, 3)
            use_hybrid = st.checkbox("Use hybrid search (semantic + keyword)", value=True)
            if use_hybrid:
                alpha = st.slider("Semantic weight (alpha)", 0.0, 1.0, 0.7)
            
            st.subheader("OpenAI Settings")
            with_llm = st.checkbox("Enable OpenAI Response", value=True)
            if with_llm:
                openai_model = st.selectbox("OpenAI Model", 
                    ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], 
                    index=0)
                api_key = st.text_input("OpenAI API Key", 
                    value=os.getenv("OPENAI_API_KEY", ""), 
                    type="password", 
                    help="Enter your OpenAI API key")
                if api_key:
                    os.environ["OPENAI_API_KEY"] = api_key
            
            st.subheader("Data Source")
            data_source = st.radio("Choose data source:", 
                ["Use existing files", "Upload new files"], 
                index=0)
            
            if data_source == "Use existing files":
                input_dir = st.text_input("Directory path", value="sample_texts")
            else:
                st.info("Use the 'Upload Files' tab to add documents")
                input_dir = "uploaded_files"
            
            if st.button("Build/Rebuild Index", type="primary"):
                if not os.path.exists(input_dir) or not os.listdir(input_dir):
                    st.error(f"Directory '{input_dir}' is empty or doesn't exist!")
                else:
                    with st.spinner("Building index..."):
                        build_index(input_dir, model_name, chunk_size)
                    st.success("Index built successfully!")
        
        with col2:
            st.header("Search")
            query = st.text_input("Ask a question:", placeholder="e.g., What are the main causes of toddler tantrums?")
            
            col_search, col_clear = st.columns([1, 1])
            with col_search:
                search_clicked = st.button("Search", type="primary", disabled=not query)
            with col_clear:
                if st.button("Clear"):
                    st.rerun()
            
            if search_clicked and query:
                if not st.session_state["documents"] or not st.session_state["vectors"]:
                    st.warning("Please build the index first!")
                else:
                    documents = st.session_state["documents"]
                    vectors = st.session_state["vectors"]
                    tokenizer = st.session_state["tokenizer"]
                    model = st.session_state["model"]
                    bm25_index = st.session_state.get("bm25_index")
                    bm25_mapping = st.session_state.get("bm25_mapping", [])

                    with st.spinner("Searching..."):
                        if use_hybrid and bm25_index is not None and bm25_mapping:
                            results = retrieve_hybrid_information(
                                query,
                                tokenizer,
                                model,
                                vectors,
                                documents,
                                bm25_index,
                                bm25_mapping,
                                top_k=int(top_k),
                                alpha=float(alpha) if "alpha" in locals() else 0.7,
                            )
                        else:
                            results = retrieve_information(
                                query, tokenizer, model, vectors, documents, top_k=int(top_k)
                            )

                    if not results:
                        st.info("No results found.")
                    else:
                        # Display results
                        st.subheader(f"Top {len(results)} Results")
                        
                        for rank, (doc_id, chunk_id, score) in enumerate(results, start=1):
                            entry = documents[doc_id][chunk_id]
                            text = entry["text"]
                            metadata = entry.get("metadata", {})
                            file_name = metadata.get("file_name", "unknown")
                            chunk_index = metadata.get("chunk_index", 0)
                            
                            with st.expander(f"#{rank} {file_name} (Score: {score:.3f})", expanded=rank==1):
                                st.markdown(f"**Source:** {file_name} | **Chunk:** {chunk_index + 1}")
                                st.markdown(f"**Similarity Score:** {score:.4f}")
                                st.markdown("**Content:**")
                                st.write(text)
                        
                        # OpenAI Response with multi-chunk context
                        if with_llm and os.getenv("OPENAI_API_KEY"):
                            try:
                                with st.spinner("Generating AI response..."):
                                    combined_context, sources = combine_chunks_for_context(
                                        results, documents, max_context_chunks
                                    )
                                    
                                    # Create a mock entry for the combined context
                                    combined_entry = {"text": combined_context}
                                    answer = generate_llm_response(query, combined_entry, model_name=openai_model)
                                
                                st.subheader("AI Response")
                                st.markdown(answer)
                                
                                # Show sources used
                                st.subheader("Sources Used")
                                for i, source in enumerate(sources, 1):
                                    st.markdown(f"**{i}.** {source['file_name']} (Score: {source['score']:.3f})")
                                
                                # Save to history
                                save_query_to_history(query, results, sources)
                                
                            except Exception as e:
                                st.error(f"OpenAI call failed: {e}")
                        else:
                            if not os.getenv("OPENAI_API_KEY"):
                                st.info("Add your OpenAI API key to get AI-generated responses")
                            save_query_to_history(query, results)
    
    with tab2:
        st.header("Upload Files")
        st.markdown("Upload text files to add to your knowledge base")
        
        uploaded_files = st.file_uploader(
            "Choose files", 
            type=['txt'], 
            accept_multiple_files=True,
            help="Upload .txt files to add to your knowledge base"
        )
        
        if uploaded_files:
            # Create uploaded_files directory if it doesn't exist
            upload_dir = "uploaded_files"
            os.makedirs(upload_dir, exist_ok=True)
            
            for uploaded_file in uploaded_files:
                file_path = os.path.join(upload_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Uploaded: {uploaded_file.name}")
            
            st.info(f"Files saved to: {upload_dir}")
            st.markdown("Go to the 'Search' tab and select 'Upload new files' to use these documents")
    
    with tab3:
        st.header("Query History")
        
        if not st.session_state["query_history"]:
            st.info("No queries yet. Start searching to build your history!")
        else:
            for i, entry in enumerate(st.session_state["query_history"]):
                with st.expander(f"{entry['timestamp']} - {entry['query'][:50]}..."):
                    st.markdown(f"**Query:** {entry['query']}")
                    st.markdown(f"**Results:** {entry['num_results']} chunks found")
                    if entry.get('sources'):
                        st.markdown("**Sources:**")
                        for source in entry['sources']:
                            st.markdown(f"- {source['file_name']} (Score: {source['score']:.3f})")
            
            if st.button("Clear History"):
                st.session_state["query_history"] = []
                st.rerun()


if __name__ == "__main__":
    main()

