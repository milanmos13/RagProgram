import argparse
import json
import os
import re
import uuid
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi

try:
    # Optional dependency for LLM step
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# ========== Article-aligned functions ==========

def chunking(directory_path, model_name, chunk_size, 
             para_seperator="\n\n", separator=" ", overlap_size=50):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    documents = {}
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        base = os.path.basename(file_path)
        sku = os.path.splitext(base)[0]
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            doc_id = str(uuid.uuid4())    

            # Prefer splitting on blank lines; fall back to provided separator
            paragraphs = re.split(r"\n\s*\n", text.strip()) if para_seperator == "\n\n" else re.split(para_seperator, text)

            # Collect chunks at the document level first
            doc_chunks: List[str] = []

            for paragraph in paragraphs:
                words = paragraph.split(separator)
                current_chunk_str = ""
                chunk = []
                for word in words:
                    if current_chunk_str:
                        new_chunk = current_chunk_str + separator + word
                    else:
                        new_chunk = current_chunk_str + word    
                    if len(tokenizer.tokenize(new_chunk)) <= chunk_size:
                        current_chunk_str = new_chunk
                    else:
                        if current_chunk_str:
                            chunk.append(current_chunk_str)
                        current_chunk_str = word

                if current_chunk_str:   
                    chunk.append(current_chunk_str)

                # Extend document chunks with this paragraph's chunks
                doc_chunks.extend(chunk)

            # Merge a very short title-like first chunk with the next, if present
            MIN_CHARS = 40
            if len(doc_chunks) > 1 and len(doc_chunks[0]) < MIN_CHARS:
                merged = (doc_chunks[0] + " " + doc_chunks[1]).strip()
                doc_chunks = [merged] + doc_chunks[2:]

            # Build per-document chunk map with metadata and overlap
            doc_chunk_map = {}
            for i, chunk_text in enumerate(doc_chunks):
                chunk_id = str(uuid.uuid4())
                # Add source information and chunk position
                metadata = {
                    "file_name": sku,
                    "chunk_index": i,
                    "total_chunks": len(doc_chunks),
                    "source_file": filename
                }
                doc_chunk_map[chunk_id] = {"text": chunk_text, "metadata": metadata}

            documents[doc_id] = doc_chunk_map
    return documents


def map_document_embeddings(documents, tokenizer, model):
    mapped_document_db = {}
    for id, dict_content in documents.items():
        mapped_embeddings = {}
        for content_id, text_content in dict_content.items():
            text = text_content.get("text")
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()
            mapped_embeddings[content_id] = embeddings
        mapped_document_db[id] = mapped_embeddings
    return mapped_document_db


def build_bm25_index(documents):
    """
    Build a BM25 index over all chunks for hybrid search.
    Returns the BM25Okapi instance and an index-to-(doc_id, chunk_id) mapping.
    """
    corpus_tokens: List[List[str]] = []
    index_mapping: List[Tuple[str, str]] = []

    for doc_id, chunk_map in documents.items():
        for chunk_id, data in chunk_map.items():
            text = data.get("text", "")
            tokens = text.lower().split()
            corpus_tokens.append(tokens)
            index_mapping.append((doc_id, chunk_id))

    if not corpus_tokens:
        return None, []

    bm25 = BM25Okapi(corpus_tokens)
    return bm25, index_mapping


def retrieve_information(query, tokenizer, model, mapped_document_db, documents, top_k):
    query_inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    query_embeddings = model(**query_inputs).last_hidden_state.mean(dim=1).squeeze()
    query_embeddings=query_embeddings.tolist()
    query_embeddings=np.array(query_embeddings)
    
    scores = {}
    for doc_id, chunk_dict in mapped_document_db.items():
        for chunk_id, chunk_embeddings in chunk_dict.items():
            chunk_embeddings = np.array(chunk_embeddings) 

            normalized_query = np.linalg.norm(query_embeddings)
            normalized_chunk = np.linalg.norm(chunk_embeddings)

            if normalized_chunk == 0 or normalized_query == 0:
                score = 0.0
            else:
                score = float(np.dot(chunk_embeddings, query_embeddings)/ (normalized_chunk * normalized_query))  
            
            scores[(doc_id, chunk_id )] = score   
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    # Deduplicate by exact text content while preserving order
    seen_texts = set()
    deduped: List[Tuple[str, str, float]] = []
    for (doc_id, chunk_id), score in sorted_scores:
        text = documents[doc_id][chunk_id]["text"]  # type: ignore[index]
        key = text.strip()
        if key in seen_texts:
            continue
        seen_texts.add(key)
        deduped.append((doc_id, chunk_id, score))
        if len(deduped) >= top_k:
            break
  
    return deduped


def retrieve_hybrid_information(
    query,
    tokenizer,
    model,
    mapped_document_db,
    documents,
    bm25_index,
    bm25_mapping,
    top_k,
    alpha: float = 0.7,
):
    """
    Hybrid retrieval that combines semantic similarity (vectors) with BM25 keyword scores.
    alpha controls the weight of semantic vs keyword (alpha*vector + (1-alpha)*bm25).
    """
    # Fallback to pure vector search if BM25 is not available
    if bm25_index is None or not bm25_mapping:
        return retrieve_information(query, tokenizer, model, mapped_document_db, documents, top_k)

    # Vector scores (semantic)
    query_inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    query_embeddings = model(**query_inputs).last_hidden_state.mean(dim=1).squeeze()
    query_embeddings = np.array(query_embeddings.tolist())

    vec_scores = {}
    for doc_id, chunk_dict in mapped_document_db.items():
        for chunk_id, chunk_embeddings in chunk_dict.items():
            chunk_vec = np.array(chunk_embeddings)
            norm_q = np.linalg.norm(query_embeddings)
            norm_c = np.linalg.norm(chunk_vec)
            if norm_q == 0 or norm_c == 0:
                score = 0.0
            else:
                score = float(np.dot(chunk_vec, query_embeddings) / (norm_q * norm_c))
            vec_scores[(doc_id, chunk_id)] = score

    max_vec = max(vec_scores.values()) if vec_scores else 1.0

    # BM25 scores (keyword)
    query_tokens = query.lower().split()
    bm25_raw_scores = bm25_index.get_scores(query_tokens)
    max_bm25 = max(bm25_raw_scores) if len(bm25_raw_scores) > 0 else 1.0

    bm25_scores = {}
    for idx, raw_score in enumerate(bm25_raw_scores):
        doc_id, chunk_id = bm25_mapping[idx]
        bm25_scores[(doc_id, chunk_id)] = float(raw_score)

    # Combine scores
    combined_scores = {}
    keys = set(vec_scores.keys()) | set(bm25_scores.keys())
    for key in keys:
        vec = vec_scores.get(key, 0.0)
        bm = bm25_scores.get(key, 0.0)
        vec_norm = vec / max_vec if max_vec > 0 else 0.0
        bm_norm = bm / max_bm25 if max_bm25 > 0 else 0.0
        combined = alpha * vec_norm + (1.0 - alpha) * bm_norm
        combined_scores[key] = combined

    sorted_scores = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)

    # Deduplicate by exact text content while preserving order
    seen_texts = set()
    deduped: List[Tuple[str, str, float]] = []
    for (doc_id, chunk_id), score in sorted_scores:
        text = documents[doc_id][chunk_id]["text"]
        key = text.strip()
        if key in seen_texts:
            continue
        seen_texts.add(key)
        deduped.append((doc_id, chunk_id, score))
        if len(deduped) >= top_k:
            break

    return deduped


def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def combine_chunks_for_context(results, documents, max_chunks=3):
    """Combine multiple top chunks into a single context for LLM"""
    combined_context = ""
    sources = []
    
    for i, (doc_id, chunk_id, score) in enumerate(results[:max_chunks]):
        chunk_data = documents[doc_id][chunk_id]
        text = chunk_data["text"]
        metadata = chunk_data.get("metadata", {})
        
        # Add chunk with source info
        source_info = f"[Source {i+1}: {metadata.get('file_name', 'unknown')}]"
        combined_context += f"{source_info}\n{text}\n\n"
        sources.append({
            "file_name": metadata.get("file_name", "unknown"),
            "chunk_index": metadata.get("chunk_index", 0),
            "score": score
        })
    
    return combined_context.strip(), sources


def generate_llm_response(query, relavent_text, *, model_name="gpt-4o-mini"):
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Please `pip install openai`. ")

    client = OpenAI()
    context = relavent_text.get("text") if isinstance(relavent_text, dict) else str(relavent_text)
    prompt = (
        "You are an intelligent search engine. You will be provided with some retrieved "
        "context, as well as the users query.\n\n"
        "Your job is to understand the request, and answer based on the retrieved context.\n"
        "Here is context:\n\n<context>\n" + context + "\n</context>\n\n" 
        "Question: " + query
    )

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Answer using only the provided context when possible."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content


# ========== CLI to run the article-like pipeline ==========

def main():
    parser = argparse.ArgumentParser(description="Article-aligned RAG pipeline")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--model_name", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--chunk_size", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--query", default=None)
    parser.add_argument("--doc_store", default="doc_store.json")
    parser.add_argument("--vec_store", default="vector_store.json")
    parser.add_argument("--with_llm", action="store_true", help="Generate an LLM answer using OpenAI")
    parser.add_argument("--openai_model", default="gpt-4o-mini")
    args = parser.parse_args()

    documents = chunking(
        directory_path=args.input_dir,
        model_name=args.model_name,
        chunk_size=args.chunk_size,
        para_seperator="\n\n",
        separator=" ",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    mapped_document_db = map_document_embeddings(documents, tokenizer, model)

    save_json(args.doc_store, documents) 
    save_json(args.vec_store, mapped_document_db)

    if args.query:
        top_results = retrieve_information(args.query, tokenizer, model, mapped_document_db, documents, args.top_k)
        if not top_results:
            print("No results")
            return
        print("Top results:\n")
        for rank, (doc_id, chunk_id, score) in enumerate(top_results, start=1):
            related_text = documents[doc_id][chunk_id]
            text = related_text["text"] if isinstance(related_text, dict) else str(related_text)
            meta = related_text.get("metadata", {}) if isinstance(related_text, dict) else {}
            file_name = meta.get("file_name", "unknown")
            snippet = text[:160].replace("\n", " ") + ("..." if len(text) > 160 else "")
            print(f"{rank}. {file_name} | score={score:.4f}\n{snippet}\n")

        if args.with_llm:
            # Use the best chunk as context for LLM
            best_doc_id, best_chunk_id, _ = top_results[0]
            best_related = documents[best_doc_id][best_chunk_id]
            try:
                answer = generate_llm_response(args.query, best_related, model_name=args.openai_model)
                print("LLM answer:\n")
                print(answer)
            except Exception as e:
                print(f"LLM step failed: {e}")
    else:
        print(
            f"Processed {len(documents)} documents; total chunks: "
            f"{sum(len(v) for v in documents.values())}"
        )


if __name__ == "__main__":
    main()


