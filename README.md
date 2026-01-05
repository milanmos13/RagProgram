# Advanced RAG System

A sophisticated Retrieval Augmented Generation (RAG) system built from scratch with enhanced features including multi-chunk context, source citations, file upload, and a modern Streamlit UI.

## Features

### Core RAG Capabilities
- **Semantic Search**: Uses BGE embedding models for high-quality semantic similarity
- **Multi-chunk Context**: Combines multiple relevant chunks for better LLM responses
- **Source Citations**: Shows which documents and chunks were used in responses
- **Chunk Overlap**: Prevents context loss at chunk boundaries
- **Metadata Tracking**: Tracks file names, chunk positions, and similarity scores

### UI/UX Enhancements
- **Modern Interface**: Clean, tabbed interface with better layout
- **File Upload**: Drag & drop text files directly in the browser
- **Query History**: Save and replay previous searches
- **Real-time Search**: Instant search with loading indicators
- **Responsive Design**: Works on desktop and mobile devices

### Advanced Features
- **Multiple Embedding Models**: Support for BGE and Sentence Transformers
- **Configurable Parameters**: Adjustable chunk size, top-k results, context chunks
- **OpenAI Integration**: Multiple model support (GPT-4o, GPT-4o-mini, GPT-3.5-turbo)
- **Error Handling**: Graceful error handling with helpful messages

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd RAG

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set up OpenAI (Optional)

```bash
export OPENAI_API_KEY=your_api_key_here
```

### 3. Run the Application

```bash
streamlit run app.py
```

### 4. Usage

1. **Upload Files**: Use the "Upload Files" tab to add your documents
2. **Build Index**: Go to "Search" tab and click "Build/Rebuild Index"
3. **Search**: Enter your question and click "Search"
4. **View Results**: See ranked results with source citations
5. **AI Response**: Get AI-generated answers using multiple chunks as context

## Project Structure

```
RAG/
├── app.py           # Streamlit app (advanced UI with RAG)
├── rag.py           # Core RAG functions
├── requirements.txt # Python dependencies
├── README.md        # This file
├── .gitignore       # Git ignore rules
└── sample_texts/    # Sample documents
    ├── text1.txt
    ├── text2.txt
    └── text3.txt
```

## Configuration

### Embedding Models
- `BAAI/bge-small-en-v1.5` (default) - High quality, fast
- `sentence-transformers/all-MiniLM-L6-v2` - Smaller, faster

### Parameters
- **Chunk Size**: 50-1000 tokens (default: 200)
- **Top K**: 1-10 results (default: 3)
- **Max Context Chunks**: 1-5 chunks for LLM (default: 3)

## Development

### Adding New Features

1. **New Embedding Models**: Add to the model selection in `app.py`
2. **Custom Chunking**: Modify the `chunking()` function in `rag.py`
3. **UI Components**: Add new tabs or sections in the Streamlit app
4. **LLM Providers**: Extend `generate_llm_response()` for other providers

### Core Functions

- `chunking()`: Splits documents into chunks with metadata
- `map_document_embeddings()`: Creates vector embeddings
- `retrieve_information()`: Performs semantic search
- `combine_chunks_for_context()`: Combines multiple chunks
- `generate_llm_response()`: Generates AI responses

## Performance Tips

1. **Chunk Size**: Larger chunks (300-500 tokens) for better context
2. **Model Choice**: BGE models for quality, MiniLM for speed
3. **Top K**: 3-5 results usually optimal for most queries
4. **Context Chunks**: 2-3 chunks provide good balance of context vs. noise

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Based on the article: [Building Retrieval Augmented Generation (RAG) From Scratch](https://medium.com/red-buffer/building-retrieval-augmented-generation-rag-from-scratch-74c1cd7ae2c0)
- Uses BGE embedding models from BAAI
- Built with Streamlit, Transformers, and OpenAI

