# Cocktails RAG MCP Server

MCP server for looking up cocktails and suggesting recipes from a local dataset.
It retrieves recipes with FAISS, reranks the matches with a Hugging Face model,
and sends the selected context to Groq to generate an answer.

## Requirements

- **Python 3.11+**
- **uv package manager** - https://docs.astral.sh/uv/getting-started/installation/

## Quick Start

1. **Get a Groq API key**: https://console.groq.com/keys

2. **Setup:**

   ```bash
   # Clone the repository
   git clone https://github.com/00200200/cocktails-rag-mcp.git
   cd cocktails-rag-mcp

   # Copy environment template
   cp .env.example .env

   # Edit .env and add your GROQ_API_KEY
   nano .env

   # Install dependencies
   uv sync
   ```

3. **Initialize the pipeline:**

   This loads the embedding and reranker models and creates or loads the FAISS
   index. Model weights are downloaded on first use; the Groq API key must be set.

   ```bash
   uv run python -c "from src.rag.rag import RAG; RAG(); print('Pipeline initialized')"
   ```

4. **Install for Claude Desktop:**

   ### Automatic (Recommended)

   ```bash
   uv run fastmcp install claude-desktop fastmcp.json --name cocktails --env-file .env
   ```

   ### Manual

   Edit config file:

   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

   ```json
   {
   	"mcpServers": {
   		"cocktails": {
   			"command": "uv",
   			"args": [
   				"run",
   				"--with","faiss-cpu",
   				"--with","fastmcp",
   				"--with","jq",
   				"--with","langchain",
   				"--with","langchain-community",
   				"--with","langchain-groq",
   				"--with","langchain-huggingface",
   				"--with","pandas",
   				"--with","python-dotenv",
   				"--with","sentence-transformers",
   				"fastmcp",
   				"run",
   				"/ABSOLUTE/PATH/TO/src/mcp/server.py:mcp"
   			],
   			"env": {
   				"GROQ_API_KEY": "your_groq_api_key_here"
   			}
   		}
   	}
   }
   ```

   Replace `/ABSOLUTE/PATH/TO/` with your project path and `GROQ_API_KEY` with your API key.

## Example Usage

<p align="center">
  <img src="assets/claude_example_1.png" width="600">
</p>

<p align="center">
  <img src="assets/claude_example_2.png" width="600">
</p>

## Local Testing

```bash
# Test RAG pipeline directly
uv run python -m src.rag.rag

# Test MCP server locally
uv run python src/mcp/server.py
```

## Development

### Code Formatting

```bash
# Format code with black
uv tool run black .

# Sort imports with isort
uv tool run isort .
```

## Project Structure

```
RAG/
├── src/
│   ├── mcp/         # MCP server implementation (FastMCP)
│   ├── rag/         # RAG pipeline (retrieve, rerank, generate)
│   ├── db/          # FAISS vector database handler
│   └── data/        # Data loading utilities
├── data/            # Cocktail dataset
├── faiss_index/     # Generated FAISS index (auto-created on first run)
├── notebooks/       # EDA notebook
├── fastmcp.json     # FastMCP configuration
├── pyproject.toml   # Project dependencies
└── .env.example     # Environment template
```

## Tech Stack

- **MCP Framework:** FastMCP
- **RAG Framework:** LangChain
- **Embeddings:** BAAI/bge-m3 (local via HuggingFace)
- **Vector DB:** FAISS (local)
- **Reranker:** BAAI/bge-reranker-v2-m3 (local via HuggingFace)
- **LLM:** Groq API (llama-3.1-8b-instant)
- **Package Manager:** uv
