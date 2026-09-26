#Dynamic GitHub Issue RAG Assistant

A Retrieval-Augmented Generation (RAG) system that answers natural-language questions about GitHub issues. It syncs issues from a GitHub repository into a vector database, routes each question to the best retrieval strategy using an LLM classifier, and generates grounded answers using Gemini.

Built on LangGraph, ChromaDB, Sentence-Transformers, and the Gemini API.

How it works
On startup, the app syncs GitHub issues into a local ChromaDB vector store, embedding only new or updated issues.
When you ask a question, an LLM router classifies it into one of four retrieval strategies.
The matching retrieval node fetches relevant issues (from the vector store, the GitHub API, or both).
The retrieved issues are formatted into context and passed to Gemini, which generates an answer grounded only in that context.
Architecture
MAIN.py
  └── SYNC.py                 (one-time: fetch + embed issues into ChromaDB)
  └── LANGRAPH_ROUTER.py       (per question: runs the LangGraph pipeline)
        ├── LLM_LOOKUP.py           → classifies question into a route
        ├── RETREIVAL.py            → runs the chosen search function
        ├── CONTEXT_CONNECTOR.py    → formats results into context text
        └── GENERATION.py           → calls Gemini for the final answer
File	Responsibility
MAIN.py	Entry point — runs the sync step, then starts the interactive Q&A loop
SYNC.py	Fetches issues from the GitHub API, diffs against ChromaDB by updated_at, embeds and upserts new/changed issues
LANGRAPH_ROUTER.py	Defines the LangGraph StateGraph, wires together the router, retrieval, and generation nodes
LLM_LOOKUP.py	Uses Gemini to classify a question into exact / semantic / latest / hybrid, extracting issue number / state if present
RETREIVAL.py	Implements the four retrieval functions: exact lookup, semantic search, latest issues (GitHub API), and hybrid search
CONTEXT_CONNECTOR.py	Formats retrieved issues (from ChromaDB or the GitHub API) into a plain-text context block
GENERATION.py	Sends the question + context to Gemini and returns a grounded answer
Retrieval strategies
Route	Trigger	Data source
exact	User asks about a specific issue number	ChromaDB (exact match on number)
semantic	User asks a conceptual/topic-based question	ChromaDB (vector similarity search)
latest	User wants the newest/recent issues	GitHub API (sorted by updated_at)
hybrid	User asks about a topic AND wants recent issues	GitHub API + on-the-fly embedding + cosine similarity ranking
Setup
Prerequisites
Python 3.10+
A Gemini API key
Installation
bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
Environment variables

Create a .env file in the project root:

GEMINI_API_KEY=your_api_key_here
Dependencies
requests
chromadb
numpy
sentence-transformers
python-dotenv
google-genai
langgraph

ChromaDB requires an existing collection named github_issues at ./chroma_db. Run the sync step first (handled automatically by MAIN.py) to create and populate it.

Usage

Run the app:

bash
python MAIN.py

On startup, it will:

Fetch up to 500 issues from the configured repo (pallets/flask by default — change OWNER/REPO in RETREIVAL.py and SYNC.py to point at a different repo).
Embed and store new/updated issues in ChromaDB.
Prompt you to ask questions in a loop.

Example session:

Ask a question (type 'exit' to quit): What is issue #6139?
Ask a question (type 'exit' to quit): What issues discuss request contexts?
Ask a question (type 'exit' to quit): What are the latest security issues?

Type exit to quit.

Project structure
.
├── MAIN.py
├── SYNC.py
├── LANGRAPH_ROUTER.py
├── LLM_LOOKUP.py
├── RETREIVAL.py
├── CONTEXT_CONNECTOR.py
├── GENERATION.py
├── chroma_db/              # persistent vector store (generated)
├── issues.json             # raw GitHub issues (generated)
├── processed_issues.json   # cleaned issues (generated)
├── .env                    # API keys (not committed)
└── requirements.txt
Notes / known limitations
The Gemini model name used (gemini-3.5-flash-lite) should be verified against the current list of available models before deployment.
latest_issues and hybrid_search call the GitHub API live and are subject to GitHub's unauthenticated rate limits; add a GitHub token to HEADERS in SYNC.py / RETREIVAL.py for higher limits.
The sync step is capped at 500 issues (TARGET_ISSUES in SYNC.py); increase this to index a larger repo.
