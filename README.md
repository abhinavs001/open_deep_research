# Open Deep Research integrated with RAG

Open Deep Research is an open source assistant that automates research and produces customizable reports on any topic. It allows you to customize the research and writing process with specific models, prompts, report structure, and search tools. This fork extends the functionality by integrating a Retrieval-Augmented Generation (RAG) pipeline with local document ingestion and real-time streaming support.

![report-generation](https://github.com/abhinavs001/open_deep_research/blob/main/Updated_workflow_image.png)

## 🚀 Quickstart

Ensure you have API keys set for your desired search tools and models.

Available search tools:

- Tavily
- Perplexity
- Exa
- ArXiv
- PubMed
- Linkup
- DuckDuckGo
- Google Search

Open Deep Research uses:

- A planner LLM to determine section structure.
- A writer LLM to generate content.

✅ Enhanced with:
- ChromaDB vector store for local document retrieval.
- Reranker using LLM to score local documents by semantic relevance.
- Combines web and local RAG using hybrid retrieval.

### Installation

Clone the repository and activate a virtual environment then:

```bash
pip install -r requirements.txt
```

Ensure API keys are set:

```bash
export TAVILY_API_KEY=...
export OPENAI_API_KEY=...
export EXA_API_KEY=...
```

### Run a query

Update `RESEARCH_TOPIC` in `src/run_query.py` and run:

```bash
python src/run_query.py
```

The pipeline will:
- Plan the report using a planner LLM.
- Search web + local documents (ChromaDB).
- Rerank relevant chunks using LLM.
- Generate sections and stream updates.
- Combine results into a final report.

## 🧠 Streaming & RAG Support [Remaining]

This fork adds:

- 📄 PDF Upload (via Streamlit interface).
- 🔍 Semantic Search over indexed local documents.
- 📶 Streaming Responses using Vercel AI SDK-compatible chunked format.

Start the Streamlit app:

```bash
streamlit run src/streamlit_app.py
```

You can:
- Upload multiple PDFs.
- Query over documents and web.
- Get streaming answers.

## 🔧 RAG Pipeline Enhancements

- Integrate OpenAI `text-embedding-3-large` for generating embeddings from ingested documents.
- Store and retrieve data using a performant vector store (e.g., Chroma).
- Implemented semantic search over locally indexed documents.
- Introduced a reranking step to improve result relevance (e.g., using similarity scoring or LLM-based ranking).

## 🧪 RAG Components

- Embedding model: OpenAI `text-embedding-3-large`
- Vector DB: ChromaDB
- Reranker: LLM-based scoring per document
- Hybrid Retrieval: Combine vector + keyword

## 📌 API Keys

Required:
- OPENAI_API_KEY
- TAVILY_API_KEY
- EXA_API_KEY
- ANTHROPIC_API_KEY
- LINKUP_API_KEY
- LANGSMITH_API_KEY

## 📖 Configuration Options

All configs (model, search, depth, etc.) passed as thread dictionary in LangGraph.

Example:
```python
thread = {
  "configurable": {
    "thread_id": str(uuid.uuid4()),
    "search_api": "exa",
    "planner_model": "gpt-4",
    "writer_model": "gpt-4",
    "max_search_depth": 2
  }
}
```

## 🧭 Workflow

1. Generate report plan from LLM
2. Search web and local PDFs
3. Use reranker to filter relevant content
4. Write sections
5. Write final sections (conclusion, summary)
6. Assemble report

## ✨ Studio UI Support

Optional: Launch with LangGraph Studio for GUI interface
```bash
langgraph dev
```

Access:
- http://127.0.0.1:2024
- https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

## 🔚 Final Output

Output is streamed live and can be saved or viewed directly in terminal or UI.

---

For more details, see original [LangGraph project](https://github.com/langchain-ai/open_deep_research).
