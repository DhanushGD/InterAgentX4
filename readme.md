
# 🚀 InterAgentX 4 — Multi-Agent Interoperability with Live Workflow

**InterAgentX 4** is the sequel to [**InterAgentX**](https://github.com/DhanushGD/InterAgentX), demonstrating **agent interoperability** (Python ↔ JavaScript) and **tool interoperability** via **A2A (JSON-RPC)** and a shared **MCP** layer (arXiv + DuckDuckGo) across languages and framework.  
It combines **RAG** (LlamaIndex + ChromaDB), **Web Research**, **Evidence Verification**, and **JS-based Summarization** inside a **results-first Streamlit UI** with a **live workflow visualization**.

---

## ✨ Key Features

- **Results-first UI**: Final summary at the top, agent outputs in expanders, live Graphviz workflow.
- **3 Python Agents + 1 JavaScript Agent**:
  - 🧠 **RAG Analyst (Python, Coordinator)** — drafts from local context and delegates.
  - 🔎 **Web Researcher (Python)** — fetches external evidence via MCP (arXiv + DuckDuckGo).
  - ✅ **Evidence Verifier (Python)** — validates claims, flags contradictions/caveats.
  - 🧩 **JS Summarizer (JavaScript)** — synthesizes all prior outputs into a single structured response.
- **A2A Protocol (JSON-RPC streaming)** for agent-to-agent communication across languages/frameworks.
- **Shared MCP Tool Layer** (arXiv/DDGS) for consistent, language-agnostic tool access.
- **RAG foundation** using **LlamaIndex** over **ChromaDB** (semantic chunking + MMR retrieval).
- **Auditability**: raw orchestrator output and verified evidence bundle.
- **Agent SDKs**:
  - **Python Agents** use **CrewAI**, **A2A Python SDK**, and **MCP** integration for orchestration and tool access.
  - **JavaScript Agent** uses **@a2a-js/sdk** for A2A interoperability and **OpenAI** models for summarization.
- **Agent Cards**:
  - All agents expose their agent card for A2A communication, enabling discovery and standardized JSON-RPC streaming across Python and JavaScript runtimes.

---

## 🗺 Architecture

<img width="1431" height="554" alt="image" src="https://github.com/user-attachments/assets/dc8ed628-cb80-4aee-9ad3-fcf22b09a16a" />

## 🧩 Agent Roles & Interop

- 🧠 RAG Analyst (Python, Coordinator):
Uses LlamaIndex + ChromaDB to ground answers; delegates (query + context) to Web Researcher & Evidence Verifier via A2A.

- 🔎 Web Researcher (Python):
Queries MCP tools (arXiv, DuckDuckGo), returns web evidence → Evidence Verifier and evidence → JS Summarizer.

- ✅ Evidence Verifier (Python):
Validates claims against local context + web evidence; outputs verified facts → JS Summarizer.

- 🧩 JS Summarizer (JavaScript):
Consumes RAG draft, evidence, and verified facts to produce a single structured response (Markdown/JSON).


### Interoperability Proofs

- **Agent-to-Agent (A2A)**: Uniform JSON-RPC streaming for Python ↔ JavaScript.
- **Model Context Protocol (MCP)**: Shared tools accessible across agents/languages.
- **RAG Consistency**: LlamaIndex + ChromaDB indexing of user uploads, consistent retrieval semantics.


### Implementation Details

- **Python Agents** (RAG Analyst, Web Researcher, Evidence Verifier)  
  Each Python agent exposes its capabilities via **A2A Python SDK**, enabling JSON-RPC streaming and standardized message handling.

- **JavaScript Agent** (JS Summarizer)  
  Built using **OpenAI models** and exposes its agent card via **@a2a-js/sdk**, ensuring seamless interoperability with Python agents through the same A2A protocol.
  
## 🚀 How to Run

### 1. Start the MCP server
```bash
# Terminal 1: RAG Analyst

python mcp_server.py
```

### 2. Start Agent Servers
Make sure all agent servers are running:

```bash
# Terminal 2: RAG Analyst
python agents/rag_analyst_server.py

# Terminal 3: Web Researcher
python agents/web_researcher_server.py

# Terminal 4: Evidence Verifier
python agents/evidence_verifier_server.py

# Terminal 5: JS Summarizer
cd js-summarizer-agent && npm run dev
```

### 3. Run Streamlit App
```bash
# Terminal 6: RAG Analyst

cd InteragentSeq
streamlit run app/streamlit_app.py
```

## 🎯 How It Works

### Agent Communication Flow
1️⃣ User uploads docs → Indexed with LlamaIndex + ChromaDB 
2️⃣ User Query → RAG Analyst receives query + local context
3️⃣ RAG Analyst retrieves local context & drafts an initial answer
4️⃣ If needed, Analyst delegates to Web Researcher with Query + Local Context → MCP (arXiv/DDG) external evidence
5️⃣ Analyst sends Query + Local Context + Web Evidence to Evidence Verifier → verified facts & caveats
6️⃣ Analyst forwards draft + evidence + verified facts to JS Summarizer → Final Structured Summary

### Real-Time Display Features
- **Agent Status Indicators**: Shows which agent is currently active
- **Progress Updates**: Live updates as agents complete their tasks
- **Expandable Details**: Click to see full outputs from each agent
- **Processing Metrics**: Character counts and processing summaries

## 📊 UI Features

### 📥 Ingest & Index Tab
- Upload PDF/DOCX/TXT files or ZIP archives
- Automatic document chunking and embedding
- ChromaDB vector index creation
- Real-time progress updates

### 🔎 Query Tab
- **Multi-Agent Analysis Button**: Triggers the full agent workflow
- **Agent Communication Flow**: Toggle to show/hide real-time agent interactions
- **Progress Tracking**: Visual progress bar and status messages
- **Detailed Results**: Expandable sections for each agent's output
- **Processing Summary**: Metrics for each agent's contribution

## 🔧 Configuration

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here
MCP_HTTP_URL=http://127.0.0.1:8000/mcp

# Optional
OPENAI_CHAT_MODEL=gpt-4o
RAG_ANALYST_PORT=9101
WEB_RESEARCHER_BASE=http://localhost:9102
EVIDENCE_VERIFIER_BASE=http://localhost:9103
JS_SUMMARIZER_BASE=http://localhost:9104
```

### Sidebar Settings
- **ChromaDB Path**: Location of your vector database
- **Collection Name**: ChromaDB collection identifier
- **OpenAI Model**: GPT model for analysis
- **API Key**: OpenAI API key (password field)
- **Retrieval Settings**: Top-K, MMR diversification
- **MCP URL**: HTTP endpoint for MCP server


## 🔄 End-to-End Workflow

- Upload documents in the Streamlit UI.
- Chunk & Index with LlamaIndex → ChromaDB (MMR; semantic chunking).
- Query: RAG Analyst drafts grounded answer.
- Delegate (A2A): Web Researcher gathers external evidence via MCP; Evidence Verifier performs validation.
- Synthesize (JS): Summarizer combines all prior outputs, producing Final Structured Output.
- Display: Results-first summary + expandable agent sections + live workflow visualization.


## 🛠 Tech Stack

- **UI**: Streamlit
- **RAG**: LlamaIndex (semantic chunking, MMR), ChromaDB
- **Agents**: CrewAI  (Python), JS (Summarizer)
- **Inter-Agent**: A2A (JSON-RPC streaming)
- **Tools**: MCP server (arXiv, DuckDuckGo Search)
- **Models**: OpenAI (configurable)
- **Agent SDKs**:
  - **Python**: A2A Python SDK for agent servers.
  - **JavaScript**: @a2a-js/sdk for JS Summarizer agent.




