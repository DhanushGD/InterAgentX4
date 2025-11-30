
# app/streamlit_app.py (only the changed block)
import os
import streamlit as st
from uuid import uuid4
import asyncio
import httpx
import os
import time
import tempfile
import requests
import streamlit as st
from a2a.types import AgentCard, AgentCapabilities, SendStreamingMessageRequest, MessageSendParams
from new_rag_utils import (
    build_llamaindex_settings,
    save_uploaded_zip_to_dir,
    save_uploaded_files_to_dir,
    load_documents_from_dir,
    chunk_documents,
    create_or_update_index,
    load_index_from_chroma,
    retrieve_context,
    DEFAULT_DB_DIR,
    DEFAULT_COLLECTION,
    ensure_dir,
)
from a2a.client import A2ACardResolver, A2AClient
from dotenv import load_dotenv
from a2a.types import SendMessageRequest, MessageSendParams

# from agents.common_config import RAG_ANALYST_BASE 
RAG_ANALYST_PORT = int(os.environ.get("RAG_ANALYST_PORT", "9101"))
RAG_ANALYST_BASE = f"http://localhost:{RAG_ANALYST_PORT}"

st.set_page_config(page_title="InterAgentX 4 — RAG with MCP + A2A", layout="wide", page_icon="🧠")

st.title("🧠 InterAgentX — Multi-Agent Analysis with Real-Time Communication")
st.caption("🤖 Multi-agent collaboration: RAG Analyst → Web Researcher → Evidence Verifier → JS Summarizer with real-time communication flow")
st.sidebar.header("⚙️ Settings")

db_dir = st.sidebar.text_input("ChromaDB path", value=DEFAULT_DB_DIR)
collection_name = st.sidebar.text_input("Collection name", value=DEFAULT_COLLECTION)

openai_model = st.sidebar.text_input("OpenAI chat model", value=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"))
load_dotenv()
openai_api_key = st.sidebar.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))

top_k = st.sidebar.slider("Top-K retrieval", min_value=1, max_value=10, value=5, step=1)
use_mmr = st.sidebar.checkbox("Use MMR diversification", value=False)
mmr_diversity = st.sidebar.slider("MMR diversity (0–1)", 0.0, 1.0, 0.3, 0.05)

mcp_http_url = st.sidebar.text_input("MCP (HTTP streamable) URL", value=os.getenv("MCP_HTTP_URL", "http://127.0.0.1:8000/mcp"))
st.sidebar.caption("CrewAI connects to MCP via **streamable HTTP** for remote servers (recommended).")  


# Ensure local db dir and llamaindex settings are initialized
ensure_dir(db_dir)
build_llamaindex_settings()

tab_ingest, tab_query = st.tabs(["📥 Ingest & Index", "🔎 Query"])

with tab_ingest:
    st.subheader("Upload a directory (ZIP) or multiple files")
    st.write(
        "Upload a ZIP containing PDFs/DOCX/TXT (recommended), or upload multiple files directly. "
        "You can also enter a local folder path if running Streamlit locally."
    )

    zip_file = st.file_uploader("Upload ZIP folder", type=["zip"])
    files = st.file_uploader("Upload multiple files", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    local_dir = st.text_input("Local directory path (optional)")

    do_index = st.button("🚀 Build / Update Index")

    if do_index:
        with st.spinner("Processing ingestion..."):
            work_dir = tempfile.mkdtemp(prefix="interagentx_")
            ingestion_dir = work_dir

            if zip_file is not None:
                st.info("Extracting ZIP...")
                ingestion_dir = save_uploaded_zip_to_dir(zip_file.getvalue(), ingestion_dir)
            elif files:
                st.info("Saving uploaded files...")
                save_uploaded_files_to_dir([(f.name, f.getvalue()) for f in files], ingestion_dir)
            elif local_dir and os.path.isdir(local_dir):
                st.info("Using local directory path...")
                ingestion_dir = local_dir
            else:
                st.error("Please upload a ZIP, multiple files, or provide a valid local directory.")
                st.stop()

            st.info("📄 Loading documents...")
            documents = load_documents_from_dir(ingestion_dir)
            st.write(f"Loaded **{len(documents)}** documents")

            if len(documents) == 0:
                st.warning("No documents found. Check file extensions or directory path.")
                st.stop()

            st.info("🧩 Chunking...")
            nodes = chunk_documents(documents)
            st.write(f"Created **{len(nodes)}** chunks")

            st.info("🔍 Creating/updating Chroma vector index...")
            index = create_or_update_index(nodes, db_dir=db_dir, collection_name=collection_name)
            # st.success(f"Index ready! Stored in `{db_dir}` under collection `{collection_name}`.")
            st.success(f"Index ready! Stored under collection `{collection_name}`.")


            st.session_state["index_ready"] = True
            st.session_state["index_meta"] = {
                "db_dir": db_dir,
                "collection_name": collection_name,
                "num_docs": len(documents),
                "num_nodes": len(nodes),
            }


import html

HEADINGS = {
    "block": "🧩 Multi-Agent Analysis Complete",
    "rag": "🧠 RAG ANALYST DRAFT:",
    "web": "🔎 WEB RESEARCHER FINDINGS:",
    "verify": "✅ EVIDENCE VERIFIER ANALYSIS:",
    "summary": "🧩 FINAL COMPREHENSIVE SUMMARY:",
}

def parse_agent_outputs(outputs: dict):
    """
    Parse multi-agent text sections from `outputs` that contain:
    - final_answer: string
    - raw_lines: list[str] (each line can be part of the block)

    Returns:
      agent_outputs: dict with rag_draft, web_researcher, evidence_verifier, final_summary
      final_answer: str (for UI display; prefers final_summary when present)
    """
    raw_lines = outputs.get('raw_lines') or []
    final_answer = (outputs.get('final_answer') or '').strip()

    # 1) Join raw_lines into a single text block and unescape HTML entities
    text_block = ""
    if raw_lines and any(isinstance(x, str) for x in raw_lines):
        joined = "\n".join([str(x) for x in raw_lines if x is not None])
        joined = html.unescape(joined)

        # Prefer slicing from the block marker if present (so we ignore any noise before it)
        start_idx = joined.find(HEADINGS["block"])
        text_block = joined[start_idx:] if start_idx != -1 else joined
    else:
        # Fall back to final_answer if raw_lines are empty or invalid
        text_block = html.unescape(final_answer)

    # Prepare return structure
    agent_outputs = {
        'rag_draft': '',
        'web_researcher': '',
        'evidence_verifier': '',
        'final_summary': '',
    }

    # Helper to extract text between two markers
    def extract_between(text: str, start_marker: str, end_marker: str | None):
        if not text or start_marker not in text:
            return ""
        start = text.find(start_marker) + len(start_marker)
        if end_marker and end_marker in text[start:]:
            end = text.find(end_marker, start)
            return text[start:end].strip()
        return text[start:].strip()

    # 2) Extract each section using clear boundaries
    agent_outputs['rag_draft'] = extract_between(text_block, HEADINGS['rag'], HEADINGS['web'])
    agent_outputs['web_researcher'] = extract_between(text_block, HEADINGS['web'], HEADINGS['verify'])
    agent_outputs['evidence_verifier'] = extract_between(text_block, HEADINGS['verify'], HEADINGS['summary'])
    agent_outputs['final_summary'] = extract_between(text_block, HEADINGS['summary'], None)

    # 3) Sensible fallback: if summary missing but final_answer exists, use final_answer
    if not agent_outputs['final_summary'] and final_answer:
        agent_outputs['final_summary'] = html.unescape(final_answer)

    # Prefer showing final_summary in UI
    final_for_ui = agent_outputs['final_summary'] or final_answer

    # Debug prints (remove/replace with logger if you prefer)
    print(f"[STREAMLIT] final_answer (raw): {final_answer[:200]}")
    print(f"[STREAMLIT] raw_lines length: {len(raw_lines)}")
    if raw_lines:
        print(f"[STREAMLIT] First raw_line: {raw_lines[0] if isinstance(raw_lines[0], str) else type(raw_lines[0])}")
    print(f"[STREAMLIT] Text block length: {len(text_block)}")
    print("[STREAMLIT] Agent outputs parsed:")
    for k, v in agent_outputs.items():
        print(f"  - {k}: {len(v)} chars")

    return agent_outputs, final_for_ui





with tab_query:
    st.subheader("🤖 Multi-Agent Analysis with Real-Time Communication")

    query = st.text_input(
        "Enter your question here",
        placeholder="What is supervised learning vs reinforcement learning?"
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        run_multi_agent = st.button("🚀 Run Multi-Agent Analysis", type="primary", use_container_width=True)
    with col2:
        show_workflow = st.checkbox("Show Live Agent Workflow (visual)", value=True)

    if run_multi_agent:
        if not query.strip():
            st.error("Please enter a question.")
            st.stop()

        if not openai_api_key:
            st.error("Please set OPENAI_API_KEY in the sidebar.")
            st.stop()

        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["MCP_HTTP_URL"] = mcp_http_url

        # Retrieve context
        with st.spinner("🔎 Retrieving relevant context from ChromaDB..."):
            idx = load_index_from_chroma(db_dir=db_dir, collection_name=collection_name)
            context, nodes, sources = retrieve_context(
                idx, query, top_k=top_k, use_mmr=use_mmr, mmr_diversity=mmr_diversity
            )
            st.success(f"Retrieved {len(nodes)} relevant chunks from {len(sources)} sources")

        # Progress & status placeholders
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("🏁 Starting Multi-Agent Analysis...")
        progress_bar.progress(10)

        # --- Results-first section placeholders ---
        st.markdown("---")
        st.markdown("## 🤖 Multi-Agent Analysis Results")
        st.subheader("🧩 Final Comprehensive Summary")
        final_summary_ph = st.empty()  # filled neatly at the end

        # Expanders: persistent placeholders for live updates
        rag_expander_ph = st.empty()
        web_expander_ph = st.empty()
        verify_expander_ph = st.empty()
        js_expander_ph = st.empty()

        # Initial render (awaiting content)
        with rag_expander_ph.expander("🧠 RAG Analyst Draft", expanded=False):
            st.markdown("— awaiting completion —")
        with web_expander_ph.expander("🔎 Web Researcher Findings", expanded=False):
            st.markdown("— awaiting completion —")
        with verify_expander_ph.expander("✅ Evidence Verifier Analysis", expanded=False):
            st.markdown("— awaiting completion —")
        with js_expander_ph.expander("🧩 JS Summarizer (Complete Final Summary)", expanded=False):
            st.markdown("— awaiting completion —")

        # Optional: raw orchestrator output for audit
        raw_final_answer_ph = st.empty()
        with raw_final_answer_ph.expander("📄 Raw Final Answer (as returned by orchestrator)", expanded=False):
            st.code("—")

        # --- Live Agent Workflow (Graphviz) ---
        workflow_graph_ph = st.empty()
        if show_workflow:
            st.markdown("---")
            st.markdown("### 🔁 Agent Workflow (live)")

        # Minimal agent state
        agent_status = {
            "rag_analyst": {"status": "pending", "output": "", "duration": 0},
            "web_researcher": {"status": "pending", "output": "", "duration": 0},
            "evidence_verifier": {"status": "pending", "output": "", "duration": 0},
            "js_summarizer": {"status": "pending", "output": "", "duration": 0},
        }

        # Color map (include error red)
        def build_workflow_dot(status_map: dict) -> str:
            color_for = {
                "pending":   "#9aa0a6",   # gray
                "running":   "#1a73e8",   # blue
                "completed": "#34a853",   # green
                "error":     "#ea4335",   # red
            }
            def node(key: str, label: str):
                stt = status_map.get(key, {}).get("status", "pending")
                color = color_for.get(stt, "#9aa0a6")
                return f'"{label}" [shape=box, style="rounded,filled", fillcolor="{color}", color="{color}", fontcolor="white"];'

            rag_node = node("rag_analyst", "🧠 RAG Analyst (Coordinator)")
            web_node = node("web_researcher", "🔎 Web Researcher")
            ev_node  = node("evidence_verifier", "✅ Evidence Verifier")
            js_node  = node("js_summarizer", "🧩 JS Summarizer")
            final_node = '"📦 Final Structured Output" [shape=box, style="rounded,filled", fillcolor="#202124", color="#202124", fontcolor="white"];'

            edges = [
                '"🧠 RAG Analyst (Coordinator)" -> "🔎 Web Researcher" [label="delegate: query + context"];',
                '"🧠 RAG Analyst (Coordinator)" -> "✅ Evidence Verifier" [label="delegate: query + context"];',
                '"🔎 Web Researcher" -> "✅ Evidence Verifier" [label="handoff: web evidence"];',
                '"🔎 Web Researcher" -> "🧩 JS Summarizer" [label="evidence"];',
                '"✅ Evidence Verifier" -> "🧩 JS Summarizer" [label="verified facts"];',
                '"🧩 JS Summarizer" -> "📦 Final Structured Output" [label="synthesize"];',
            ]

            dot = f'''
            digraph G {{
                rankdir=LR;
                bgcolor="#0e1117";
                graph [fontname="Helvetica"];
                node  [fontname="Helvetica"];
                edge  [fontname="Helvetica", color="#8ab4f8", fontcolor="#8ab4f8"];

                // Legend
                subgraph cluster_legend {{
                    label="Legend";
                    color="#1f1f1f";
                    "Pending"   [shape=box, style=filled, fillcolor="#9aa0a6", fontcolor="white"];
                    "Running"   [shape=box, style=filled, fillcolor="#1a73e8", fontcolor="white"];
                    "Completed" [shape=box, style=filled, fillcolor="#34a853", fontcolor="white"];
                    "Error"     [shape=box, style=filled, fillcolor="#ea4335", fontcolor="white"];
                }}

                // Nodes
                {rag_node}
                {web_node}
                {ev_node}
                {js_node}
                {final_node}

                // Edges
                {' '.join(edges)}
            }}
            '''
            return dot

        # Initial workflow render
        if show_workflow:
            workflow_graph_ph.graphviz_chart(build_workflow_dot(agent_status))

        try:
            import sys, os, asyncio, html, time
            sys.path.append(os.path.dirname(__file__) if '__file__' in globals() else os.getcwd())
            from run_a2a_terminal import run_rag_stream

            start_time = time.time()
            # last_graph_update = 0.0  # throttle graph re-renders
            last_graph_update = [0]  # throttle graph re-renders (mutable container)


            def render_agent_expander(agent_key: str, title: str, ph):
                out = (agent_status.get(agent_key, {}).get("output", "") or "").strip()
                with ph.expander(title, expanded=False):
                    st.markdown(out or "— awaiting content —")

            # Progress callback: live expanders + graph
            def update_agent_status(agent_name, status, message=""):
                # Update local state
                agent_status.setdefault(agent_name, {"status": "pending", "output": "", "duration": 0})
                agent_status[agent_name]["status"] = status
                if message:
                    agent_status[agent_name]["output"] += html.unescape(message) + "\n"

                # Live expanders
                if agent_name == "rag_analyst":
                    render_agent_expander("rag_analyst", "🧠 RAG Analyst Draft", rag_expander_ph)
                elif agent_name == "web_researcher":
                    render_agent_expander("web_researcher", "🔎 Web Researcher Findings", web_expander_ph)
                elif agent_name == "evidence_verifier":
                    render_agent_expander("evidence_verifier", "✅ Evidence Verifier Analysis", verify_expander_ph)
                elif agent_name == "js_summarizer":
                    render_agent_expander("js_summarizer", "🧩 JS Summarizer (Complete Final Summary)", js_expander_ph)

                # Live workflow graph (throttle ~10fps)
                now = time.time()
                if show_workflow and (now - last_graph_update[0]) > 0.1:
                # if show_workflow and (now - last_graph_update) > 0.1:
                    workflow_graph_ph.graphviz_chart(build_workflow_dot(agent_status))
                    last_graph_update[0] = now
                    

            # Run the streaming analysis
            async def run_analysis():
                outputs = await run_rag_stream(query, context, progress_callback=update_agent_status)
                return outputs

            # Handle asyncio in Streamlit
            try:
                outputs = asyncio.run(run_analysis())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                outputs = loop.run_until_complete(run_analysis())
                loop.close()

            # Parse outputs (your existing helper)
            agent_outputs, final_answer = parse_agent_outputs(outputs)

            # Backfill expanders if some agents didn't stream messages
            def backfill(agent_key: str, parsed_text: str, title: str, ph):
                has_stream = (agent_status.get(agent_key, {}).get("output", "").strip() != "")
                if not has_stream and (parsed_text or "").strip():
                    agent_status.setdefault(agent_key, {"status": "pending", "output": "", "duration": 0})
                    agent_status[agent_key]["output"] = parsed_text
                    agent_status[agent_key]["status"] = "completed"
                    render_agent_expander(agent_key, title, ph)

            backfill("rag_analyst", agent_outputs.get("rag_draft", ""), "🧠 RAG Analyst Draft", rag_expander_ph)
            backfill("web_researcher", agent_outputs.get("web_researcher", ""), "🔎 Web Researcher Findings", web_expander_ph)
            backfill("evidence_verifier", agent_outputs.get("evidence_verifier", ""), "✅ Evidence Verifier Analysis", verify_expander_ph)
            backfill("js_summarizer", agent_outputs.get("final_summary", ""), "🧩 JS Summarizer (Complete Final Summary)", js_expander_ph)

            # Complete progress
            progress_bar.progress(100)
            total_duration = time.time() - start_time
            status_text.text(f"✅ Multi-Agent Analysis Complete! (⏱ {total_duration:.1f}s)")

            # Final Summary — neat formatting
            final_summary = agent_outputs["final_summary"] or "—"
            final_summary = "\n\n".join([s.strip() for s in final_summary.split("\n") if s.strip()])
            final_summary_ph.markdown(final_summary)

            # Raw final answer
            with raw_final_answer_ph.expander("📄 Raw Final Answer (as returned by orchestrator)", expanded=False):
                st.code(final_answer or "—")

            # Final graph render to lock completed state
            if show_workflow:
                workflow_graph_ph.graphviz_chart(build_workflow_dot(agent_status))

            # Optional: Handoff summary
            st.markdown("#### 🔎 Handoff Summary")
            st.markdown(
                "- 🧠 **RAG Analyst** processes local context, **delegates** to Web Researcher & Evidence Verifier.\n"
                "- 🔍 **Web Researcher** returns external **evidence**.\n"
                "- ✅ **Evidence Verifier** validates using **query + context + web evidence**.\n"
                "- 🧩 **JS Summarizer** synthesizes **all previous outputs** into the final structured response."
            )

        except Exception as e:
            status_text.text("❌ Error during Multi-Agent Analysis.")
            st.error(f"An error occurred: {e}")
