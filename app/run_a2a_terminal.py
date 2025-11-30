import os
import sys
import json
import asyncio
from uuid import uuid4

from dotenv import load_dotenv
import httpx

# Load environment variables
load_dotenv()

# A2A client (modern JSON-RPC)
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import SendStreamingMessageRequest, MessageSendParams
from dotenv import load_dotenv

# -----------------------------
# Hardcoded inputs per your ask
# -----------------------------
QUERY = "what is Supervised learning Supervised learning Reinforcement learning and what is their uses?"
CHROMA_DB_DIR = r"C:\Users\DhanushGanesan\Desktop\DHANU\InterAgentX\InteragentSeq\chroma_db"

CHROMA_COLLECTION = "interagentx_rag"

# Default A2A base URLs (adjust if you changed ports)
RAG_ANALYST_BASE = os.getenv("RAG_ANALYST_BASE", "http://localhost:9101")
# Helpers are called by coordinator; only needed if you want to test them directly:
WEB_RESEARCHER_BASE = os.getenv("WEB_RESEARCHER_BASE", "http://localhost:9102")
EVIDENCE_VERIFIER_BASE = os.getenv("EVIDENCE_VERIFIER_BASE", "http://localhost:9103")


# -----------------------------
# Retrieval helpers (import yours)
# -----------------------------
def load_index_and_context(query: str) -> tuple[str, list, dict]:
    """
    Load your Chroma index and get local context, nodes, sources.
    Replace imports with your real module paths.
    """
    try:
        # Replace these imports with your actual project paths
        from new_rag_utils import load_index_from_chroma, retrieve_context  # noqa
        idx = load_index_from_chroma(db_dir=CHROMA_DB_DIR, collection_name=CHROMA_COLLECTION)
        context, nodes, sources = retrieve_context(
            idx, query, top_k=5, use_mmr=True, mmr_diversity=0.3
        )
        return context, nodes, sources
    except Exception as e:
        # Minimal fallback so script still runs—replace with the real functions in your repo
        print(f"[WARN] Could not import your rag retrieval helpers: {e}", file=sys.stderr)
        print("[WARN] Using demo fallback context. Replace with your real retrieval.", file=sys.stderr)
        demo_ctx = "[Demo context snippet about ML categories]\nSupervised / Unsupervised / Reinforcement..."
        return demo_ctx, ["n1", "n2"], {"demo.pdf:p1": "snippet..."}



import asyncio
import json
from uuid import uuid4
import httpx

# Make sure these are imported from your A2A SDK
# from your_module import A2ACardResolver, A2AClient, SendStreamingMessageRequest, MessageSendParams
# And RAG_ANALYST_BASE is defined

async def run_rag_stream(query: str, context: str, progress_callback=None) -> dict:
    """
    Calls RAG Analyst over A2A (JSON-RPC streaming), buckets outputs, and EMITS LIVE STATUS EVENTS:
      - rag_analyst: running -> completed
      - web_researcher: running -> completed
      - evidence_verifier: running -> completed
      - js_summarizer: running -> completed

    Returns:
      {
        final_answer: str,
        web_researcher: str,
        evidence_verifier: str,
        rag_draft: str,
        raw_lines: [str, ...],
        raw_tail: str
      }
    """
    timeout = httpx.Timeout(connect=5.0, read=180.0, write=180.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as httpx_client:
        # Resolve agent card
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=RAG_ANALYST_BASE)
        card = await resolver.get_agent_card()

        # Create JSON-RPC client
        client = A2AClient(httpx_client=httpx_client, agent_card=card)

        # Embed local context in message parts (servers read context.message)
        payload = {
            "message": {
                "role": "user",
                "parts": [
                    {"kind": "text", "text": query},
                    {"kind": "text", "text": f"[[LOCAL_CONTEXT]]\n{context}"},
                ],
                "messageId": uuid4().hex,
            }
        }

        streaming_req = SendStreamingMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**payload),
        )

        # Buckets
        buckets = {
            "final_answer": "",
            "web_researcher": [],
            "evidence_verifier": [],
            "rag_draft": [],
            "raw_lines": [],
        }

        # Flags: first-seen to emit "running" exactly once per agent
        seen_running = {
            "rag_analyst": False,
            "web_researcher": False,
            "evidence_verifier": False,
            "js_summarizer": False,
        }
        js_completed = False

        print("▶ Calling RAG Analyst (A2A streaming)...")
        if progress_callback:
            progress_callback("rag_analyst", "running", "Starting RAG Analyst analysis...")

        try:
            stream = client.send_message_streaming(streaming_req)
        except Exception as e:
            print(f"❌ Error calling agent: {e}")
            if progress_callback:
                progress_callback("rag_analyst", "error", f"Error: {e}")
            return {
                "final_answer": f"Error calling agent: {e}",
                "web_researcher": "",
                "evidence_verifier": "",
                "rag_draft": "",
                "raw_lines": [f"Error: {e}"],
                "raw_tail": f"Error: {e}",
            }

        # --- Streaming loop ---
        async for chunk in stream:
            chunk_texts = []

            # Try structured extraction first
            parts_obj = None
            if hasattr(chunk, 'root') and hasattr(chunk.root, 'result') and hasattr(chunk.root.result, 'parts'):
                parts_obj = chunk.root.result

            if parts_obj and getattr(parts_obj, "parts", None):
                try:
                    for p in parts_obj.parts:
                        if hasattr(p, 'root') and hasattr(p.root, 'text') and isinstance(p.root.text, str):
                            chunk_texts.append(p.root.text)
                        elif hasattr(p, 'text') and isinstance(p.text, str):
                            chunk_texts.append(p.text)
                except Exception:
                    pass

            # Fallback to dict model dump
            if not chunk_texts:
                try:
                    d = chunk.model_dump(mode="json", exclude_none=True)
                    res = d.get("result") or {}

                    # Full message parts
                    if res.get("kind") == "message":
                        parts = res.get("parts") or []
                        for p in parts:
                            if isinstance(p, dict) and p.get("kind") == "text" and isinstance(p.get("text"), str):
                                chunk_texts.append(p["text"])

                    # Delta payloads
                    elif res.get("kind") in ("delta", "textDelta"):
                        delta = res.get("delta") or {}
                        t = delta.get("text") or delta.get("delta")
                        if isinstance(t, str):
                            chunk_texts.append(t)

                    if not chunk_texts:
                        chunk_texts.append(json.dumps(d, ensure_ascii=False))
                except Exception:
                    chunk_texts.append(str(chunk))

            # Route each line
            for chunk_text in chunk_texts:
                for s in str(chunk_text).split('\n'):
                    s = s.strip()
                    if not s:
                        continue

                    # Capture raw line
                    buckets["raw_lines"].append(s)

                    # --- New-format final summary (JS Summarizer) ---
                    if s.startswith("🧩 Final Summary:"):
                        if progress_callback and not seen_running["js_summarizer"]:
                            progress_callback("js_summarizer", "running", "Synthesizing all agent outputs...")
                            seen_running["js_summarizer"] = True

                        buckets["final_answer"] = s.split("🧩 Final Summary:", 1)[-1].strip()

                        if progress_callback:
                            progress_callback("js_summarizer", "completed", "Final summary generated")
                        js_completed = True
                        continue

                    # --- Prefixed legacy lines ---
                    if s.startswith("🔎 Web Researcher:"):
                        if progress_callback and not seen_running["web_researcher"]:
                            progress_callback("web_researcher", "running", "Web research in progress...")
                            seen_running["web_researcher"] = True
                        buckets["web_researcher"].append(s.replace("🔎 Web Researcher:", "").strip())
                        continue

                    if s.startswith("✅ Evidence Verifier:"):
                        if progress_callback and not seen_running["evidence_verifier"]:
                            progress_callback("evidence_verifier", "running", "Evidence verification in progress...")
                            seen_running["evidence_verifier"] = True
                        buckets["evidence_verifier"].append(s.replace("✅ Evidence Verifier:", "").strip())
                        continue

                    if s.startswith("🧠 Draft:"):
                        if progress_callback and not seen_running["rag_analyst"]:
                            progress_callback("rag_analyst", "running", "RAG Analyst processing draft...")
                            seen_running["rag_analyst"] = True
                        buckets["rag_draft"].append(s.replace("🧠 Draft:", "").strip())
                        continue

                    if s.startswith("🧩 Final Answer:"):
                        buckets["final_answer"] = s.split("🧩 Final Answer:", 1)[-1].strip()
                        # We’ll mark js summarizer completed at the end if needed
                        continue

                    # --- Content-based fallbacks (if prefixes missing) ---
                    lower = s.lower()
                    # Web-ish: URLs or reference-style lines
                    if ("http://" in s or "https://" in s) or ("—" in s and "[" in s and "]" in s):
                        if progress_callback and not seen_running["web_researcher"]:
                            progress_callback("web_researcher", "running", "Web research in progress...")
                            seen_running["web_researcher"] = True
                        buckets["web_researcher"].append(s)
                        continue

                    # Evidence-ish keywords
                    if any(m in lower for m in ("confidence:", "caveat", "contradict", "unsupported", "validation")):
                        if progress_callback and not seen_running["evidence_verifier"]:
                            progress_callback("evidence_verifier", "running", "Evidence verification in progress...")
                            seen_running["evidence_verifier"] = True
                        buckets["evidence_verifier"].append(s)
                        continue

                    # RAG-ish: first generic non-empty draft line
                    if not buckets["rag_draft"] and len(s) > 0 and not s.startswith(("🟨","🟦","🟧")):
                        if progress_callback and not seen_running["rag_analyst"]:
                            progress_callback("rag_analyst", "running", "RAG Analyst processing draft...")
                            seen_running["rag_analyst"] = True
                        buckets["rag_draft"].append(s)
                        continue

                    # Final-answer-ish if sources are mentioned and long enough
                    if ("source" in lower or ("[" in s and "]" in s)) and len(s) > 60:
                        buckets["final_answer"] = s
                        continue

        # OPTIONAL: a tiny pause to let UI render last update
        await asyncio.sleep(0.1)

        # Emit completed statuses for agents that produced outputs
        if progress_callback:
            if buckets["rag_draft"]:
                progress_callback("rag_analyst", "completed", "RAG Analyst draft ready")
            if buckets["web_researcher"]:
                progress_callback("web_researcher", "completed", "Web research collected")
            if buckets["evidence_verifier"]:
                progress_callback("evidence_verifier", "completed", "Evidence verified")

            # If final_answer exists but we never saw JS completed (legacy format)
            if buckets["final_answer"] and not js_completed:
                if not seen_running["js_summarizer"]:
                    progress_callback("js_summarizer", "running", "Synthesizing final answer...")
                progress_callback("js_summarizer", "completed", "Final summary generated")

        # Fallback: ensure final answer is set
        if not buckets["final_answer"]:
            for line in reversed(buckets["raw_lines"]):
                if line.startswith("🧩 Final Summary:"):
                    buckets["final_answer"] = line.split("🧩 Final Summary:", 1)[-1].strip()
                    break
                if line.startswith("🧩 Final Answer:"):
                    buckets["final_answer"] = line.split("🧩 Final Answer:", 1)[-1].strip()
                    break
            # As a last resort: take last long informative line
            if not buckets["final_answer"]:
                for line in reversed(buckets["raw_lines"]):
                    if len(line) > 80:
                        buckets["final_answer"] = line.strip()
                        break

        # Merge lists to strings
        result = {
            "final_answer": buckets["final_answer"].strip(),
            "web_researcher": "\n".join(buckets["web_researcher"]).strip(),
            "evidence_verifier": "\n".join(buckets["evidence_verifier"]).strip(),
            "rag_draft": "\n".join(buckets["rag_draft"]).strip(),
            "raw_lines": buckets["raw_lines"],
            "raw_tail": "\n".join(buckets["raw_lines"][-10:]).strip(),
        }
        return result

# -----------------------------
# Main
# -----------------------------
def print_outputs(outputs: dict, sources: dict):
    print("\n" + "="*80)
    print("🤖 INTERAGENTX MULTI-AGENT ANALYSIS COMPLETE")
    print("="*80)

    # Print all agent outputs first
    print("\n📋 ALL AGENT RESPONSES:")
    print("-" * 50)

    if outputs.get("rag_draft"):
        draft = outputs["rag_draft"].strip()
        print("🧠 RAG ANALYST OUTPUT:")
        print(draft)
        print(f"   → {len(draft)} characters\n")

    if outputs.get("web_researcher"):
        web = outputs["web_researcher"].strip()
        print("🔍 WEB RESEARCHER OUTPUT:")
        print(web)
        print(f"   → {len(web)} characters\n")

    if outputs.get("evidence_verifier"):
        evidence = outputs["evidence_verifier"].strip()
        print("✅ EVIDENCE VERIFIER OUTPUT:")
        print(evidence)
        print(f"   → {len(evidence)} characters\n")

    if outputs.get("raw_lines"):
        print("📄 RAW LINES FROM ALL AGENTS:")
        for i, line in enumerate(outputs["raw_lines"], 1):
            print(f"   {i:2d}. {line}")
        print(f"   → {len(outputs['raw_lines'])} total lines\n")

    # Print final summary
    print("="*80)
    print("🧩 FINAL SUMMARY:")
    print("="*80)

    final_answer = outputs.get("final_answer", "").strip()
    if final_answer:
        print(final_answer)
    else:
        print("❌ No final summary generated")

    print("\n" + "-"*80)
    print("📊 AGENT PROCESSING SUMMARY")
    print("-"*80)

    if outputs.get("rag_draft"):
        draft = outputs["rag_draft"].strip()
        print(f"🧠 RAG Analyst Contributions: {len(draft)} characters processed")

    if outputs.get("web_researcher"):
        web = outputs["web_researcher"].strip()
        print(f"🔍 Web Researcher: {len(web)} characters of external evidence integrated")

    if outputs.get("evidence_verifier"):
        evidence = outputs["evidence_verifier"].strip()
        print(f"✅ Evidence Verifier: {len(evidence)} characters of validation analysis")

    print(f"📈 Total Processing: {len(final_answer)} characters in final summary")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Ensure env
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY must be set.", file=sys.stderr)
        sys.exit(1)
    if not os.getenv("MCP_HTTP_URL"):
        # default MCP per your server
        os.environ["MCP_HTTP_URL"] = "http://127.0.0.1:8000/mcp"

    # Load index & context
    context, nodes, sources = load_index_and_context(QUERY)
    # print(context)

    # Run the A2A streaming call
    try:
        outputs = asyncio.run(run_rag_stream(QUERY, context))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        outputs = loop.run_until_complete(run_rag_stream(QUERY, context))
        loop.close()

    # Print results
    print_outputs(outputs, sources)
