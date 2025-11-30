
# agents/common_config.py
import os
from dotenv import load_dotenv

load_dotenv()
# --- OpenAI / LLM ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4-turbo")  # adjust if needed

# --- MCP Layer ---
MCP_HTTP_URL = os.environ.get("MCP_HTTP_URL", "http://127.0.0.1:8000/mcp")

# --- A2A Agent URLs/Ports ---
RAG_ANALYST_PORT = int(os.environ.get("RAG_ANALYST_PORT", "9101"))
WEB_RESEARCHER_PORT = int(os.environ.get("WEB_RESEARCHER_PORT", "9102"))
EVIDENCE_VERIFIER_PORT = int(os.environ.get("EVIDENCE_VERIFIER_PORT", "9103"))
JS_SUMMARIZER_PORT = int(os.environ.get("JS_SUMMARIZER_PORT", "4100"))

RAG_ANALYST_BASE = f"http://localhost:{RAG_ANALYST_PORT}"
WEB_RESEARCHER_BASE = f"http://localhost:{WEB_RESEARCHER_PORT}"
EVIDENCE_VERIFIER_BASE = f"http://localhost:{EVIDENCE_VERIFIER_PORT}"
JS_SUMMARIZER_BASE = f"http://localhost:{JS_SUMMARIZER_PORT}"

# Debug output
print(f"Loading config from: {os.getcwd()}")
print(f"OPENAI_API_KEY loaded: {bool(OPENAI_API_KEY)} (length: {len(OPENAI_API_KEY) if OPENAI_API_KEY else 0})")
if OPENAI_API_KEY:
    print(f"API Key starts with: {OPENAI_API_KEY[:10]}...")
print(f"OPENAI_MODEL: {OPENAI_MODEL}")