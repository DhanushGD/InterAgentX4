
# agents/rag_analyst_server.py
import uvicorn
from typing import Any

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from utils import (
    mk_agent_skill,
    context_attributes_snapshot,
    get_raw_request_json,
    get_message_dict,
    extract_user_text_from_message,
    extract_local_context_from_message,
    safe_enqueue,

)
from a2a.types import SendMessageRequest, MessageSendParams
import json
# new_agent_text_delta
from a2a.types import AgentCard, AgentCapabilities
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.utils import new_agent_text_message
from a2a.server.events import EventQueue
from a2a.types import AgentCard, AgentCapabilities, SendStreamingMessageRequest, MessageSendParams
from uuid import uuid4
from a2a.client import A2ACardResolver, A2AClient
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH

from crewai import Agent as CrewAgent, LLM, Task, Crew
from crewai.mcp import MCPServerHTTP  # adjust import if different

from agents.common_config import (
    OPENAI_API_KEY, OPENAI_MODEL, MCP_HTTP_URL,
    RAG_ANALYST_PORT, WEB_RESEARCHER_BASE, EVIDENCE_VERIFIER_BASE, JS_SUMMARIZER_BASE
)
import asyncio


import re



class RAGAnalystCrew:
    def __init__(self):
        self.llm = LLM(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
        self.mcp_conn = MCPServerHTTP(url=MCP_HTTP_URL, streamable=True)
        self.agent = CrewAgent(
            role="RAG Analyst",
            goal=(
                "Answer strictly grounded in provided local context. "
                "If insufficient, engage helper agents via A2A or call MCP as needed."
            ),
            backstory=("InterAgentX coordinator; prioritizes local corpus, uses helpers to fill gaps."),
            llm=self.llm,
            # mcps=[self.mcp_conn],
            verbose=True,
        )

        
    def build_draft_task(self, query: str, local_context: str) -> Task:
        prompt = (
            "SYSTEM:\n"
            "You are the RAG Analyst coordinator. You must decide whether to delegate to helper agents.\n\n"
            "POLICY:\n"
            "- PRIORITY: Use ONLY the provided local context for the first draft.\n"
            "- DO NOT call external tools yourself. If external evidence is needed, DELEGATE.\n"
            "- If the context is even slightly incomplete OR the question asks for experiments, results, baselines,\n"
            "  evaluations, or paper-specific evidence, you MUST delegate.\n"
            "- Even when the context looks enough, delegate to VERIFY unless confidence >= 0.95.\n"
            "- Use the following control tags exactly:\n"
            "  <DELEGATE_WEB/> when you need external evidence; <DELEGATE_VERIFY/> when you need validation;\n"
            "  <NO_DELEGATE/> only if confidence >= 0.95 and answer fully grounded locally.\n"
            "- Always produce a draft answer in <DRAFT>...</DRAFT> after tags.\n"
            "- If you are unsure, include <DELEGATE_WEB/><DELEGATE_VERIFY/>.\n\n"
            "QUESTION:\n"
            f"{query}\n\n"
            "LOCAL CONTEXT:\n"
            f"{local_context}\n\n"
            "OUTPUT FORMAT:\n"
            "First line MUST contain one or both tags (<DELEGATE_WEB/>, <DELEGATE_VERIFY/>) OR <NO_DELEGATE/>.\n"
            "Then a draft answer wrapped in <DRAFT>...</DRAFT> with inline local source hints like [filename] or [filename:pN].\n"
            "Example:\n"
            "<DELEGATE_WEB/><DELEGATE_VERIFY/>\n"
            "<DRAFT>Short answer grounded in local context... [doc:p2]</DRAFT>\n"
        )
        return Task(
            description=prompt,
            agent=self.agent,
            expected_output="Control tags on first line + <DRAFT>...</DRAFT> answer with local source hints."
        )

    def build_synth_task(self, query: str, local_context: str, draft: str, web_evidence: str, validation: str) -> Task:
        prompt = (
            f"Question:\n{query}\n\n"
            f"Local Context:\n{local_context[:6000]}\n\n"
            f"First Draft:\n{draft}\n\n"
            f"Web Evidence (concise):\n{web_evidence[:4000]}\n\n"
            f"Validation Notes:\n{validation[:3000]}\n\n"
            "Synthesize a final answer:\n"
            "- Prioritize local context; use web evidence only to fill gaps or confirm.\n"
            "- Include inline source hints: local [file:pN], web [URL].\n"
            "- Be concise and accurate; include caveats when confidence is low."
        )
        return Task(description=prompt, agent=self.agent, expected_output="Final synthesized answer with source hints.")

    def run_sync(self, task: Task) -> str:
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        res = crew.kickoff()
        return getattr(res, "raw", None) or str(res)


class RAGAnalystExecutor(AgentExecutor):
    def __init__(self):
        self.rag = RAGAnalystCrew()
        import os
        self.force_helpers = os.environ.get("FORCE_HELPERS", "0") in ("1", "true", "yes")

    def _should_delegate(self, query: str, local_context: str, draft: str) -> bool:
        if self.force_helpers:
            return True
        q = (query or "").lower()
        intent_triggers = [
            "fresh", "latest", "recent", "new", "today", "up-to-date",
            "experiment", "experiments", "results", "evaluation", "baseline",
            "paper", "arxiv", "research", "study", "evidence", "validate"
        ]
        trigger_hit = any(t in q for t in intent_triggers)
        short_ctx = len(local_context) < 1200
        low_conf = ("i don't know" in draft.lower()) or ("insufficient" in draft.lower())
        return trigger_hit or short_ctx or low_conf

    async def _call_js_summarizer(self, summarizer_data: dict) -> str:
        """Call JS Summarizer Agent via A2A for final synthesis."""
        import httpx
        timeout = httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=JS_SUMMARIZER_BASE)
            card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=card)

            # Format the data as a structured message for the summarizer
            message_text = json.dumps(summarizer_data, ensure_ascii=False, indent=2)

            message_payload = {
                "message": {
                    "role": "user",
                    "parts": [
                        {"kind": "text", "text": f"Please provide a comprehensive summary of the following analysis results:\n\n{message_text}"}
                    ],
                    "messageId": uuid4().hex,
                }
            }

            req = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**message_payload),
            )
            resp = await client.send_message(req)

            # Extract the summarized result
            try:
                d = resp.model_dump(mode="json", exclude_none=True)
                res = d.get("result") or {}
                parts = res.get("parts") or []
                if parts and isinstance(parts[0], dict) and parts[0].get("kind") == "text":
                    return parts[0]["text"]
                return json.dumps(d, ensure_ascii=False)
            except Exception:
                return f"Error extracting summary from JS Summarizer response: {resp}"

    async def _call_helper_non_streaming(self, base_url: str, message_payload: dict) -> str:
        """Call helper via JSON-RPC non-streaming and return text."""
        import httpx
        timeout = httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as httpx_client:
        # async with httpx.AsyncClient() as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
            card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=card)

            req = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**message_payload),
            )
            resp = await client.send_message(req)

            # Try extracting text from resp.result.parts[0].text
            try:
                d = resp.model_dump(mode="json", exclude_none=True)
                res = d.get("result") or {}
                parts = res.get("parts") or []
                if parts and isinstance(parts[0], dict) and parts[0].get("kind") == "text":
                    return parts[0]["text"]
                # fallback to whole JSON
                return json.dumps(d, ensure_ascii=False)
            except Exception:
                pass

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # 0) Extract query + local context from typed message
        message = get_message_dict(context)
        query = extract_user_text_from_message(message)
        local_context = extract_local_context_from_message(message)

        if not query.strip():
            await safe_enqueue(event_queue, new_agent_text_message(
                "❗ No query text provided to RAG Analyst (from message.parts)."
            ))
            return

        # 1) Draft using local context (CrewAI in thread)
        print(f"[RAG ANALYST] Generating draft for query: {query[:100]}...")
        print(f"[RAG ANALYST] Local context length: {len(local_context)} characters")
        draft_task = self.rag.build_draft_task(query, local_context)
        loop = asyncio.get_running_loop()
        try:
            draft_text = await loop.run_in_executor(None, self.rag.run_sync, draft_task)
            print(f"[RAG ANALYST] Draft generated: {len(draft_text)} characters")
            print(f"[RAG ANALYST] Draft preview: {draft_text[:300]}...")
        except Exception as e:
            final_combined = (
                "🧠 Draft:\n(error)\n\n"
                f"❌ Draft generation failed: {e}\n\n"
                "🧩 Final Answer:\nError: Could not generate draft. Please check OPENAI_API_KEY and CrewAI configuration."
            )
            await safe_enqueue(event_queue, new_agent_text_message(final_combined))
            return

        # Optional tag parsing (safe)
        try:
            tags = self._parse_control_tags(draft_text)
            draft_only = self._extract_draft(draft_text)
        except Exception:
            draft_only = draft_text

        # Always delegate to show multi-agent workflow (ignore NO_DELEGATE tag)
        should_delegate_web = True
        should_delegate_verify = True

        # 2) Web Researcher (A2A non-streaming)
        web_evidence_text = ""
        if should_delegate_web:
            helper_message_payload = {
                "message": {
                    "role": "user",
                    "parts": [
                        {"kind": "text", "text": query},
                        {"kind": "text", "text": f"[[LOCAL_CONTEXT]]\n{local_context or ''}"},
                    ],
                    "messageId": uuid4().hex,
                }
            }
            try:
                print(f"[RAG ANALYST] Calling Web Researcher with query: {query[:100]}...")
                web_evidence_text = await self._call_helper_non_streaming(
                    WEB_RESEARCHER_BASE, helper_message_payload
                )
                print(f"[RAG ANALYST] Web Researcher returned {len(web_evidence_text)} characters")
                print(f"[RAG ANALYST] Web Researcher output preview: {web_evidence_text[:300]}...")
            except Exception as e:
                web_evidence_text = f"(web research error: {e})"
                print(f"[RAG ANALYST] Web Researcher error: {e}")

        # 3) Evidence Verifier (A2A non-streaming)
        validation_text = ""
        if should_delegate_verify:
            verifier_message_payload = {
                "message": {
                    "role": "user",
                    "parts": [
                        {"kind": "text", "text": query},
                        {"kind": "text", "text": f"[[LOCAL_CONTEXT]]\n{(local_context or '')}\n\n[WEB_EVIDENCE]\n{web_evidence_text[:2000]}"},
                    ],
                    "messageId": uuid4().hex,
                }
            }
            try:
                print(f"[RAG ANALYST] Calling Evidence Verifier with query: {query[:100]}...")
                validation_text = await self._call_helper_non_streaming(
                    EVIDENCE_VERIFIER_BASE, verifier_message_payload
                )
                print(f"[RAG ANALYST] Evidence Verifier returned {len(validation_text)} characters")
                print(f"[RAG ANALYST] Evidence Verifier output preview: {validation_text[:300]}...")
            except Exception as e:
                validation_text = f"(verification error: {e})"
                print(f"[RAG ANALYST] Evidence Verifier error: {e}")

        # 4) Final synthesis (CrewAI in thread)
        print(f"[RAG ANALYST] Starting final synthesis with:")
        print(f"  - Draft length: {len(draft_only)} characters")
        print(f"  - Web evidence length: {len(web_evidence_text)} characters")
        print(f"  - Validation length: {len(validation_text)} characters")
        try:
            final_task = self.rag.build_synth_task(
                query, local_context, draft_only, web_evidence_text, validation_text
            )
            final_answer = await loop.run_in_executor(None, self.rag.run_sync, final_task)
            print(f"[RAG ANALYST] Final synthesis completed: {len(final_answer)} characters")
            print(f"[RAG ANALYST] Final synthesis preview: {final_answer[:300]}...")
        except Exception as e:
            final_answer = (
                f"{draft_only}\n\n"
                f"(final synthesis error: {e})"
            )
            print(f"[RAG ANALYST] Final synthesis error: {e}")

        # 5) Delegate to JS Summarizer Agent for final synthesis
        try:
            # Prepare structured data for the JS summarizer
            summarizer_payload = {
                "query": query,
                "local_context": local_context,
                "rag_draft": draft_only,
                "web_evidence": web_evidence_text or "(none)",
                "evidence_verification": validation_text or "(none)",
                "crew_synthesis": final_answer
            }

            print(f"[RAG ANALYST] Delegating to JS Summarizer with payload: {summarizer_payload}")

            # Send to JS Summarizer via A2A (non-streaming for simplicity)
            summarized_result = await self._call_js_summarizer(summarizer_payload)

            print(f"[RAG ANALYST] Received summarized result: {summarized_result[:200]}...")

            # Create a comprehensive response with all agent outputs
            comprehensive_response = f"""🧩 Multi-Agent Analysis Complete

🧠 RAG ANALYST DRAFT:
{draft_only}

🔎 WEB RESEARCHER FINDINGS:
{web_evidence_text}

✅ EVIDENCE VERIFIER ANALYSIS:
{validation_text}

🧩 FINAL COMPREHENSIVE SUMMARY:
{summarized_result}
"""

            print(f"[RAG ANALYST] All agent outputs collected:")
            print(f"[RAG ANALYST] Draft length: {len(draft_only)}")
            print(f"[RAG ANALYST] Web evidence length: {len(web_evidence_text)}")
            print(f"[RAG ANALYST] Validation length: {len(validation_text)}")
            print(f"[RAG ANALYST] Final summary length: {len(summarized_result)}")
            print(f"[RAG ANALYST] Total response length: {len(comprehensive_response)}")

            # Send the comprehensive response as a single message
            print("[RAG ANALYST] Sending comprehensive multi-agent response...")
            await safe_enqueue(event_queue, new_agent_text_message(comprehensive_response))

            print("[RAG ANALYST] Comprehensive response sent successfully")

        except Exception as e:
            print(f"[RAG ANALYST] Delegation failed: {e}")
            # Fallback to original format if delegation fails
            final_combined = (
                f"🧠 Draft:\n{draft_only}\n\n"
                f"🔎 Web Researcher:\n{web_evidence_text or '(none)'}\n\n"
                f"✅ Evidence Verifier:\n{validation_text or '(none)'}\n\n"
                f"🧩 Final Answer:\n{final_answer}\n\n"
                f"⚠️ Summarization delegation failed: {e}"
            )
            await safe_enqueue(event_queue, new_agent_text_message(final_combined))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")




def build_app():
    skill = mk_agent_skill(
        id_="rag_answer",
        name="RAG Answering",
        description="Answers grounded in local context; delegates via A2A when needed.",
        tags=["rag", "mcp", "coordinator"],
        examples=["answer with local corpus", "use helpers if needed"],
    )
    public_card = AgentCard(
        name="RAG Analyst Agent",
        description="Coordinator RAG agent; prioritizes local corpus; delegates via A2A.",
        url=f"http://localhost:{RAG_ANALYST_PORT}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        supports_authenticated_extended_card=False,
    )
    handler = DefaultRequestHandler(agent_executor=RAGAnalystExecutor(), task_store=None)
    return A2AStarletteApplication(agent_card=public_card, http_handler=handler).build()

if __name__ == "__main__":
    uvicorn.run(build_app(), host="0.0.0.0", port=RAG_ANALYST_PORT)
