
# agents/evidence_verifier_server.py
import uvicorn
from typing import Any

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import AgentCard, AgentCapabilities
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.utils import new_agent_text_message
from a2a.server.events import EventQueue

from crewai import Agent as CrewAgent, LLM, Task, Crew
from crewai.mcp import MCPServerHTTP

from agents.common_config import (
    OPENAI_API_KEY, OPENAI_MODEL, MCP_HTTP_URL, EVIDENCE_VERIFIER_PORT,
)
import asyncio
from utils import (
    mk_agent_skill,
    context_attributes_snapshot,
    get_raw_request_json,
    get_message_dict,
    extract_user_text_from_message,
    extract_local_context_from_message,
    safe_enqueue,


)
import json


class EvidenceVerifierCrew:
    def __init__(self):
        self.llm = LLM(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
        self.mcp_conn = MCPServerHTTP(url=MCP_HTTP_URL, streamable=True)
        self.agent = CrewAgent(
            role="Evidence Verifier",
            goal=("Validate claims, detect contradictions, assign confidence, propose caveats."),
            backstory=("Critical assessor guarding against hallucinations."),
            llm=self.llm,
            mcps=[self.mcp_conn],
            verbose=True,
        )

       
    def build_task(self, query: str, local_context: str, web_evidence: str) -> Task:
        prompt = (
            "ROLE: Evidence Verifier\n"
            "Goal: Validate the claims against local context and web evidence.\n"
            "Instructions:\n"
            "- Identify contradictions, unsupported claims, missing citations.\n"
            "- Provide a confidence score [0–1].\n"
            "- Provide caveats if needed.\n\n"
            f"Question:\n{query}\n\n"
            f"Local Context:\n{local_context[:8000]}\n\n"
            f"Web Evidence:\n{web_evidence[:6000]}\n\n"
            "Output:\n"
            "- 3–5 bullets: validation findings\n"
            "- Confidence: x.xx\n"
            "- Caveats: ...\n"
        )
        return Task(description=prompt, agent=self.agent, expected_output="Validation bullets + confidence + caveats")


    def run_sync(self, task: Task) -> str:
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        result = crew.kickoff()
        return getattr(result, "raw", None) or str(result)


class EvidenceVerifierExecutor(AgentExecutor):
    def __init__(self):
        self.verifier = EvidenceVerifierCrew()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Extract query and context from message (similar to RAG Analyst)
        message = get_message_dict(context)
        query = extract_user_text_from_message(message)

        # local context may also include a [WEB_EVIDENCE] section appended by the coordinator
        lc = extract_local_context_from_message(message)

        # Pull web evidence short section from local context if appended by RAG analyst
        web_evidence = ""
        if isinstance(lc, str) and "[WEB_EVIDENCE]" in lc:
            parts = lc.split("[WEB_EVIDENCE]", 1)
            local_context = parts[0].strip()
            web_evidence = parts[1].strip()
        else:
            local_context = lc

        if not query.strip():
            await safe_enqueue(event_queue, new_agent_text_message("❗ No query text provided to Evidence Verifier (from message.parts)."))
            return

        task = self.verifier.build_task(query, local_context, web_evidence)

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, self.verifier.run_sync, task)
            await safe_enqueue(event_queue, new_agent_text_message(result))
        except Exception as e:
            await safe_enqueue(event_queue, new_agent_text_message(f"❌ Evidence verification failed: {str(e)}"))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")



def build_app():
    skill = mk_agent_skill(
        id_="evidence_verify",
        name="Claim Validation",
        description="Validate claims, detect contradictions, assign confidence, and caveats.",
        tags=["validation", "verification", "mcp"],
        examples=["verify these claims", "spot contradictions for query X"],
    )
    public_card = AgentCard(
        name="Evidence Verifier Agent",
        description="Validates claims against context and web evidence.",
        url=f"http://localhost:{EVIDENCE_VERIFIER_PORT}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        supports_authenticated_extended_card=False,
    )
    handler = DefaultRequestHandler(agent_executor=EvidenceVerifierExecutor(), task_store=None)
    return A2AStarletteApplication(agent_card=public_card, http_handler=handler).build()


if __name__ == "__main__":
    uvicorn.run(build_app(), host="0.0.0.0", port=EVIDENCE_VERIFIER_PORT)