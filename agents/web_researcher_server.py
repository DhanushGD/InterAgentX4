
# agents/web_researcher_server.py
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
import json

from a2a.types import AgentCard, AgentCapabilities
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.utils import new_agent_text_message
from a2a.server.events import EventQueue

# from crewai import Agent as CrewAgent, LLM
from crewai.mcp import MCPServerHTTP  # adjust import if different
import asyncio

from agents.common_config import (
    OPENAI_API_KEY, OPENAI_MODEL, MCP_HTTP_URL, WEB_RESEARCHER_PORT,
)
# from agents.utils import extract_user_text, get_params_from_context, mk_agent_skill, safe_enqueue
from crewai import Agent as CrewAgent, LLM, Task, Crew

class WebResearcherCrew:
    def __init__(self):
        self.llm = LLM(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
        self.mcp_conn = MCPServerHTTP(url=MCP_HTTP_URL, streamable=True)
        self.agent = CrewAgent(
            role="Web Researcher",
            goal=("Find minimal external evidence (web/arXiv) and synthesize with citations."),
            backstory=("Specialist in web/arXiv discovery and citation hygiene."),
            llm=self.llm,
            mcps=[self.mcp_conn],
            verbose=True,
        )

    
    def build_task(self, query: str) -> Task:
        prompt = (
            "ROLE: Web Researcher\n"
            "Goal: Find minimal external evidence using MCP tools (duckduckgo_search/arxiv_search). Return concise bullets with citations.\n"
            "Instructions:\n"
            "- Use MCP tools to fetch authoritative sources (prefer papers, official sites).\n"
            "- Return 3–5 bullets: Title — 1–2 line finding — [URL]\n"
            "- If limited, return what you have; no filler.\n\n"
            f"Query: {query}\n\n"
            "Output:\n"
            "- Bulleted synthesis with [URL] citations only.\n"
        )
        return Task(
            description=prompt,
            agent=self.agent,
            expected_output="Bulleted synthesis with [URL] citations.",
        )

    def run_sync(self, task: Task) -> str:
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        result = crew.kickoff()
        return getattr(result, "raw", None) or str(result)


class WebResearcherExecutor(AgentExecutor):
    def __init__(self):
        self.researcher = WebResearcherCrew()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Extract query from message (similar to RAG Analyst)
        message = get_message_dict(context)
        query = extract_user_text_from_message(message)
        local_context = extract_local_context_from_message(message)

        if not query.strip():
            await safe_enqueue(event_queue, new_agent_text_message("❗ No query text provided to Web Researcher (from message.parts)."))
            return

        # Build task: encourage MCP tools usage based on query
        task = self.researcher.build_task(query)

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, self.researcher.run_sync, task)
            await safe_enqueue(event_queue, new_agent_text_message(result))
        except Exception as e:
            await safe_enqueue(event_queue, new_agent_text_message(f"❌ Web research failed: {str(e)}"))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")

def build_app():
    skill = mk_agent_skill(
        id_="web_research",
        name="External Web/ArXiv Research",
        description="Search web/arXiv via MCP and return concise synthesis.",
        tags=["research", "mcp", "duckduckgo", "arxiv"],
        examples=["find latest on topic X", "summarize web sources for Y"],
    )
    public_card = AgentCard(
        name="Web Researcher Agent",
        description="External evidence finder (DuckDuckGo/ArXiv via MCP).",
        url=f"http://localhost:{WEB_RESEARCHER_PORT}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        supports_authenticated_extended_card=False,
    )
    handler = DefaultRequestHandler(agent_executor=WebResearcherExecutor(), task_store=None)
    return A2AStarletteApplication(agent_card=public_card, http_handler=handler).build()

if __name__ == "__main__":
    uvicorn.run(build_app(), host="0.0.0.0", port=WEB_RESEARCHER_PORT)
