import os
from typing import List, Dict, Any

from fastmcp import FastMCP
from ddgs import DDGS            
import arxiv                     

mcp = FastMCP(name="InterAgentX-5-MCP")


@mcp.tool(name="duckduckgo_search", description="Search the web via DuckDuckGo (ddgs) and return top results.")
def duckduckgo_search(query: str, max_results: int = 1) -> List[Dict[str, Any]]:
    """Return a list of dicts: {'title','href','body','source'}"""
    results: List[Dict[str, Any]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title"),
                "href": r.get("href"),
                "body": r.get("body"),
                "source": "duckduckgo"
            })
    return results

@mcp.tool(name="arxiv_search", description="Search arXiv papers and return metadata + PDF links.")
def arxiv_search(query: str, max_results: int = 1, sort_by: str = "submitted_date") -> List[Dict[str, Any]]:
    """
    Return a list of dicts: {'title','authors','summary','pdf_url','entry_id','published'}
    """
    sort = arxiv.SortCriterion.SubmittedDate if sort_by == "submitted_date" else arxiv.SortCriterion.Relevance
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=sort)

    out: List[Dict[str, Any]] = []
    for r in client.results(search):
        pdf_link = None
        for l in r.links:
            if getattr(l, "rel", None) == "related" and getattr(l, "title", None) == "pdf":
                pdf_link = l.href
        out.append({
            "title": r.title,
            "authors": [a.name for a in r.authors],
            "summary": r.summary,
            "pdf_url": pdf_link or r.pdf_url,
            "entry_id": r.entry_id,
            "published": r.published.isoformat() if r.published else None,
        })
    return out

if __name__ == "__main__":
    # mcp.run(transport="sse", host="127.0.0.1", port=8000)
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000, path="/mcp",json_response=True)


# import asyncio
# from fastmcp import Client

# async def main():
#     async with Client("http://127.0.0.1:8000/mcp") as c:
#         tools = await c.list_tools()
#         print("Tools:", tools)
#         res = await c.call_tool("duckduckgo_search", {"query":"OpenSSL 3.x CVE","max_results":2})
#         print("DDG:", res)

# asyncio.run(main())


# from crewai.mcp import MCPServerHTTP

# connection = MCPServerHTTP(url="http://", streamable=True)
# from crewai import Agent, Task, Crew

# agent = Agent(
#     role="Example Agent",
#     mcps=[connection],
#     llm=LLM(model="gpt-4", api_key="YOUR_API_KEY"),
#     verbose=True)

# task = Task(
#     description="Use the MCP tools to answer the user's query.",
#     agent=agent,
#     expected_output="A concise answer based on MCP tool results."
# )

# Crew = Crew(
#     agents=[agent], 
#     tasks=[task],
#     verbose=True,)
#     result = Crew.kickoff()
#     print("Final Result:", result)
