# Copyright (c) 2026 David vonThenen. All rights reserved.
# Restricted distribution. Unauthorized copying or hosting of this file via any medium
# is strictly prohibited. Proprietary and confidential.

"""MCP server that answers questions with Tavily or keyless web search results."""
from __future__ import annotations

from typing import Any
import os

from mcp.server import MCPServer

from common.config import load_settings
from common.duckduckgo_client import DuckDuckGoSearchClient
from common.tavily_client import TavilySearchClient


MCP_HOST = "127.0.0.1"
MCP_PORT = 7000

settings = load_settings()

# Force DuckDuckGo using an environment variable
duckduckgo_enable = os.getenv("FORCE_DUCKDUCKGO")

if settings.tavily_api_key and not duckduckgo_enable:
    search_provider = "tavily"
    search_client = TavilySearchClient(settings)
else:
    search_provider = "duckduckgo"
    search_client = DuckDuckGoSearchClient(settings)

mcp = MCPServer("Web Search")


def _format_results(question: str, results: list[dict[str, Any]]) -> str:
    if not results:
        return f"Provider: {search_provider}\n\nNo results found for: {question}"

    provider = str(results[0].get("provider") or search_provider)
    lines = [f"Provider: {provider}", ""]

    for index, result in enumerate(results, start=1):
        title = str(result.get("title") or "Untitled result")
        content = str(result.get("content") or "").strip()
        url = str(result.get("url") or "").strip()

        lines.append(f"{index}. {title}")
        if content:
            lines.append(content)
        if url:
            lines.append(url)
        lines.append("")

    return "\n".join(lines).rstrip()


@mcp.tool()
def answer_question(question: str) -> str:
    """Search the web for a question and return the most relevant results."""
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty")

    results = search_client.search(question)
    return _format_results(question, results)


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host=MCP_HOST,
        port=MCP_PORT,
    )
