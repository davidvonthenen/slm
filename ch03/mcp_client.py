# Copyright (c) 2026 David vonThenen. All rights reserved.
# Restricted distribution. Unauthorized copying or hosting of this file via any medium
# is strictly prohibited. Proprietary and confidential.

"""Command-line client for the web-search MCP server."""
from __future__ import annotations

import argparse
import asyncio
import sys

from mcp import Client
from mcp.types import TextContent


MCP_URL = "http://127.0.0.1:7000/mcp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask the web-search MCP server a question.")
    parser.add_argument("--question", default="Where is the world wide corporate headquarters for Microsoft?", help="Question to ask.")
    return parser.parse_args()


async def ask_question(question: str) -> str:
    async with Client(MCP_URL) as client:
        result = await client.call_tool(
            "answer_question",
            {"question": question},
        )

    text = "\n".join(
        block.text for block in result.content if isinstance(block, TextContent)
    )

    if result.is_error:
        raise RuntimeError(text or "The MCP tool call failed")
    if not text:
        raise RuntimeError("The MCP server returned no text")

    return text


def main() -> None:
    args = parse_args()
    try:
        print(asyncio.run(ask_question(args.question)))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
