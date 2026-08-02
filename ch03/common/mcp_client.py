# Copyright (c) 2026 David vonThenen. All rights reserved.
# Restricted distribution. Unauthorized copying or hosting of this file via any medium
# is strictly prohibited. Proprietary and confidential.

"""Reusable client for the streamable-HTTP web-search MCP server."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MCPToolResult:
    """Normalized result returned by one MCP tool call."""

    text: str
    content_blocks: list[dict[str, Any]]
    is_error: bool


class MCPToolCallError(RuntimeError):
    """Raised when the MCP server reports an error or returns no text."""

    def __init__(self, message: str, result: MCPToolResult | None = None) -> None:
        super().__init__(message)
        self.result = result


class MCPWebSearchClient:
    """Call the web-search MCP server using the MCP Python SDK."""

    def __init__(
        self,
        url: str | None = None,
        tool_name: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.url = (url or os.getenv("MCP_URL", "http://127.0.0.1:7000/mcp")).strip()
        self.tool_name = (
            tool_name or os.getenv("MCP_TOOL_NAME", "answer_question")
        ).strip()
        raw_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else os.getenv("MCP_TIMEOUT_SECONDS", "30")
        )

        if not self.url:
            raise ValueError("The MCP URL must not be empty")
        if not self.tool_name:
            raise ValueError("The MCP tool name must not be empty")

        try:
            self.timeout_seconds = float(raw_timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("The MCP timeout must be a number") from exc
        if self.timeout_seconds <= 0:
            raise ValueError("The MCP timeout must be greater than zero")

    @staticmethod
    def _serialize_content_block(block: Any) -> dict[str, Any]:
        """Convert an MCP content block into JSON-serializable trace data."""

        model_dump = getattr(block, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump(mode="json")
            except TypeError:
                dumped = model_dump()
            if isinstance(dumped, dict):
                return dumped

        payload: dict[str, Any] = {
            "type": str(getattr(block, "type", type(block).__name__)),
        }
        text = getattr(block, "text", None)
        if isinstance(text, str):
            payload["text"] = text
        else:
            payload["representation"] = repr(block)
        return payload

    async def answer_question_async(self, question: str) -> MCPToolResult:
        """Ask the configured MCP tool the exact supplied question."""

        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question must not be empty")

        # Keep the SDK import local so syntax checks and dependency injection can
        # run in environments where the MCP package is not installed yet.
        from mcp import Client
        from mcp.types import TextContent

        async def invoke() -> Any:
            async with Client(self.url) as client:
                return await client.call_tool(
                    self.tool_name,
                    {"question": normalized_question},
                )

        try:
            result = await asyncio.wait_for(invoke(), timeout=self.timeout_seconds)
        except TimeoutError as exc:
            raise TimeoutError(
                f"MCP tool call timed out after {self.timeout_seconds:g} seconds"
            ) from exc

        raw_blocks = list(getattr(result, "content", None) or [])
        content_blocks = [self._serialize_content_block(block) for block in raw_blocks]
        text = "\n".join(
            block.text
            for block in raw_blocks
            if isinstance(block, TextContent) and isinstance(block.text, str)
        ).strip()
        normalized_result = MCPToolResult(
            text=text,
            content_blocks=content_blocks,
            is_error=bool(getattr(result, "is_error", False)),
        )

        if normalized_result.is_error:
            raise MCPToolCallError(
                normalized_result.text or "The MCP tool call failed",
                result=normalized_result,
            )
        if not normalized_result.text:
            raise MCPToolCallError(
                "The MCP server returned no text",
                result=normalized_result,
            )

        return normalized_result

    def answer_question(self, question: str) -> MCPToolResult:
        """Synchronous wrapper used by the Flask RAG request path."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.answer_question_async(question))

        raise RuntimeError(
            "MCPWebSearchClient.answer_question cannot run inside an active event loop; "
            "use 'await answer_question_async(...)' instead"
        )


__all__ = [
    "MCPToolCallError",
    "MCPToolResult",
    "MCPWebSearchClient",
]
