"""Synchronous facade over a persistent MCP stdio session.

The MCP client library is async and its transport is a subprocess pipe. The agent and
the API are synchronous and are called from worker threads. Rather than spawn a server
per request, one session is held open on a private event loop thread for the life of
the process, and every call is marshalled onto that loop.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field

from mcp import types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from qc_copilot.config import Settings
from qc_copilot.models import ToolCall


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict

    def as_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass(frozen=True)
class ToolResult:
    call: ToolCall
    structured: dict | None = None
    text: str = ""

    def as_model_text(self) -> str:
        """What the model sees. Structured content when present, otherwise the text."""
        if self.structured is not None:
            return json.dumps(self.structured, ensure_ascii=False)
        return self.text or (self.call.error or "")


@dataclass
class MCPToolClient:
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    startup_timeout: float = 60.0
    call_timeout: float = 30.0

    def __post_init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._session: ClientSession | None = None
        self._failure: BaseException | None = None
        self._tools: list[ToolSpec] = []

    @classmethod
    def from_settings(cls, settings: Settings) -> MCPToolClient:
        # The server subprocess inherits this process's configuration so both sides
        # agree on which database backs the tools.
        env = {
            **os.environ,
            "TOOLS_BACKEND": settings.tools_backend,
            "SQLITE_PATH": str(settings.sqlite_path),
            "MYSQL_HOST": settings.mysql_host,
            "MYSQL_PORT": str(settings.mysql_port),
            "MYSQL_USER": settings.mysql_user,
            "MYSQL_PASSWORD": settings.mysql_password,
            "MYSQL_DATABASE": settings.mysql_database,
        }
        return cls(command=[sys.executable, "-m", "qc_copilot.mcp_server.server"], env=env)

    def start(self) -> MCPToolClient:
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._serve(), self._loop)
        if not self._ready.wait(self.startup_timeout):
            raise RuntimeError("MCP server did not become ready in time")
        if self._failure is not None:
            raise RuntimeError(f"MCP server failed to start: {self._failure}") from self._failure
        return self

    async def _serve(self) -> None:
        self._stop = asyncio.Event()
        params = StdioServerParameters(
            command=self.command[0], args=self.command[1:], env=self.env or None
        )
        try:
            async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                self._tools = [
                    ToolSpec(t.name, t.description or "", t.input_schema) for t in listed.tools
                ]
                self._session = session
                self._ready.set()
                await self._stop.wait()
        except BaseException as exc:
            self._failure = exc
            print(
                f"MCP SERVER STARTUP FAILURE: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )

            error_file = "/tmp/mcp-server-error.log"
            if os.path.exists(error_file):
                try:
                    with open(error_file, encoding="utf-8") as f:
                        child_error = f.read()
                    print("MCP CHILD PROCESS ERROR:", file=sys.stderr, flush=True)
                    print(child_error, file=sys.stderr, flush=True)
                except Exception as log_exc:
                    print(f"Could not read MCP child error: {log_exc}", file=sys.stderr, flush=True)
            self._ready.set()
            raise
        finally:
            self._session = None

    @property
    def tools(self) -> list[ToolSpec]:
        return list(self._tools)

    def tool_names(self) -> list[str]:
        return [t.name for t in self._tools]

    def call_tool(self, name: str, arguments: dict) -> ToolResult:
        if self._session is None:
            raise RuntimeError("MCP client is not started")
        started = time.perf_counter()
        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, arguments, read_timeout_seconds=self.call_timeout),
            self._loop,
        )
        clean_args = {
            k: v
            for k, v in arguments.items()
            if isinstance(v, str | int | float | bool | type(None))
        }
        try:
            result = future.result(timeout=self.call_timeout + 5)
        except Exception as exc:
            return ToolResult(
                call=ToolCall(
                    name=name,
                    arguments=clean_args,
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            )
        latency_ms = (time.perf_counter() - started) * 1000
        text = "\n".join(
            block.text for block in result.content if isinstance(block, types.TextContent)
        )
        if result.is_error:
            return ToolResult(
                call=ToolCall(
                    name=name, arguments=clean_args, ok=False, error=text, latency_ms=latency_ms
                ),
                text=text,
            )
        return ToolResult(
            call=ToolCall(name=name, arguments=clean_args, ok=True, latency_ms=latency_ms),
            structured=result.structured_content,
            text=text,
        )

    def close(self) -> None:
        if self._stop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._stop.set)
        # Give the session a moment to unwind before stopping the loop.
        deadline = time.perf_counter() + 5
        while self._session is not None and time.perf_counter() < deadline:
            time.sleep(0.05)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

