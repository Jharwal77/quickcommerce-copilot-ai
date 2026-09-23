"""End-to-end test of the MCP tools over a real stdio subprocess."""

import os
import sys

import pytest

from qc_copilot.config import get_settings
from qc_copilot.mcp_server.client import MCPToolClient
from qc_copilot.tools.repository import SqliteRepository


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db = tmp_path_factory.mktemp("mcp") / "darkstore.sqlite"
    repo = SqliteRepository(db)
    repo.seed(get_settings().catalog_dir)
    repo.close()
    env = {**os.environ, "TOOLS_BACKEND": "sqlite", "SQLITE_PATH": str(db)}
    mcp = MCPToolClient(
        command=[sys.executable, "-m", "qc_copilot.mcp_server.server"], env=env
    ).start()
    yield mcp
    mcp.close()


def test_server_advertises_exactly_the_two_read_only_tools(client):
    assert client.tool_names() == ["get_inventory", "search_catalog"]


def test_tool_schemas_convert_to_openai_function_format(client):
    spec = next(t for t in client.tools if t.name == "get_inventory")
    tool = spec.as_openai_tool()
    assert tool["type"] == "function"
    assert set(tool["function"]["parameters"]["properties"]) == {"sku", "dark_store"}
    assert tool["function"]["parameters"]["required"] == ["sku", "dark_store"]


def test_get_inventory_returns_structured_stock(client):
    result = client.call_tool(
        "get_inventory", {"sku": "QC-DAIRY--NANDH-PANEER-200", "dark_store": "DS-BLR-001"}
    )
    assert result.call.ok is True
    assert result.call.name == "get_inventory"
    assert result.structured["found"] is True
    assert result.structured["record"]["available_units"] == 35
    assert result.call.latency_ms > 0


def test_get_inventory_reports_an_unknown_product_without_erroring(client):
    result = client.call_tool("get_inventory", {"sku": "unobtainium", "dark_store": "DS-BLR-001"})
    assert result.call.ok is True
    assert result.structured["found"] is False
    assert "search_catalog" in result.structured["message"]


def test_schema_violations_surface_as_tool_errors(client):
    result = client.call_tool("get_inventory", {"sku": "x", "dark_store": "DS-BLR-001"})
    assert result.call.ok is False
    assert "at least 3 characters" in (result.call.error or "")


def test_unknown_tool_is_an_error_not_an_exception(client):
    result = client.call_tool("delete_everything", {})
    assert result.call.ok is False


def test_search_catalog_lists_stores_for_follow_up_calls(client):
    result = client.call_tool("search_catalog", {"query": "paneer", "limit": 5})
    assert result.call.ok is True
    assert result.structured["count"] == 3
    assert {s["store_id"] for s in result.structured["stores"]} >= {"DS-BLR-001", "DS-MUM-002"}
    assert "Nandhini Fresh Paneer 200 g" in result.as_model_text()


def test_non_serialisable_arguments_are_dropped_from_the_record(client):
    result = client.call_tool("search_catalog", {"query": "paneer", "limit": 2, "nested": {"a": 1}})
    assert "nested" not in result.call.arguments
