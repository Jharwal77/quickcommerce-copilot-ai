"""MCP server exposing read-only inventory and catalog tools.

Every tool takes a Pydantic-validated input and returns a Pydantic model, so the
schema the agent sees is generated from the same types that validate the data. The
tools never write. The server speaks MCP over stdio, which is how the agent connects.

Usage:
    python -m qc_copilot.mcp_server.server
"""

from __future__ import annotations

from typing import Annotated

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, ConfigDict, Field

from qc_copilot.config import Settings, get_settings
from qc_copilot.tools.repository import (
    CatalogHit,
    InventoryRecord,
    InventoryRepository,
    MySQLRepository,
    SqliteRepository,
    StoreRecord,
)


class InventoryResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    found: bool
    record: InventoryRecord | None = None
    message: str


class CatalogSearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    count: int
    hits: list[CatalogHit]
    stores: list[StoreRecord] = Field(
        default_factory=list,
        description="Dark stores whose identifiers can be used with get_inventory.",
    )


def build_repository(settings: Settings) -> InventoryRepository:
    if settings.tools_backend == "sqlite":
        return SqliteRepository(settings.sqlite_path)

    return MySQLRepository(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_database,
    )


def create_server(repository: InventoryRepository) -> MCPServer:
    server = MCPServer(
        "darkstore-tools",
        instructions=(
            "Read-only live data for Nimbus Now dark stores. Use search_catalog to find a "
            "SKU and the list of store identifiers, then get_inventory for stock at one store."
        ),
    )

    @server.tool(
        name="get_inventory",
        description=(
            "Live stock for one SKU at one dark store. Returns on-hand, reserved, and "
            "available units, whether the item is in stock, and whether it is at or below "
            "its reorder point. Requires the exact SKU (or exact product name) and the "
            "store identifier such as DS-BLR-001."
        ),
    )
    def get_inventory(
        sku: Annotated[
            str,
            Field(
                min_length=3,
                max_length=80,
                description="SKU or exact product name",
            ),
        ],
        dark_store: Annotated[
            str,
            Field(
                min_length=3,
                max_length=40,
                description="Store identifier, e.g. DS-BLR-001",
            ),
        ],
    ) -> InventoryResult:
        resolved = repository.resolve_sku(sku)

        if resolved is None:
            return InventoryResult(
                found=False,
                message=(f"No product matches '{sku}'. Use search_catalog to find the SKU."),
            )

        record = repository.get_inventory(
            resolved,
            dark_store.strip().upper(),
        )

        if record is None:
            return InventoryResult(
                found=False,
                message=(f"No inventory row for {resolved} at store '{dark_store}'."),
            )

        return InventoryResult(
            found=True,
            record=record,
            message=(
                f"{record.product_name} at {record.store_name}: "
                f"{record.available_units} available "
                f"({record.on_hand_units} on hand, "
                f"{record.reserved_units} reserved)."
            ),
        )

    @server.tool(
        name="search_catalog",
        description=(
            "Search the product catalog by name, brand, category, or SKU fragment. "
            "Returns matching products with SKU, price, and storage class, plus "
            "the list of dark store identifiers."
        ),
    )
    def search_catalog(
        query: Annotated[
            str,
            Field(min_length=2, max_length=120),
        ],
        limit: Annotated[
            int,
            Field(ge=1, le=25),
        ] = 10,
    ) -> CatalogSearchResult:
        hits = repository.search_catalog(query, limit)

        return CatalogSearchResult(
            query=query,
            count=len(hits),
            hits=hits,
            stores=repository.list_stores(),
        )

    return server


def main() -> None:
    settings = get_settings()
    print(f'MCP START: backend={settings.tools_backend}', file=__import__('sys').stderr, flush=True)
    repository = build_repository(settings)
    print('MCP REPOSITORY: connected', file=__import__('sys').stderr, flush=True)

    try:
        create_server(repository).run(transport="stdio")
    finally:
        repository.close()


if __name__ == "__main__":
    main()
