"""Read-only access to live inventory and catalog state.

Three backends implement the same four queries:
- MySQL for local development
- Postgres for the existing deployment/setup
- SQLite for tests and single-container usage

All backends are seeded from the same generated JSON, so they expose
the same inventory and catalog data.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Protocol

import mysql.connector
from pydantic import BaseModel, ConfigDict, Field


class InventoryRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    sku: str
    product_name: str
    store_id: str
    store_name: str
    on_hand_units: int = Field(ge=0)
    reserved_units: int = Field(ge=0)
    available_units: int = Field(ge=0)
    reorder_point: int = Field(ge=0)
    below_reorder_point: bool
    in_stock: bool
    last_restocked_at: str


class CatalogHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    sku: str
    name: str
    brand: str
    category: str
    grammage: str
    price_inr: float
    mrp_inr: float
    storage: str
    diet_tags: list[str]
    is_substitutable: bool


class StoreRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    store_id: str
    name: str
    city: str


class InventoryRepository(Protocol):
    def get_inventory(
        self,
        sku: str,
        store_id: str,
    ) -> InventoryRecord | None: ...

    def search_catalog(
        self,
        query: str,
        limit: int,
    ) -> list[CatalogHit]: ...

    def list_stores(self) -> list[StoreRecord]: ...

    def resolve_sku(
        self,
        name_or_sku: str,
    ) -> str | None: ...


SCHEMA = """
CREATE TABLE IF NOT EXISTS dark_stores (
    store_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT NOT NULL,
    state TEXT NOT NULL,
    service_radius_km REAL NOT NULL,
    opens_at TEXT NOT NULL,
    closes_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    sku TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    brand TEXT NOT NULL,
    category TEXT NOT NULL,
    grammage TEXT NOT NULL,
    price_inr REAL NOT NULL,
    mrp_inr REAL NOT NULL,
    storage TEXT NOT NULL,
    diet_tags TEXT NOT NULL,
    is_substitutable BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS inventory (
    sku TEXT NOT NULL REFERENCES products(sku),
    store_id TEXT NOT NULL REFERENCES dark_stores(store_id),
    on_hand_units INTEGER NOT NULL,
    reserved_units INTEGER NOT NULL,
    reorder_point INTEGER NOT NULL,
    last_restocked_at TEXT NOT NULL,
    PRIMARY KEY (sku, store_id)
);
"""


_INVENTORY_SQL = """
SELECT
    i.sku,
    p.name AS product_name,
    i.store_id,
    s.name AS store_name,
    i.on_hand_units,
    i.reserved_units,
    i.reorder_point,
    i.last_restocked_at
FROM inventory i
JOIN products p ON p.sku = i.sku
JOIN dark_stores s ON s.store_id = i.store_id
WHERE i.sku = {p} AND i.store_id = {p}
"""


_SEARCH_SQL = """
SELECT
    sku,
    name,
    brand,
    category,
    grammage,
    price_inr,
    mrp_inr,
    storage,
    diet_tags,
    is_substitutable
FROM products
WHERE LOWER(name) LIKE {p}
   OR LOWER(brand) LIKE {p}
   OR LOWER(category) LIKE {p}
   OR LOWER(sku) LIKE {p}
ORDER BY name
LIMIT {p}
"""


_STORES_SQL = """
SELECT store_id, name, city
FROM dark_stores
ORDER BY store_id
"""


_RESOLVE_SQL = """
SELECT sku
FROM products
WHERE LOWER(sku) = {p}
   OR LOWER(name) = {p}
LIMIT 1
"""


def _inventory_record(row: dict) -> InventoryRecord:
    available = max(
        0,
        row["on_hand_units"] - row["reserved_units"],
    )

    return InventoryRecord(
        sku=row["sku"],
        product_name=row["product_name"],
        store_id=row["store_id"],
        store_name=row["store_name"],
        on_hand_units=row["on_hand_units"],
        reserved_units=row["reserved_units"],
        available_units=available,
        reorder_point=row["reorder_point"],
        below_reorder_point=(row["on_hand_units"] <= row["reorder_point"]),
        in_stock=available > 0,
        last_restocked_at=str(row["last_restocked_at"]),
    )


def _catalog_hit(row: dict) -> CatalogHit:
    return CatalogHit(
        sku=row["sku"],
        name=row["name"],
        brand=row["brand"],
        category=row["category"],
        grammage=row["grammage"],
        price_inr=float(row["price_inr"]),
        mrp_inr=float(row["mrp_inr"]),
        storage=row["storage"],
        diet_tags=json.loads(row["diet_tags"]),
        is_substitutable=bool(row["is_substitutable"]),
    )


def seed_rows(catalog_dir: Path) -> dict[str, list[tuple]]:
    """Flatten generated JSON into insert-ready rows."""

    stores = json.loads((catalog_dir / "dark_stores.json").read_text(encoding="utf-8"))

    products = json.loads((catalog_dir / "products.json").read_text(encoding="utf-8"))

    inventory = json.loads((catalog_dir / "inventory.json").read_text(encoding="utf-8"))

    return {
        "dark_stores": [
            (
                s["store_id"],
                s["name"],
                s["city"],
                s["state"],
                s["service_radius_km"],
                s["opens_at"],
                s["closes_at"],
            )
            for s in stores
        ],
        "products": [
            (
                p["sku"],
                p["name"],
                p["brand"],
                p["category"],
                p["grammage"],
                p["price_inr"],
                p["mrp_inr"],
                p["storage"],
                json.dumps(p["diet_tags"]),
                p["is_substitutable"],
            )
            for p in products
        ],
        "inventory": [
            (
                r["sku"],
                r["store_id"],
                r["on_hand_units"],
                r["reserved_units"],
                r["reorder_point"],
                r["last_restocked_at"],
            )
            for r in inventory
        ],
    }


class SqliteRepository:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(
            self.path,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self._conn.close()

    def seed(self, catalog_dir: Path) -> int:
        rows = seed_rows(catalog_dir)

        with self._conn:
            self._conn.executescript(SCHEMA)

            for table in (
                "inventory",
                "products",
                "dark_stores",
            ):
                self._conn.execute(f"DELETE FROM {table}")

            self._conn.executemany(
                "INSERT INTO dark_stores VALUES (?,?,?,?,?,?,?)",
                rows["dark_stores"],
            )

            self._conn.executemany(
                "INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?,?)",
                rows["products"],
            )

            self._conn.executemany(
                "INSERT INTO inventory VALUES (?,?,?,?,?,?)",
                rows["inventory"],
            )

        return sum(len(v) for v in rows.values())

    def get_inventory(
        self,
        sku: str,
        store_id: str,
    ) -> InventoryRecord | None:
        row = self._conn.execute(
            _INVENTORY_SQL.format(p="?"),
            (sku, store_id),
        ).fetchone()

        return _inventory_record(dict(row)) if row else None

    def search_catalog(
        self,
        query: str,
        limit: int,
    ) -> list[CatalogHit]:
        like = f"%{query.strip().lower()}%"

        rows = self._conn.execute(
            _SEARCH_SQL.format(p="?"),
            (like, like, like, like, limit),
        ).fetchall()

        return [_catalog_hit(dict(row)) for row in rows]

    def list_stores(self) -> list[StoreRecord]:
        rows = self._conn.execute(_STORES_SQL).fetchall()

        return [StoreRecord(**dict(row)) for row in rows]

    def resolve_sku(
        self,
        name_or_sku: str,
    ) -> str | None:
        key = name_or_sku.strip().lower()

        row = self._conn.execute(
            _RESOLVE_SQL.format(p="?"),
            (key, key),
        ).fetchone()

        return row["sku"] if row else None


class MySQLRepository:
    """MySQL implementation of the inventory repository."""

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
    ) -> None:
        self.config = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
            "ssl_disabled": False,
        }

        self._conn = mysql.connector.connect(**self.config)

    def close(self) -> None:
        self._conn.close()

    def ping(self) -> bool:
        cursor = self._conn.cursor()

        try:
            cursor.execute("SELECT 1")
            return cursor.fetchone() is not None
        finally:
            cursor.close()

    def seed(self, catalog_dir: Path) -> int:
        rows = seed_rows(catalog_dir)

        cursor = self._conn.cursor()

        try:
            # MySQL schema
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS dark_stores (
                    store_id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    city VARCHAR(255) NOT NULL,
                    state VARCHAR(255) NOT NULL,
                    service_radius_km DOUBLE NOT NULL,
                    opens_at VARCHAR(50) NOT NULL,
                    closes_at VARCHAR(50) NOT NULL
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS products (
                    sku VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    brand VARCHAR(255) NOT NULL,
                    category VARCHAR(255) NOT NULL,
                    grammage VARCHAR(100) NOT NULL,
                    price_inr DOUBLE NOT NULL,
                    mrp_inr DOUBLE NOT NULL,
                    storage VARCHAR(100) NOT NULL,
                    diet_tags TEXT NOT NULL,
                    is_substitutable BOOLEAN NOT NULL
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS inventory (
                    sku VARCHAR(255) NOT NULL,
                    store_id VARCHAR(255) NOT NULL,
                    on_hand_units INT NOT NULL,
                    reserved_units INT NOT NULL,
                    reorder_point INT NOT NULL,
                    last_restocked_at VARCHAR(100) NOT NULL,
                    PRIMARY KEY (sku, store_id),
                    FOREIGN KEY (sku)
                        REFERENCES products(sku),
                    FOREIGN KEY (store_id)
                        REFERENCES dark_stores(store_id)
                )
                """
            )

            # Clear existing data before reseeding.
            for table in (
                "inventory",
                "products",
                "dark_stores",
            ):
                cursor.execute(f"DELETE FROM {table}")

            cursor.executemany(
                """
                INSERT INTO dark_stores
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                """,
                rows["dark_stores"],
            )

            cursor.executemany(
                """
                INSERT INTO products
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                rows["products"],
            )

            cursor.executemany(
                """
                INSERT INTO inventory
                VALUES (%s,%s,%s,%s,%s,%s)
                """,
                rows["inventory"],
            )

            self._conn.commit()

            return sum(len(v) for v in rows.values())

        except Exception:
            self._conn.rollback()
            raise

        finally:
            cursor.close()

    def get_inventory(
        self,
        sku: str,
        store_id: str,
    ) -> InventoryRecord | None:
        cursor = self._conn.cursor(dictionary=True)

        try:
            cursor.execute(
                _INVENTORY_SQL.format(p="%s"),
                (sku, store_id),
            )

            row = cursor.fetchone()

            return _inventory_record(row) if row else None

        finally:
            cursor.close()

    def search_catalog(
        self,
        query: str,
        limit: int,
    ) -> list[CatalogHit]:
        like = f"%{query.strip().lower()}%"

        cursor = self._conn.cursor(dictionary=True)

        try:
            cursor.execute(
                _SEARCH_SQL.format(p="%s"),
                (
                    like,
                    like,
                    like,
                    like,
                    limit,
                ),
            )

            rows = cursor.fetchall()

            return [_catalog_hit(row) for row in rows]

        finally:
            cursor.close()

    def list_stores(self) -> list[StoreRecord]:
        cursor = self._conn.cursor(dictionary=True)

        try:
            cursor.execute(_STORES_SQL)

            rows = cursor.fetchall()

            return [StoreRecord(**row) for row in rows]

        finally:
            cursor.close()

    def resolve_sku(
        self,
        name_or_sku: str,
    ) -> str | None:
        key = name_or_sku.strip().lower()

        cursor = self._conn.cursor(dictionary=True)

        try:
            cursor.execute(
                _RESOLVE_SQL.format(p="%s"),
                (key, key),
            )

            row = cursor.fetchone()

            return row["sku"] if row else None

        finally:
            cursor.close()
