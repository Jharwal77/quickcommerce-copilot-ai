"""Load policy markdown and generated catalog JSON into KBDocument objects.

Live, mutable state deliberately stays out of the knowledge base. Inventory counts
and order status change by the minute, so embedding them would guarantee stale
answers. Those are served by the read-only tools instead; the vector store holds only
descriptions and policies, which are stable.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from qc_copilot.models import KBDocument, SourceKind

_FRONTMATTER = "---"


def parse_frontmatter(raw: str) -> tuple[dict[str, object], str]:
    """Split a leading YAML frontmatter block from the markdown body."""
    if not raw.startswith(_FRONTMATTER):
        return {}, raw.strip()

    parts = raw.split(_FRONTMATTER, 2)
    if len(parts) < 3:
        return {}, raw.strip()

    meta = yaml.safe_load(parts[1]) or {}
    if not isinstance(meta, dict):
        raise ValueError("frontmatter must be a mapping")
    return meta, parts[2].strip()


def load_policy_documents(policies_dir: Path) -> list[KBDocument]:
    documents: list[KBDocument] = []
    for path in sorted(policies_dir.glob("*.md")):
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        doc_id = str(meta.get("doc_id") or path.stem)
        documents.append(
            KBDocument(
                doc_id=doc_id,
                title=str(meta.get("title") or path.stem.replace("-", " ").title()),
                kind=SourceKind.POLICY,
                text=body,
                metadata={
                    "source_path": str(path.name),
                    "version": str(meta.get("version", "")),
                    "effective_date": str(meta.get("effective_date", "")),
                    "operator": str(meta.get("operator", "")),
                    "synthetic": True,
                },
            )
        )
    return documents


def _render_product(product: dict) -> str:
    diet = ", ".join(product["diet_tags"]) or "no dietary tags"
    substitutes = ", ".join(product["substitute_skus"]) or "none listed"
    discount = product["mrp_inr"] - product["price_inr"]
    lines = [
        f"{product['name']} is a {product['category']} product sold by Nimbus Now "
        f"under SKU {product['sku']}.",
        f"Brand: {product['brand']}. Pack size: {product['grammage']}.",
        f"Maximum retail price is INR {product['mrp_inr']:.2f} and the current selling "
        f"price is INR {product['price_inr']:.2f}"
        + (f", a discount of INR {discount:.2f}." if discount > 0 else "."),
        f"Storage class: {product['storage']}. Stated shelf life: "
        f"{product['shelf_life_days']} days. Dietary attributes: {diet}.",
        (
            f"This product may be substituted. Listed substitute SKUs: {substitutes}."
            if product["is_substitutable"]
            else "This product is excluded from substitution."
        ),
    ]
    return "\n\n".join(lines)


def _render_dark_store(store: dict) -> str:
    pincodes = ", ".join(store["serviceable_pincodes"])
    return "\n\n".join(
        [
            f"{store['name']} is a Nimbus Now dark store with identifier "
            f"{store['store_id']}, located in {store['city']}, {store['state']}.",
            f"It serves the pincodes {pincodes} within a service radius of "
            f"{store['service_radius_km']} km.",
            f"Operating hours are {store['opens_at']} to {store['closes_at']} local time.",
            f"Average pick and pack time is {store['avg_pick_pack_minutes']} minutes and "
            f"the store operates a rider pool of {store['rider_count']} riders.",
        ]
    )


def load_catalog_documents(catalog_dir: Path) -> list[KBDocument]:
    documents: list[KBDocument] = []

    products = json.loads((catalog_dir / "products.json").read_text(encoding="utf-8"))
    for product in products:
        documents.append(
            KBDocument(
                doc_id=f"product-{product['sku']}",
                title=product["name"],
                kind=SourceKind.PRODUCT,
                text=_render_product(product),
                metadata={
                    "sku": product["sku"],
                    "brand": product["brand"],
                    "category": product["category"],
                    "storage": product["storage"],
                    "price_inr": product["price_inr"],
                    "synthetic": True,
                },
            )
        )

    stores = json.loads((catalog_dir / "dark_stores.json").read_text(encoding="utf-8"))
    for store in stores:
        documents.append(
            KBDocument(
                doc_id=f"store-{store['store_id']}",
                title=store["name"],
                kind=SourceKind.DARK_STORE,
                text=_render_dark_store(store),
                metadata={
                    "store_id": store["store_id"],
                    "city": store["city"],
                    "synthetic": True,
                },
            )
        )

    return documents


def load_all_documents(policies_dir: Path, catalog_dir: Path) -> list[KBDocument]:
    return load_policy_documents(policies_dir) + load_catalog_documents(catalog_dir)
