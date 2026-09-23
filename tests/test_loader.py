import json

import pytest

from qc_copilot.config import get_settings
from qc_copilot.ingest.loader import (
    load_all_documents,
    load_catalog_documents,
    load_policy_documents,
    parse_frontmatter,
)
from qc_copilot.models import SourceKind


@pytest.fixture(scope="module")
def settings():
    return get_settings()


def test_parse_frontmatter_splits_metadata_from_body():
    meta, body = parse_frontmatter("---\ndoc_id: p1\ntitle: A Title\n---\n\n# Heading\n\nBody.")
    assert meta["doc_id"] == "p1"
    assert meta["title"] == "A Title"
    assert body.startswith("# Heading")


def test_parse_frontmatter_without_a_block_returns_the_whole_body():
    meta, body = parse_frontmatter("# Heading\n\nBody.")
    assert meta == {}
    assert body == "# Heading\n\nBody."


def test_parse_frontmatter_rejects_a_non_mapping_block():
    with pytest.raises(ValueError):
        parse_frontmatter("---\n- one\n- two\n---\nbody")


def test_policy_documents_use_their_declared_doc_ids(settings):
    docs = load_policy_documents(settings.policies_dir)
    doc_ids = {d.doc_id for d in docs}
    expected = {"policy-returns", "policy-delivery", "policy-substitutions", "policy-store-ops"}
    assert expected <= doc_ids
    assert all(d.kind is SourceKind.POLICY for d in docs)
    assert all(d.metadata["synthetic"] is True for d in docs)


def test_policy_body_excludes_the_frontmatter(settings):
    docs = load_policy_documents(settings.policies_dir)
    assert all("doc_id:" not in d.text for d in docs)


def test_product_documents_state_price_pack_and_storage(settings):
    docs = {d.doc_id: d for d in load_catalog_documents(settings.catalog_dir)}
    doc = docs["product-QC-DAIRY--NANDH-PANEER-200"]
    assert "INR 83.60" in doc.text
    assert "INR 95.00" in doc.text
    assert "200 g" in doc.text
    assert "chilled" in doc.text


def test_product_document_marks_a_non_substitutable_item(settings):
    docs = {d.doc_id: d for d in load_catalog_documents(settings.catalog_dir)}
    doc = docs["product-QC-BABY-C-TINYT-BABY-DIAPE-32"]
    assert "excluded from substitution" in doc.text


def test_dark_store_documents_list_pincodes_and_hours(settings):
    docs = {d.doc_id: d for d in load_catalog_documents(settings.catalog_dir)}
    doc = docs["store-DS-DEL-001"]
    assert "110017" in doc.text
    assert "07:00" in doc.text
    assert "3.5 km" in doc.text


def test_live_state_is_never_placed_in_the_knowledge_base(settings):
    """Inventory counts and order ids must not be indexed, or answers go stale."""
    orders = json.loads((settings.catalog_dir / "orders.json").read_text())
    order_ids = {o["order_id"] for o in orders}
    documents = load_all_documents(settings.policies_dir, settings.catalog_dir)
    corpus = "\n".join(d.text for d in documents)
    assert not any(order_id in corpus for order_id in order_ids)
    assert "on_hand_units" not in corpus
    assert "reserved_units" not in corpus


def test_every_product_and_store_becomes_exactly_one_document(settings):
    products = json.loads((settings.catalog_dir / "products.json").read_text())
    stores = json.loads((settings.catalog_dir / "dark_stores.json").read_text())
    docs = load_catalog_documents(settings.catalog_dir)
    assert len(docs) == len(products) + len(stores)
    assert len({d.doc_id for d in docs}) == len(docs)
