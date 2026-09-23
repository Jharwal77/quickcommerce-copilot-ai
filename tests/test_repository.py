import pytest

from qc_copilot.config import get_settings
from qc_copilot.tools.repository import (
    InventoryRecord,
    SqliteRepository,
    seed_rows,
)

SETTINGS = get_settings()


@pytest.fixture(scope="module")
def sqlite_repo(tmp_path_factory):
    repo = SqliteRepository(tmp_path_factory.mktemp("db") / "darkstore.sqlite")
    repo.seed(SETTINGS.catalog_dir)
    yield repo
    repo.close()


def test_seed_rows_cover_every_generated_record():
    rows = seed_rows(SETTINGS.catalog_dir)
    assert len(rows["dark_stores"]) == 6
    assert len(rows["products"]) == 88
    assert len(rows["inventory"]) == 6 * 88


def test_seed_is_idempotent(sqlite_repo):
    first = sqlite_repo.seed(SETTINGS.catalog_dir)
    second = sqlite_repo.seed(SETTINGS.catalog_dir)
    assert first == second == 6 + 88 + 6 * 88
    assert len(sqlite_repo.list_stores()) == 6


def test_get_inventory_derives_availability_from_on_hand_minus_reserved(sqlite_repo):
    record = sqlite_repo.get_inventory("QC-DAIRY--NANDH-PANEER-200", "DS-BLR-001")
    assert isinstance(record, InventoryRecord)
    assert record.on_hand_units == 40
    assert record.reserved_units == 5
    assert record.available_units == 35
    assert record.in_stock is True
    assert record.below_reorder_point is False
    assert record.store_name == "Indiranagar Dark Store"


def test_get_inventory_marks_zero_on_hand_as_out_of_stock(sqlite_repo):
    record = sqlite_repo.get_inventory("QC-DAIRY--AMULY-SALTED-BUT-500", "DS-BLR-001")
    assert record is not None
    assert record.on_hand_units == 0
    assert record.in_stock is False
    assert record.below_reorder_point is True


def test_get_inventory_returns_none_for_unknown_pairs(sqlite_repo):
    assert sqlite_repo.get_inventory("QC-NOPE", "DS-BLR-001") is None
    assert sqlite_repo.get_inventory("QC-DAIRY--NANDH-PANEER-200", "DS-NOPE") is None


def test_search_catalog_is_case_insensitive_and_bounded(sqlite_repo):
    hits = sqlite_repo.search_catalog("PANEER", limit=10)
    names = [h.name for h in hits]
    assert "Nandhini Fresh Paneer 200 g" in names
    assert "Chefsnap Paneer Butter Masala Ready Meal 285 g" in names
    assert len(sqlite_repo.search_catalog("a", limit=3)) == 3


def test_search_catalog_matches_brand_and_category(sqlite_repo):
    assert all(h.brand == "Tinytots" for h in sqlite_repo.search_catalog("tinytots", 20))
    assert {h.category for h in sqlite_repo.search_catalog("baby care", 20)} == {"Baby Care"}


def test_search_catalog_decodes_diet_tags(sqlite_repo):
    (hit,) = [h for h in sqlite_repo.search_catalog("roasted almonds 500", 5)]
    assert hit.diet_tags == ["vegetarian", "gluten-free"]
    assert hit.is_substitutable is True


def test_resolve_sku_accepts_sku_or_exact_name(sqlite_repo):
    assert sqlite_repo.resolve_sku("qc-dairy--nandh-paneer-200") == "QC-DAIRY--NANDH-PANEER-200"
    assert sqlite_repo.resolve_sku("Nandhini Fresh Paneer 200 g") == "QC-DAIRY--NANDH-PANEER-200"
    assert sqlite_repo.resolve_sku("paneer") is None
