"""Generate the synthetic quick-commerce dataset used by the knowledge base and tools.

Every value here is invented. Brand names are fictional so that no output can be
mistaken for a real company's catalog, inventory, or order history. The script is
deterministic: the same seed always produces byte-identical files.

Usage:
    python scripts/generate_data.py [--seed 20240501] [--out data]
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

SEED = 20240501

# Fixed clock so that generated order timestamps and ETAs are reproducible.
NOW = datetime(2026, 3, 14, 9, 30, tzinfo=UTC)

CITIES = [
    ("Bengaluru", "KA"),
    ("Delhi", "DL"),
    ("Mumbai", "MH"),
]

# The catalog and store tables are kept one row per record. The formatter would
# explode each tuple across eight lines, which makes the tables far harder to scan.
# fmt: off
# (store_code, display_name, city_index, pincodes, radius_km, opens, closes)
DARK_STORES = [
    ("DS-BLR-001", "Indiranagar Dark Store", 0, ["560038", "560008", "560071"], 3.0, "06:00", "23:59"),
    ("DS-BLR-002", "Koramangala Dark Store", 0, ["560034", "560095", "560047"], 2.5, "06:00", "23:59"),
    ("DS-DEL-001", "Saket Dark Store", 1, ["110017", "110019", "110048"], 3.5, "07:00", "23:00"),
    ("DS-DEL-002", "Rohini Dark Store", 1, ["110085", "110089"], 4.0, "07:00", "22:30"),
    ("DS-MUM-001", "Andheri West Dark Store", 2, ["400053", "400058", "400102"], 2.5, "06:30", "23:30"),
    ("DS-MUM-002", "Powai Dark Store", 2, ["400076", "400072"], 3.0, "07:00", "23:00"),
]

# (category, [(brand, item, unit, [sizes], low_price, high_price, storage, diet_tags)])
CATALOG_TEMPLATES: list[tuple[str, list[tuple]]] = [
    (
        "Fruits & Vegetables",
        [
            ("Farmlyft", "Tomato Hybrid", "g", [500, 1000], 22, 48, "ambient", ["vegetarian"]),
            ("Farmlyft", "Onion", "g", [1000, 2000], 30, 72, "ambient", ["vegetarian"]),
            ("Farmlyft", "Banana Robusta", "g", [500, 1000], 34, 62, "ambient", ["vegetarian"]),
            ("Greengram", "Baby Spinach", "g", [100, 250], 25, 49, "chilled", ["vegetarian"]),
            ("Greengram", "Cherry Tomato", "g", [200, 400], 45, 85, "chilled", ["vegetarian"]),
            ("Farmlyft", "Alphonso Mango", "g", [1000], 240, 420, "ambient", ["vegetarian", "seasonal"]),
        ],
    ),
    (
        "Dairy & Eggs",
        [
            ("Nandhini Fresh", "Toned Milk", "ml", [500, 1000], 27, 56, "chilled", ["vegetarian"]),
            ("Nandhini Fresh", "Full Cream Milk", "ml", [500, 1000], 33, 68, "chilled", ["vegetarian"]),
            ("Nandhini Fresh", "Curd", "g", [400, 1000], 35, 82, "chilled", ["vegetarian"]),
            ("Nandhini Fresh", "Paneer", "g", [200, 400], 95, 180, "chilled", ["vegetarian"]),
            ("Amulya Gold", "Salted Butter", "g", [100, 500], 62, 285, "chilled", ["vegetarian"]),
            ("Yolkwell", "Farm Eggs", "piece", [6, 12], 48, 92, "chilled", ["non-vegetarian"]),
        ],
    ),
    (
        "Beverages",
        [
            ("Zestara", "Orange Juice No Added Sugar", "ml", [200, 1000], 35, 149, "chilled", ["vegetarian"]),
            ("Zestara", "Cold Brew Coffee", "ml", [200, 500], 89, 199, "chilled", ["vegetarian"]),
            ("Fizzko", "Lemon Soda", "ml", [250, 750], 25, 65, "ambient", ["vegetarian"]),
            ("Chaiwala Co", "Masala Chai Premix", "g", [200, 500], 110, 245, "ambient", ["vegetarian"]),
            ("Hydralite", "Electrolyte Drink Mix", "g", [100, 250], 130, 290, "ambient", ["vegetarian"]),
        ],
    ),
    (
        "Snacks & Namkeen",
        [
            ("Crispora", "Salted Potato Chips", "g", [52, 90], 20, 45, "ambient", ["vegetarian"]),
            ("Crispora", "Peri Peri Potato Chips", "g", [52, 90], 20, 45, "ambient", ["vegetarian"]),
            ("Bhujiawala", "Aloo Bhujia", "g", [200, 400], 55, 105, "ambient", ["vegetarian"]),
            ("Bhujiawala", "Moong Dal Namkeen", "g", [200, 400], 58, 112, "ambient", ["vegetarian"]),
            ("Nutlane", "Roasted Almonds", "g", [200, 500], 245, 585, "ambient", ["vegetarian", "gluten-free"]),
            ("Nutlane", "Trail Mix", "g", [200, 400], 210, 399, "ambient", ["vegetarian"]),
        ],
    ),
    (
        "Staples & Grains",
        [
            ("Annapurti", "Sona Masoori Rice", "kg", [5, 10], 385, 760, "ambient", ["vegetarian", "gluten-free"]),
            ("Annapurti", "Whole Wheat Atta", "kg", [5, 10], 240, 470, "ambient", ["vegetarian"]),
            ("Annapurti", "Toor Dal", "g", [500, 1000], 92, 178, "ambient", ["vegetarian", "gluten-free"]),
            ("Annapurti", "Chana Dal", "g", [500, 1000], 78, 152, "ambient", ["vegetarian", "gluten-free"]),
            ("Goldpress", "Sunflower Oil", "ml", [1000], 135, 165, "ambient", ["vegetarian"]),
            ("Goldpress", "Cold Pressed Groundnut Oil", "ml", [1000], 285, 340, "ambient", ["vegetarian"]),
        ],
    ),
    (
        "Personal Care",
        [
            ("Purelin", "Aloe Face Wash", "ml", [100, 200], 165, 295, "ambient", ["cruelty-free"]),
            ("Purelin", "Sunscreen SPF 50", "ml", [50, 100], 320, 545, "ambient", ["cruelty-free"]),
            ("Dentiva", "Fluoride Toothpaste", "g", [100, 200], 85, 158, "ambient", ["vegetarian"]),
            ("Silksoft", "Anti Dandruff Shampoo", "ml", [180, 340], 210, 385, "ambient", ["cruelty-free"]),
            ("Silksoft", "Body Lotion", "ml", [200, 400], 199, 365, "ambient", ["cruelty-free"]),
        ],
    ),
    (
        "Home & Cleaning",
        [
            ("Shinekit", "Dishwash Gel", "ml", [500, 1000], 105, 195, "ambient", []),
            ("Shinekit", "Floor Cleaner", "ml", [1000], 165, 210, "ambient", []),
            ("Shinekit", "Laundry Liquid", "ml", [1000, 2000], 285, 540, "ambient", []),
            ("Wrapwell", "Garbage Bags Medium", "piece", [30, 60], 95, 175, "ambient", []),
        ],
    ),
    (
        "Baby Care",
        [
            ("Tinytots", "Baby Diapers Medium", "piece", [32, 62], 549, 999, "ambient", []),
            ("Tinytots", "Baby Wipes", "piece", [72, 144], 145, 265, "ambient", []),
            ("Tinytots", "Baby Lotion", "ml", [100, 200], 185, 320, "ambient", []),
        ],
    ),
    (
        "Frozen & Ready to Eat",
        [
            ("Frostbite", "Green Peas Frozen", "g", [500, 1000], 82, 155, "frozen", ["vegetarian"]),
            ("Frostbite", "Mixed Vegetables Frozen", "g", [500, 1000], 88, 165, "frozen", ["vegetarian"]),
            ("Frostbite", "Aloo Tikki Frozen", "g", [400], 149, 179, "frozen", ["vegetarian"]),
            ("Chefsnap", "Paneer Butter Masala Ready Meal", "g", [285], 165, 199, "ambient", ["vegetarian"]),
        ],
    ),
    (
        "Bakery",
        [
            ("Ovenly", "Whole Wheat Bread", "g", [400], 48, 62, "ambient", ["vegetarian"]),
            ("Ovenly", "Multigrain Bread", "g", [400], 58, 75, "ambient", ["vegetarian"]),
            ("Ovenly", "Butter Croissant", "piece", [2, 4], 95, 185, "ambient", ["vegetarian"]),
        ],
    ),
]
# fmt: on

ORDER_STATUSES = [
    "PLACED",
    "PACKED",
    "OUT_FOR_DELIVERY",
    "DELIVERED",
    "CANCELLED",
    "RETURN_REQUESTED",
]

# Statuses that mean the order has not reached a terminal state yet.
IN_FLIGHT_STATUSES = frozenset({"PLACED", "PACKED", "OUT_FOR_DELIVERY"})

# Statuses that imply the order physically reached the customer.
DELIVERED_STATUSES = frozenset({"DELIVERED", "RETURN_REQUESTED"})


@dataclass(frozen=True)
class DarkStore:
    store_id: str
    name: str
    city: str
    state: str
    serviceable_pincodes: list[str]
    service_radius_km: float
    opens_at: str
    closes_at: str
    avg_pick_pack_minutes: int
    rider_count: int


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    brand: str
    category: str
    grammage: str
    pack_size: float
    pack_unit: str
    mrp_inr: float
    price_inr: float
    storage: str
    diet_tags: list[str]
    shelf_life_days: int
    is_substitutable: bool
    substitute_skus: list[str]


@dataclass(frozen=True)
class InventoryRow:
    sku: str
    store_id: str
    on_hand_units: int
    reserved_units: int
    reorder_point: int
    last_restocked_at: str


@dataclass(frozen=True)
class OrderItem:
    sku: str
    quantity: int
    unit_price_inr: float


@dataclass(frozen=True)
class Order:
    order_id: str
    store_id: str
    customer_pincode: str
    status: str
    placed_at: str
    promised_eta_minutes: int
    delivered_at: str | None
    items: list[dict]
    order_total_inr: float
    payment_mode: str
    substitution_applied: bool


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in text.upper()).strip("-")


def build_dark_stores(rng: random.Random) -> list[DarkStore]:
    stores: list[DarkStore] = []
    for store_id, name, city_idx, pincodes, radius, opens, closes in DARK_STORES:
        city, state = CITIES[city_idx]
        stores.append(
            DarkStore(
                store_id=store_id,
                name=name,
                city=city,
                state=state,
                serviceable_pincodes=pincodes,
                service_radius_km=radius,
                opens_at=opens,
                closes_at=closes,
                avg_pick_pack_minutes=rng.randint(3, 7),
                rider_count=rng.randint(8, 22),
            )
        )
    return stores


def build_products(rng: random.Random) -> list[Product]:
    products: list[Product] = []
    group_of: dict[str, tuple[str, str, str]] = {}
    by_group: dict[tuple[str, str, str], list[str]] = {}

    for category, items in CATALOG_TEMPLATES:
        for brand, item, unit, sizes, low, high, storage, diet in items:
            for size in sizes:
                grammage = f"{size} {unit}" if unit != "piece" else f"{size} pieces"
                sku = f"QC-{_slug(category)[:6]}-{_slug(brand)[:5]}-{_slug(item)[:10]}-{size}"
                span = high - low
                # Larger packs sit higher in the band, so unit economics stay sensible.
                position = 0.0 if len(sizes) == 1 else sizes.index(size) / (len(sizes) - 1)
                mrp = round(low + span * position, 2)
                discount = rng.choice([0.0, 0.03, 0.05, 0.08, 0.12, 0.15])
                price = round(mrp * (1 - discount), 2)
                products.append(
                    Product(
                        sku=sku,
                        name=f"{brand} {item} {grammage}",
                        brand=brand,
                        category=category,
                        grammage=grammage,
                        pack_size=float(size),
                        pack_unit=unit,
                        mrp_inr=mrp,
                        price_inr=price,
                        storage=storage,
                        diet_tags=list(diet),
                        shelf_life_days=_shelf_life(storage, category, rng),
                        is_substitutable=category != "Baby Care",
                        substitute_skus=[],
                    )
                )
                group_key = (category, brand, item)
                group_of[sku] = group_key
                by_group.setdefault(group_key, []).append(sku)

    # Substitutes are other pack sizes of the same category, brand, and item. Grouping on
    # the identity the SKU was built from, rather than matching on the display name, keeps
    # a ready meal from listing a raw ingredient as its substitute.
    resolved: list[Product] = []
    for product in products:
        siblings = [sku for sku in by_group[group_of[product.sku]] if sku != product.sku]
        resolved.append(Product(**{**asdict(product), "substitute_skus": sorted(siblings)[:2]}))
    return resolved


def _shelf_life(storage: str, category: str, rng: random.Random) -> int:
    if category == "Fruits & Vegetables":
        return rng.randint(2, 7)
    if category == "Bakery":
        return rng.randint(3, 6)
    if storage == "chilled":
        return rng.randint(5, 21)
    if storage == "frozen":
        return rng.randint(180, 365)
    return rng.randint(180, 540)


def build_inventory(
    products: list[Product], stores: list[DarkStore], rng: random.Random
) -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    for store in stores:
        for product in products:
            # Roughly one in nine store/SKU pairs is genuinely out of stock, which gives
            # the evaluation set real out-of-stock and substitution cases to answer.
            out_of_stock = rng.random() < 0.11
            on_hand = 0 if out_of_stock else rng.randint(4, 180)
            reserved = 0 if on_hand == 0 else rng.randint(0, min(6, on_hand))
            rows.append(
                InventoryRow(
                    sku=product.sku,
                    store_id=store.store_id,
                    on_hand_units=on_hand,
                    reserved_units=reserved,
                    reorder_point=rng.choice([6, 10, 12, 18, 24]),
                    last_restocked_at=(NOW - timedelta(hours=rng.randint(2, 96))).isoformat(),
                )
            )
    return rows


def build_orders(
    products: list[Product], stores: list[DarkStore], rng: random.Random, count: int = 40
) -> list[Order]:
    orders: list[Order] = []
    for index in range(1, count + 1):
        store = rng.choice(stores)
        status = rng.choices(ORDER_STATUSES, weights=[6, 6, 10, 60, 10, 8], k=1)[0]
        eta = rng.choice([8, 10, 12, 15, 18, 20, 25])
        # An order still in flight has to be minutes old, not days old, otherwise the
        # status and the timestamp contradict each other and no correct answer exists.
        if status in IN_FLIGHT_STATUSES:
            placed_at = NOW - timedelta(minutes=rng.randint(1, eta + 12))
        else:
            placed_at = NOW - timedelta(minutes=rng.randint(30, 60 * 24 * 9))
        picked = rng.sample(products, rng.randint(1, 5))
        items = [
            OrderItem(sku=p.sku, quantity=rng.randint(1, 3), unit_price_inr=p.price_inr)
            for p in picked
        ]
        total = round(sum(i.quantity * i.unit_price_inr for i in items), 2)
        # A return can only be requested against an order that already arrived, so both
        # terminal delivered states carry a delivery timestamp.
        delivered_at = (
            (placed_at + timedelta(minutes=eta + rng.randint(-3, 14))).isoformat()
            if status in DELIVERED_STATUSES
            else None
        )
        orders.append(
            Order(
                order_id=f"QC{240000 + index * 7}",
                store_id=store.store_id,
                customer_pincode=rng.choice(store.serviceable_pincodes),
                status=status,
                placed_at=placed_at.isoformat(),
                promised_eta_minutes=eta,
                delivered_at=delivered_at,
                items=[asdict(i) for i in items],
                order_total_inr=total,
                payment_mode=rng.choice(["UPI", "CARD", "WALLET", "COD"]),
                substitution_applied=rng.random() < 0.18,
            )
        )
    return orders


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    stores = build_dark_stores(rng)
    products = build_products(rng)
    inventory = build_inventory(products, stores, rng)
    orders = build_orders(products, stores, rng)

    catalog_dir = args.out / "catalog"
    write_json(catalog_dir / "dark_stores.json", [asdict(s) for s in stores])
    write_json(catalog_dir / "products.json", [asdict(p) for p in products])
    write_json(catalog_dir / "inventory.json", [asdict(r) for r in inventory])
    write_json(catalog_dir / "orders.json", [asdict(o) for o in orders])
    write_json(
        catalog_dir / "manifest.json",
        {
            "seed": args.seed,
            "generated_for_reference_time": NOW.isoformat(),
            "synthetic": True,
            "counts": {
                "dark_stores": len(stores),
                "products": len(products),
                "inventory_rows": len(inventory),
                "orders": len(orders),
            },
        },
    )

    print(
        f"stores={len(stores)} products={len(products)} "
        f"inventory={len(inventory)} orders={len(orders)} -> {catalog_dir}"
    )


if __name__ == "__main__":
    main()
