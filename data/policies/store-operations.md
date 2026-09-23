---
doc_id: policy-store-ops
title: Dark Store Operations Standards
operator: Nimbus Now
version: 5.1
effective_date: 2026-02-20
synthetic: true
---

# Dark Store Operations Standards

This is a fictional policy document for Nimbus Now, an invented quick-commerce
operator. It exists to support retrieval evaluation and describes no real company.

## Cold chain requirements

Storage class determines the holding temperature at the dark store:

- **Chilled** items are held between **2 and 6 degrees Celsius**.
- **Frozen** items are held at **minus 18 degrees Celsius or colder**.
- **Ambient** items are held below 30 degrees Celsius and away from direct sunlight.

A chilled item left outside its temperature band for more than **20 minutes** must be
written off and cannot be sold. Frozen items that show any sign of refreezing are written
off immediately.

## Replenishment and reorder points

Each store and SKU pair carries a reorder point. When on-hand units fall to or below the
reorder point, a replenishment request is raised on the next inbound truck slot. Stores
receive up to **two inbound trucks per day**.

Available units are defined as **on-hand units minus reserved units**. Reserved units
belong to orders that are accepted but not yet packed. An item is treated as out of stock
when available units reach zero, even if on-hand units are non-zero.

## Pick accuracy and audit

Pickers are held to a **99.2 percent** line-level pick accuracy target. Every store is
cycle-counted on a rolling basis so that each SKU is physically counted at least once per
**14 days**. Discrepancies above **2 percent** of counted units trigger a full category
recount.

## Expiry management

Items within **20 percent of their remaining shelf life** are moved to a Near Expiry tag
and discounted. Near Expiry items are excluded from substitution and are not returnable.
Items past expiry are quarantined and written off the same day.

## Store staffing

A dark store runs with a store manager, two to four pickers per shift, and a rider pool of
8 to 22 riders depending on the store's order volume. Pick and pack time is the primary
controllable component of the promised ETA, which is why it carries a hard internal
target of under 7 minutes.
