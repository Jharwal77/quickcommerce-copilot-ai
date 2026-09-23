---
doc_id: policy-delivery
title: Delivery, ETA, and Fees Policy
operator: Nimbus Now
version: 3.8
effective_date: 2026-02-01
synthetic: true
---

# Delivery, ETA, and Fees Policy

This is a fictional policy document for Nimbus Now, an invented quick-commerce
operator. It exists to support retrieval evaluation and describes no real company.

## How the promised ETA is computed

The ETA shown at checkout is the sum of three components:

1. **Pick and pack time** at the fulfilling dark store, which averages 3 to 7 minutes.
2. **Rider assignment time**, normally under 2 minutes.
3. **Travel time** from the dark store to the delivery pincode.

The promised ETA is always quoted in whole minutes and is capped at **30 minutes** for any
serviceable pincode. If the computed ETA exceeds 30 minutes, the pincode is treated as
unserviceable from that store and the order is routed to another dark store or declined.

## Delivery fees

- Orders of **INR 199 and above**: delivery fee waived.
- Orders below INR 199: flat delivery fee of **INR 25**.
- Rain or peak surge fee: up to **INR 35** additional, shown before payment.
- Handling fee on any order containing a frozen item: **INR 10**.

A small-cart fee of INR 15 applies to orders below INR 99. This is in addition to the
delivery fee.

## Late delivery remedy

If an order arrives more than **10 minutes** after the promised ETA, the customer is
credited **INR 50** to Nimbus Wallet automatically, with no need to contact support. The
credit is issued once per order and does not apply to orders that were delayed because the
customer was unreachable at the delivery address.

## Serviceability and store hours

A pincode is serviceable only if it falls inside the service radius of at least one open
dark store. Service radii range from 2.5 km to 4.0 km depending on the store. Outside a
store's operating hours, its pincodes are served by the nearest open store if the 30-minute
ETA cap can still be met; otherwise ordering is disabled for that pincode until the store
reopens.

Scheduled delivery is not offered. Nimbus Now serves on-demand orders only.

## Rider handover rules

Riders wait at the delivery address for up to **5 minutes**. After that the order is
marked undelivered and returned to the dark store. Perishable items from an undelivered
order are written off and are not restocked.
