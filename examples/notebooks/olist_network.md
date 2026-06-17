# Olist Late-Delivery Network

Predicting **`is_late`** (positive label: `1`).

## Network structure

Each arrow `A --> B` means *A directly influences B*. The highlighted node is the target.

```mermaid
flowchart TD
    n0["customers__customer_state"]
    n1["mean_freight"]
    n2["is_late"]:::target
    n3["n_items"]
    n4["total_payment"]
    n5["total_item_value"]
    n6["max_installments"]
    n7["latest_category"]
    n8["n_sellers"]
    n9["n_payments"]
    n10["n_payment_types"]
    n11["purchase_month"]
    n0 --> n2
    n0 --> n1
    n2 --> n11
    n1 --> n4
    n3 --> n4
    n9 --> n10
    n8 --> n3
    n5 --> n7
    n5 --> n6
    n4 --> n5
    classDef target fill:#e74c3c,stroke:#c0392b,color:#fff;
```

## Relationships

- `customers__customer_state` directly predicts `is_late`
- `customers__customer_state` influences `mean_freight`
- `purchase_month` depends on the outcome `is_late`
- `mean_freight` influences `total_payment`
- `n_items` influences `total_payment`
- `n_payments` influences `n_payment_types`
- `n_sellers` influences `n_items`
- `total_item_value` influences `latest_category`
- `total_item_value` influences `max_installments`
- `total_payment` influences `total_item_value`
