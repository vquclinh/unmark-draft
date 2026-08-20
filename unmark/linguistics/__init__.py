"""Pinned linguistic resources and the Vietnamese eligibility rule (B3A).

Closes GAP-2: proposal §4.3's "matches the Vietnamese syllable inventory after
stripping". Pure standard library plus PyYAML; no model, no tokenizer, no
network at import or call time.
"""

from unmark.linguistics.classify import (
    classify_candidate,
    is_vietnamese_candidate,
    make_classifier,
)
from unmark.linguistics.inventory import (
    DEFAULT_MANIFEST,
    ELIGIBILITY_SCHEMA_VERSION,
    InventoryChecksumMismatch,
    InventoryProvenance,
    InventoryUnavailable,
    SyllableInventory,
    build_inventory,
    clear_inventory_cache,
    load_inventory,
    load_inventory_cached,
    load_manifest,
    membership_form,
    try_load_inventory,
)

__all__ = [
    "DEFAULT_MANIFEST",
    "ELIGIBILITY_SCHEMA_VERSION",
    "InventoryChecksumMismatch",
    "InventoryProvenance",
    "InventoryUnavailable",
    "SyllableInventory",
    "build_inventory",
    "classify_candidate",
    "clear_inventory_cache",
    "is_vietnamese_candidate",
    "load_inventory",
    "load_inventory_cached",
    "load_manifest",
    "make_classifier",
    "membership_form",
    "try_load_inventory",
]
