"""
COCO -> "obstruction" mapping for the P1 baseline.
====================================================================

The fine-tuned P1 detector is single-class ("obstruction"). To compare it
against a NON fine-tuned baseline (a YOLO pretrained on COCO), we cannot use
the COCO classes directly: we must decide which COCO categories count as a
plausible evacuation-route obstruction and which do not.

Design choice (see thesis / todo):
  - EXCLUDE people, personal/small items, animals, vehicles, food, etc.
    A person standing in a doorway is not a structural obstruction; a phone
    on the floor is too small to matter.
  - INCLUDE large movable indoor objects that could realistically block a
    door or hallway (chairs, couches, beds, tables, luggage, plants, ...).

COCO has no wheelchair / stretcher / cart / trash-bin class, so the baseline
is expected to MISS most hospital-specific obstacles. That gap is exactly the
motivation for fine-tuning and should be reported, not hidden.

This list is deliberately easy to edit: tweak OBSTRUCTION_COCO_NAMES and the
ids are recomputed from the model's own class names at runtime.
"""

from __future__ import annotations

# COCO-80 category names judged to be plausible obstructions.
# Kept as names (not ids) so the mapping stays correct even if a model ships
# a slightly different class ordering; ids are resolved from model.names.
OBSTRUCTION_COCO_NAMES: set[str] = {
    "bench",
    "backpack",
    "handbag",
    "suitcase",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "refrigerator",
}

# Detections smaller than this fraction of the image area are ignored, to honour
# the "very small objects are not obstructions" rule. Tune as needed.
MIN_BOX_AREA_FRAC: float = 0.01


def obstruction_class_ids(names: dict[int, str] | list[str]) -> set[int]:
    """Resolve the obstruction COCO ids from a model's class-name mapping.

    `names` is `model.names` from ultralytics (dict id->name) or a list.
    """
    if isinstance(names, dict):
        items = names.items()
    else:
        items = enumerate(names)
    return {i for i, n in items if n in OBSTRUCTION_COCO_NAMES}


def is_obstruction_detection(
    cls_id: int,
    box_area_frac: float,
    obstruction_ids: set[int],
    min_area_frac: float = MIN_BOX_AREA_FRAC,
) -> bool:
    """True if a single COCO detection should be counted as an obstruction."""
    return cls_id in obstruction_ids and box_area_frac >= min_area_frac
