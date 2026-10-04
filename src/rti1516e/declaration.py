"""Private helpers for ``Federate.publish_*`` and ``Federate.subscribe_*``.

The public declaration-management surface lives on ``Federate`` in
``connection.py``; this module is reserved for shared internal helpers.
"""

from __future__ import annotations

from enum import IntEnum


class AdvisorySwitchKind(IntEnum):
    OBJECT_CLASS_RELEVANCE = 1
    ATTRIBUTE_RELEVANCE = 2
    ATTRIBUTE_SCOPE = 3
    INTERACTION_RELEVANCE = 4


def validate_subscription_options(active: bool, update_rate_designator: str) -> None:
    if type(active) is not bool:
        raise TypeError("active must be a bool")
    if not isinstance(update_rate_designator, str):
        raise TypeError("update_rate_designator must be a str")
