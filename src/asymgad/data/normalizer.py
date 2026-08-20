"""Node name normalization for unified graph construction.

All node identifiers across the five datasets must resolve to consistent
machine IDs so that the same physical host (e.g. C586) matches across
auth, dns, flows, proc, and redteam records.
"""

from __future__ import annotations

import re

DOMAIN_SUFFIX_RE = re.compile(r"@DOM\d+$", re.IGNORECASE)
MACHINE_ACCOUNT_RE = re.compile(r"\$$")


def normalize_node(name: str | None) -> str:
    """Normalize a node name to its canonical machine ID.

    Transformations:
        C5170$@DOM1  ->  C5170
        U66@DOM1     ->  U66
        C586         ->  C586  (already canonical)

    If *name* is None or empty it is returned as-is.
    """
    if not name or not isinstance(name, str):
        return name or ""

    name = DOMAIN_SUFFIX_RE.sub("", name)
    name = MACHINE_ACCOUNT_RE.sub("", name)
    return name


def extract_account_and_source(src_node: str) -> tuple[str, str, str]:
    """Parse an auth *src_node* into (account, domain, source_machine).

    Handles the two observed formats:

    ================================  =========  ======  ===============
    Input                              Account    Domain  Source Machine
    ================================  =========  ======  ===============
    ``C599$@DOM1|C1619``              C599$      DOM1    C1619
    ``ANONYMOUS LOGON@C586|C586``     ANONYMOUS  C586    C586
                                       LOGON
    ================================  =========  ======  ===============

    Returns *(account, domain, source_machine)*.  *domain* may be an
    empty string when no ``@``-separated domain is present.
    """
    if not isinstance(src_node, str) or "|" not in src_node:
        return ("", "", normalize_node(src_node))

    left, right = src_node.split("|", 1)
    source_machine = normalize_node(right)

    if "@" in left:
        account, domain = left.rsplit("@", 1)
    else:
        account, domain = left, ""

    return (account, domain, source_machine)
