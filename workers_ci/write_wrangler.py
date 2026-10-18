"""Write the Wrangler config that preview builds need.

Workers Builds installs ``requirements.txt`` before it runs Wrangler.
``real-neural-forge`` and ``youneek-xyz`` both build this repository, and each
build sets ``WRANGLER_CI_OVERRIDE_NAME`` to that Worker's name. Writing the
config here gives ``npx wrangler versions upload`` an assets entry point whose
name matches the Worker, which is what production ``wrangler deploy``
auto-generates and preview builds do not.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Same constraints Wrangler enforces: lowercase letters, digits, and dashes,
# not starting or ending with a dash, at most 63 characters.
_WORKER_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


def write_wrangler_config(root: Path, worker_name: str | None) -> Path | None:
    """Write ``root/wrangler.jsonc`` for ``worker_name``.

    Returns the path written. An empty name leaves the tree unchanged so a
    normal ``pip install`` does not create a config. An invalid name raises.
    """
    name = (worker_name or "").strip()
    if not name:
        return None
    if _WORKER_NAME.fullmatch(name) is None:
        raise ValueError(
            f"WRANGLER_CI_OVERRIDE_NAME {name!r} is not a valid Worker name."
        )

    path = root / "wrangler.jsonc"
    path.write_text(
        json.dumps(
            {
                "name": name,
                "compatibility_date": "2026-10-12",
                "observability": {"enabled": True},
                "assets": {"directory": "frontend"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
