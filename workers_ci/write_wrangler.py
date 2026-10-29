"""Write the Wrangler config that preview builds need.

Workers Builds installs ``requirements.txt`` before it runs Wrangler. A build
sets ``WRANGLER_CI_OVERRIDE_NAME`` to the Worker it intends to publish.
Writing the config here gives ``npx wrangler versions upload`` an assets entry
point whose name matches that Worker, which is what production ``wrangler
deploy`` auto-generates and preview builds do not.

This repository must never deploy the Worker ``youneek-xyz``. That name is
Andrew's portfolio. A build that still passes it is published as
``real-neural-forge-frontend`` instead. The generated config has no routes, so
it cannot claim ``youneek.xyz``, ``www.youneek.xyz``, ``forge.youneek.xyz``, or
``forge.youneek.dev``. Those hostnames belong to other Workers.
``forge.youneek.xyz`` and ``forge.youneek.dev`` are only on
``deploy/cloudflare`` (``real-neural-forge-api``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Same constraints Wrangler enforces: lowercase letters, digits, and dashes,
# not starting or ending with a dash, at most 63 characters.
_WORKER_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")

# The portfolio Worker. A push of this repo must not deploy under this name.
PORTFOLIO_WORKER = "youneek-xyz"
FRONTEND_WORKER = "real-neural-forge-frontend"
# Hostnames this generated config is not allowed to claim.
FORBIDDEN_HOSTS = (
    "youneek.xyz",
    "www.youneek.xyz",
    "forge.youneek.xyz",
    "forge.youneek.dev",
)


def write_wrangler_config(root: Path, worker_name: str | None) -> Path | None:
    """Write ``root/wrangler.jsonc`` for ``worker_name``.

    Returns the path written. An empty name leaves the tree unchanged so a
    normal ``pip install`` does not create a config. An invalid name raises.
    """
    requested = (worker_name or "").strip()
    if not requested:
        return None
    if _WORKER_NAME.fullmatch(requested) is None:
        raise ValueError(
            f"WRANGLER_CI_OVERRIDE_NAME {requested!r} is not a valid Worker name."
        )
    # The connected youneek-xyz build still sets that override. Publish a
    # different Worker so wrangler deploy cannot overwrite the portfolio.
    name = FRONTEND_WORKER if requested == PORTFOLIO_WORKER else requested
    if name == PORTFOLIO_WORKER:
        raise ValueError("Refusing to write a Worker named youneek-xyz.")

    config = {
        "name": name,
        "compatibility_date": "2026-10-12",
        "observability": {"enabled": True},
        "assets": {"directory": "frontend"},
        "workers_dev": True,
    }
    rendered = json.dumps(config, indent=2) + "\n"
    for host in FORBIDDEN_HOSTS:
        if host in rendered:
            raise ValueError(f"Frontend Worker config must not claim {host}.")
    if PORTFOLIO_WORKER in rendered:
        raise ValueError("Frontend Worker config must not name youneek-xyz.")

    path = root / "wrangler.jsonc"
    path.write_text(rendered, encoding="utf-8")
    return path
