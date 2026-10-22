"""Keep the API test suite from downloading space catalogs."""

from __future__ import annotations

import os

# Lifespan refreshes JPL files unless this is off. Tests call the refresh
# helpers directly when they want to exercise that path.
os.environ["NEURAL_FORGE_SPACE_REFRESH"] = "0"
