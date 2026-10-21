"""Train the outer-space configs and write validation scores.

Run from the repository root, after ``python -m scripts.fetch_space``:

    python -m scripts.train_space

The same command is ``python -m src.space.train``. Checkpoints are gitignored.
"""

from __future__ import annotations

from src.space.train import main

if __name__ == "__main__":
    main()
