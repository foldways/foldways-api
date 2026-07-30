"""Log the arrays in a .npz (or .npy) file: name, shape, dtype, and value range.

Useful for eyeballing service outputs like ESMC embeddings and variant scores
without writing a one-off numpy snippet.

Run it as a module from the repository root:

    uv run python -m scripts.inspect_npz mocks/esmc/output/variant_scores_0.npz
"""

import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def describe(name: str, array: np.ndarray) -> None:
    line = f"{name}: shape={array.shape} dtype={array.dtype}"
    if np.issubdtype(array.dtype, np.number) and array.size:
        line += f" min={array.min():.4g} max={array.max():.4g} mean={array.mean():.4g}"
    logger.info(line)


def main(path: str) -> None:
    file = Path(path)
    if not file.exists():
        sys.exit(f"No such file: {file}")

    loaded = np.load(file, allow_pickle=False)
    logger.info(f"{file} ({file.stat().st_size / 1024:.1f} KB)")
    if hasattr(loaded, "files"):  # an .npz archive holds several named arrays
        for name in loaded.files:
            describe(name, loaded[name])
    else:  # a bare .npy holds one array
        describe(file.stem, loaded)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: uv run python -m scripts.inspect_npz <file.npz>")
    main(sys.argv[1])
