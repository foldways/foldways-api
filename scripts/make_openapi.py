"""Write the API's OpenAPI schema to docs/openapi.json.

FastAPI builds the schema from the app, so this needs no deployment or Modal
access. The committed file is the contract the docs site renders, and its diff
is how a schema change shows up in review. CI regenerates it and fails if the
committed copy is stale, so run this and commit the result when the API changes.

Run it as a module from the repository root, so the project's imports resolve:

    uv run python -m scripts.make_openapi
"""

import json
from pathlib import Path

from app import web_app

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_PATH = PROJECT_ROOT / "docs" / "openapi.json"


def main() -> None:
    schema = web_app.openapi()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(schema, indent=2) + "\n")
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)} ({len(schema['paths'])} paths)")


if __name__ == "__main__":
    main()
