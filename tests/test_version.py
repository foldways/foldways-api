import tomllib
from pathlib import Path

from constants import API_VERSION


def test_api_version_matches_pyproject():
    """The two version declarations must agree.

    Neither can derive from the other. uv requires `project.version`, and
    API_VERSION cannot read it back at runtime because the project is never
    installed and pyproject.toml is not shipped into the Modal image. Bumping
    a release means editing both, so this guards against editing only one.
    """
    pyproject = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())
    pyproject_version = pyproject["project"]["version"]
    assert pyproject_version == API_VERSION, (
        f"Version mismatch. pyproject.toml project.version is '{pyproject_version}' and "
        f"constants.py API_VERSION is '{API_VERSION}'. Set both to the version being released."
    )
