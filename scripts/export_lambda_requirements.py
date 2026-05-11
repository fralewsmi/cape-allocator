"""Export Lambda runtime requirements from pyproject.toml."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".serverless-requirements.txt"


def main() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = pyproject["project"]
    dependencies = [
        *project.get("dependencies", []),
        *project.get("optional-dependencies", {}).get("api", []),
    ]

    OUTPUT.write_text("\n".join(dependencies) + "\n")


if __name__ == "__main__":
    main()
