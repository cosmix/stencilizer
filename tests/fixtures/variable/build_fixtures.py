"""Build the variable-font test fixtures by subsetting three installed system fonts.

Run once from the repository root with ``uv run python tests/fixtures/variable/build_fixtures.py``;
the committed subsets are what the tests read.
"""

from pathlib import Path

from fontTools import subset  # type: ignore[import-untyped]
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

FIXTURE_DIR = Path(__file__).parent
CHARACTERS = "ADPRe4&BOabdgopq0689lxÁ"

SOURCES: dict[str, tuple[Path, str]] = {
    "Ubuntu-VF-subset.ttf": (
        Path("/usr/share/fonts/truetype/ubuntu/Ubuntu[wdth,wght].ttf"),
        "Ubuntu Font Licence 1.0",
    ),
    "Inter-VF-subset.ttf": (
        Path.home() / ".local/share/fonts/InterVariable.ttf",
        "SIL Open Font License 1.1",
    ),
    "Cantarell-VF-subset.otf": (
        Path("/usr/share/fonts/opentype/cantarell/Cantarell-VF.otf"),
        "SIL Open Font License 1.1",
    ),
}


def _options() -> subset.Options:
    options = subset.Options()
    options.layout_features = []
    options.name_IDs = ["*"]
    options.name_languages = ["*"]
    options.notdef_outline = True
    options.glyph_names = True
    return options


def build_subset(source: Path, target: Path) -> None:
    """Subset ``source`` to CHARACTERS and save it as ``target``."""
    options = _options()
    font = TTFont(source)
    subsetter = subset.Subsetter(options=options)
    subsetter.populate(text=CHARACTERS)
    subsetter.subset(font)
    font.save(target)


def write_readme() -> None:
    """Name each fixture's source path and licence."""
    lines = [
        "# Variable font fixtures",
        "",
        "Built by `build_fixtures.py`, subset to the characters "
        f"`{CHARACTERS}` with fontTools.subset.",
        "",
        "| Fixture | Source | Licence |",
        "| --- | --- | --- |",
    ]
    for name, (source, licence) in SOURCES.items():
        shown = str(source).replace(str(Path.home()), "~")
        lines.append(f"| `{name}` | `{shown}` | {licence} |")
    (FIXTURE_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for name, (source, _licence) in SOURCES.items():
        build_subset(source, FIXTURE_DIR / name)
    write_readme()


if __name__ == "__main__":
    main()
