"""Tests for variable-font pinning helpers."""

from pathlib import Path

import pytest
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]

from stencilizer.cli.pinning import (
    is_cff2_font,
    is_variable_font,
    pin_stenciled,
    pinned_input,
    publish_pinned,
    validate_instance,
)
from stencilizer.exceptions import FontLoadError, FontSaveError
from stencilizer.io.instance import InstanceSpecError
from tests.font_helpers import CANTARELL, INTER, ROBOTO


def _has_stenciled_family_name(path: Path) -> bool:
    with TTFont(path) as font:
        return any(
            " Stenciled" in record.toUnicode()
            for record in font["name"].names
            if record.nameID == 1
        )


def test_pinned_input_without_instance_yields_original_path() -> None:
    with pinned_input(INTER, None) as pinned:
        assert pinned == INTER


def test_pinned_input_creates_temporary_static_instance() -> None:
    with pinned_input(INTER, "wght=700") as pinned:
        assert pinned.exists()
        with TTFont(pinned) as font:
            assert "fvar" not in font
    assert not pinned.exists()


@pytest.mark.parametrize(
    ("font_path", "instance", "message"),
    [
        (INTER, "nope=1", "unknown axis 'nope'"),
        (ROBOTO, "wght=700", "--instance requires a variable font"),
    ],
)
def test_validate_instance_rejects_invalid_specs(
    font_path: Path, instance: str, message: str
) -> None:
    with pytest.raises(InstanceSpecError, match=message):
        validate_instance(font_path, instance)


def test_pin_stenciled_writes_static_instance(tmp_path: Path) -> None:
    pinned = pin_stenciled(INTER, INTER, "wght=900", tmp_path)

    assert pinned.parent == tmp_path
    with TTFont(pinned) as font:
        assert "fvar" not in font


def test_publish_pinned_writes_stenciled_font_and_cleans_staging(tmp_path: Path) -> None:
    output = tmp_path / "out.ttf"

    result = publish_pinned(INTER, output)

    assert result == output
    assert _has_stenciled_family_name(output)
    assert list(tmp_path.iterdir()) == [output]


def test_publish_pinned_rejects_missing_parent_directory(tmp_path: Path) -> None:
    output = tmp_path / "missing" / "out.ttf"

    with pytest.raises(FontSaveError):
        publish_pinned(INTER, output)

    assert not output.parent.exists()


def test_publish_pinned_replaces_existing_file(tmp_path: Path) -> None:
    output = tmp_path / "out.ttf"
    original = b"existing output"
    output.write_bytes(original)

    publish_pinned(INTER, output)

    assert output.read_bytes() != original
    assert _has_stenciled_family_name(output)


def test_publish_pinned_rejects_directory_output_without_changes(tmp_path: Path) -> None:
    output = tmp_path / "out.ttf"
    output.mkdir()
    preserved = output / "preserved.txt"
    preserved.write_text("keep")

    with pytest.raises(FontSaveError):
        publish_pinned(INTER, output)

    assert output.is_dir()
    assert preserved.read_text() == "keep"
    assert list(output.iterdir()) == [preserved]


def test_publish_pinned_preserves_existing_output_when_replace_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "out.ttf"
    original = b"existing output"
    output.write_bytes(original)

    def fail_replace(_: Path, __: Path) -> Path:
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(FontSaveError):
        publish_pinned(INTER, output)

    assert output.read_bytes() == original
    assert list(tmp_path.iterdir()) == [output]
    assert not list(tmp_path.glob(".stencilizer-pin-*"))


@pytest.mark.parametrize(("path", "expected"), [(INTER, True), (ROBOTO, False)])
def test_is_variable_font(path: Path, expected: bool) -> None:
    assert is_variable_font(path) is expected


@pytest.mark.parametrize(("path", "expected"), [(CANTARELL, True), (INTER, False)])
def test_is_cff2_font(path: Path, expected: bool) -> None:
    assert is_cff2_font(path) is expected


def test_is_variable_font_wraps_text_file_load_error(tmp_path: Path) -> None:
    text_file = tmp_path / "not-a-font.txt"
    text_file.write_text("not a font")

    with pytest.raises(FontLoadError):
        is_variable_font(text_file)
