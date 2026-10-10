"""Tests for the loading view shown while a font opens."""

from itertools import pairwise

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPalette
from pytestqt.qtbot import QtBot

from stencilizer.gui.loader import LOADER_DELAY_MS, LoadingView, prefers_reduced_motion
from stencilizer.gui.loader_paint import CYCLE_SECONDS, MIST_CAP, PHASES, SPARK_CAP, frame_state
from stencilizer.gui.loader_scene import load_wordmark
from stencilizer.gui.theme import DARK, LIGHT, ThemeColors, palette_for

LETTERS = 11
FRAME_TIMES = (0.1, 0.8, 1.9, 2.3, 2.9, 3.35, 3.45, CYCLE_SECONDS * LETTERS + 0.5)


class FakeClock:
    """A clock the tests advance by hand, in seconds."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    """Deterministic clock injected into the view."""
    return FakeClock()


@pytest.fixture
def view(qtbot: QtBot, clock: FakeClock) -> LoadingView:
    """A loading view sized like the central grid area, full motion, light palette."""
    widget = LoadingView(clock=clock, reduced_motion=False)
    widget.setPalette(palette_for(LIGHT))
    widget.resize(520, 700)
    qtbot.addWidget(widget)
    return widget


def _luminance(color: QColor) -> float:
    def linear(channel: int) -> float:
        value = channel / 255
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * linear(color.red())
        + 0.7152 * linear(color.green())
        + 0.0722 * linear(color.blue())
    )


def _contrast(first: QColor, second: QColor) -> float:
    high, low = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _painted_fraction(image: QImage, background: QColor) -> float:
    """Share of sampled pixels that differ from the background colour."""
    step = 7
    total = painted = 0
    for y in range(0, image.height(), step):
        for x in range(0, image.width(), step):
            total += 1
            if image.pixelColor(x, y) != background:
                painted += 1
    return painted / total


def _run_to(view: LoadingView, clock: FakeClock, until: float, step: float = 1 / 60) -> None:
    """Advance the clock to ``until`` in ``step`` increments, painting every frame."""
    while clock.now < until:
        clock.now = min(clock.now + step, until)
        view.grab()


def test_wordmark_splits_into_letters_left_to_right() -> None:
    """The packaged SVG yields the eleven letters of the wordmark in reading order."""
    wordmark = load_wordmark()
    assert len(wordmark.letters) == LETTERS
    lefts = [letter.bounds.left() for letter in wordmark.letters]
    assert lefts == sorted(lefts)
    assert [len(letter.parts) for letter in wordmark.letters].count(2) == 2  # the two i's
    assert wordmark.view_box.width() > wordmark.view_box.height() * 5


def test_timeline_covers_every_phase_and_loops() -> None:
    """Every phase is reached within a cycle and the letter index wraps after the last one."""
    seen = {frame_state(t / 100, LETTERS).phase for t in range(int(CYCLE_SECONDS * 100))}
    assert seen == {name for name, _ in PHASES}
    assert frame_state(CYCLE_SECONDS * LETTERS + 0.1, LETTERS).index == 0
    assert frame_state(CYCLE_SECONDS * 3 + 0.1, LETTERS).index == 3
    assert 0.0 <= frame_state(-5.0, LETTERS).u <= 1.0


def test_delay_constant() -> None:
    """The window shows the view only after half a second."""
    assert LOADER_DELAY_MS == 500


def test_start_sets_plain_text_label_and_runs_while_shown(view: LoadingView) -> None:
    """``start`` shows the file name verbatim as plain text and ticks only while visible."""
    view.show()
    assert not view.is_running
    view.start("Fancy <b>Font</b> & Co.ttf")
    assert view.status_label.text() == "Opening Fancy <b>Font</b> & Co.ttf…"
    assert view.status_label.textFormat() == Qt.TextFormat.PlainText
    assert view.is_running


def test_start_before_show_runs_on_show(view: LoadingView) -> None:
    """A view started while hidden begins ticking when it becomes visible."""
    view.start("Roboto-Regular.ttf")
    assert not view.is_running
    view.show()
    assert view.is_running


def test_stop_and_hide_halt_the_timer(view: LoadingView) -> None:
    """``stop`` and hiding both stop the frame timer; showing again resumes a started view."""
    view.show()
    view.start("Roboto-Regular.ttf")
    view.hide()
    assert not view.is_running
    view.show()
    assert view.is_running
    view.stop()
    assert not view.is_running
    view.hide()
    view.show()
    assert not view.is_running


def test_long_file_name_is_elided(view: LoadingView) -> None:
    """A file name wider than the view is elided in the middle rather than widening it."""
    name = "A" * 400 + ".ttf"
    view.show()
    view.start(name)
    text = view.status_label.text()
    assert "…" in text
    assert len(text) < len(name)
    assert text.startswith("Opening A")
    assert view.width() == 520


@pytest.mark.parametrize("colors", [LIGHT, DARK], ids=["light", "dark"])
def test_palette_change_repaints_in_both_themes(
    view: LoadingView, clock: FakeClock, colors: ThemeColors
) -> None:
    """A palette change re-derives the scene colours; the wall takes the window colour."""
    view.show()
    view.start("Roboto-Regular.ttf")
    view.setPalette(palette_for(colors))
    clock.now = 0.8
    image = view.grab().toImage()
    assert image.pixelColor(2, 2) == QColor(colors.window)
    assert _painted_fraction(image, QColor(colors.window)) > 0.1


@pytest.mark.parametrize("colors", [LIGHT, DARK], ids=["light", "dark"])
def test_status_text_sits_on_the_wall_with_contrast(
    view: LoadingView, clock: FakeClock, colors: ThemeColors
) -> None:
    """The status label is drawn over the plain window colour at 4.5:1 or better."""
    view.show()
    view.start("Roboto-Regular.ttf")
    view.setPalette(palette_for(colors))
    clock.now = 2.9
    image = view.grab().toImage()
    label = view.status_label.geometry()
    background = image.pixelColor(label.left() - 4, label.center().y())
    assert background == QColor(colors.window)
    text = view.status_label.palette().color(QPalette.ColorRole.WindowText)
    assert _contrast(text, background) >= 4.5
    hint = view.hint_label.palette().color(QPalette.ColorRole.PlaceholderText)
    assert _contrast(hint, background) >= 4.5


def test_frames_render_distinct_non_blank_images(view: LoadingView, clock: FakeClock) -> None:
    """Deterministic clock readings give non-blank, mutually different frames."""
    view.show()
    view.start("Roboto-Regular.ttf")
    wall = QColor(LIGHT.window)
    clock.now = 0.0
    blank = _painted_fraction(view.grab().toImage(), wall)  # labels, caption and ghost strip
    frames = []
    for moment in FRAME_TIMES:
        clock.now = moment
        image = view.grab().toImage()
        assert _painted_fraction(image, wall) > blank + 0.005, moment
        frames.append(image)
    for first, second in pairwise(frames):
        assert first != second


def test_narrow_and_wide_sizes_render(view: LoadingView, clock: FakeClock) -> None:
    """The scene lays itself out again for very narrow and very wide panes."""
    view.show()
    view.start("Roboto-Regular.ttf")
    for width, height in ((360, 480), (1400, 900)):
        view.resize(width, height)
        clock.now += 1.0
        image = view.grab().toImage()
        assert (image.width(), image.height()) == (width, height)
        assert _painted_fraction(image, QColor(LIGHT.window)) > 0.05


def test_particles_stay_bounded_over_many_seconds(view: LoadingView, clock: FakeClock) -> None:
    """Sparks and paint droplets never exceed their caps, however long the loop runs."""
    view.show()
    view.start("Roboto-Regular.ttf")
    peak = 0
    for _ in range(600):
        clock.now += 0.05
        view.grab()
        peak = max(peak, view.particle_count)
        assert view.particle_count <= SPARK_CAP + MIST_CAP
    assert peak > 0


def test_restart_resets_particles(view: LoadingView, clock: FakeClock) -> None:
    """A second ``start`` begins from the first frame with no leftover particles."""
    view.show()
    view.start("Roboto-Regular.ttf")
    _run_to(view, clock, 1.0)
    assert view.particle_count > 0
    view.start("Other.otf")
    assert view.particle_count == 0
    assert view.elapsed() == 0.0


def test_reduced_motion_variant_has_no_particles(qtbot: QtBot, clock: FakeClock) -> None:
    """The calm variant animates without sparks or mist and still renders every phase."""
    view = LoadingView(clock=clock, reduced_motion=True)
    view.resize(520, 700)
    qtbot.addWidget(view)
    view.show()
    view.start("Roboto-Regular.ttf")
    assert view.reduced_motion
    _run_to(view, clock, CYCLE_SECONDS + 0.2, step=0.05)
    assert view.particle_count == 0


def test_reduced_motion_probe_defaults_to_false() -> None:
    """Without a reduced-motion hint in this Qt build the full animation is used."""
    assert prefers_reduced_motion() is False
