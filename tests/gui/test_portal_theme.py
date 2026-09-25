"""Desktop portal fallback when Qt cannot detect the system theme."""

import sys

import pytest
from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux desktop portal")


@pytest.mark.parametrize(
    ("value", "scheme"),
    [
        (1, Qt.ColorScheme.Dark),
        (2, Qt.ColorScheme.Light),
        (0, Qt.ColorScheme.Unknown),
        (42, Qt.ColorScheme.Unknown),
        ("1", Qt.ColorScheme.Unknown),
    ],
)
def test_portal_variants(value: object, scheme: Qt.ColorScheme) -> None:
    from PySide6.QtDBus import QDBusVariant

    from stencilizer.gui.portal_theme import portal_scheme

    assert portal_scheme(QDBusVariant(QDBusVariant(value))) == scheme


def test_portal_read_and_updates(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtDBus import QDBusConnection, QDBusMessage, QDBusVariant

    from stencilizer.gui import portal_theme

    class Bus:
        def isConnected(self) -> bool:  # noqa: N802
            return True

        def connect(self, *args: object) -> bool:
            assert args[3] == "SettingChanged"
            return True

        def call(self, message: QDBusMessage, _mode: object, timeout: int) -> QDBusMessage:
            assert message.arguments() == ["org.freedesktop.appearance", "color-scheme"]
            assert 0 < timeout <= 250
            reply = message.createReply()
            reply.setArguments([QDBusVariant(QDBusVariant(1))])
            return reply

    monkeypatch.setattr(QDBusConnection, "sessionBus", Bus)
    portal = portal_theme.PortalTheme(qapp)
    assert portal.scheme == Qt.ColorScheme.Dark
    changed: list[Qt.ColorScheme] = []
    portal.changed.connect(changed.append)
    portal.setting_changed("unrelated", "color-scheme", QDBusVariant(2))
    assert changed == []
    portal.setting_changed("org.freedesktop.appearance", "color-scheme", QDBusVariant(2))
    assert changed == [Qt.ColorScheme.Light]
    portal.deleteLater()


@pytest.mark.parametrize("connected", [False, True])
def test_unavailable_portal(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch, connected: bool
) -> None:
    from PySide6.QtDBus import QDBusConnection, QDBusMessage

    from stencilizer.gui import portal_theme

    class Bus:
        def isConnected(self) -> bool:  # noqa: N802
            return connected

        def connect(self, *_args: object) -> bool:
            return False

        def call(self, *_args: object) -> QDBusMessage:
            return QDBusMessage.createError("org.freedesktop.DBus.Error.ServiceUnknown", "missing")

    monkeypatch.setattr(QDBusConnection, "sessionBus", Bus)
    portal = portal_theme.PortalTheme(qapp)
    assert portal.scheme == Qt.ColorScheme.Unknown
    portal.deleteLater()


def test_unknown_qt_scheme_uses_portal(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    from stencilizer.gui import portal_theme, theme

    class DarkPortal(QObject):
        changed = portal_theme.PortalTheme.changed
        scheme = Qt.ColorScheme.Dark

    monkeypatch.setattr(portal_theme, "PortalTheme", DarkPortal)
    monkeypatch.setattr(qapp.styleHints(), "colorScheme", lambda: Qt.ColorScheme.Unknown)
    palette, stylesheet = QPalette(qapp.palette()), qapp.styleSheet()
    # Isolate the follower from other tests that share the QApplication.
    monkeypatch.setattr(theme, "_FOLLOWER_NAME", "portal-test-follower")
    theme.apply_theme(qapp)
    follower = qapp.findChild(theme._SchemeFollower, "portal-test-follower")
    assert follower is not None
    try:
        assert qapp.palette().color(QPalette.ColorRole.Window) == QColor(theme.DARK.window)
        portal = follower.findChild(DarkPortal)
        assert portal is not None
        portal.changed.emit(Qt.ColorScheme.Light)
        assert qapp.palette().color(QPalette.ColorRole.Window) == QColor(theme.LIGHT.window)
        assert follower.resolve(Qt.ColorScheme.Dark) == Qt.ColorScheme.Dark
        theme.apply_theme(qapp, Qt.ColorScheme.Dark)
        assert qapp.palette().color(QPalette.ColorRole.Window) == QColor(theme.DARK.window)
    finally:
        qapp.styleHints().colorSchemeChanged.disconnect(follower._on_scheme_changed)
        follower.deleteLater()
        qapp.setPalette(palette)
        qapp.setStyleSheet(stylesheet)
