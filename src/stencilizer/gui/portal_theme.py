"""Linux desktop appearance fallback for Qt platform plugins without theme support."""

from PySide6.QtCore import SLOT, QObject, Qt, Signal, Slot
from PySide6.QtDBus import QDBus, QDBusConnection, QDBusMessage, QDBusVariant

_SERVICE = "org.freedesktop.portal.Desktop"
_PATH = "/org/freedesktop/portal/desktop"
_INTERFACE = "org.freedesktop.portal.Settings"
_NAMESPACE = "org.freedesktop.appearance"
_KEY = "color-scheme"


def portal_scheme(value: object) -> Qt.ColorScheme:
    """Decode both the nested Read reply and the SettingChanged variant."""
    while isinstance(value, QDBusVariant):
        value = value.variant()
    if type(value) is int:
        if value == 1:
            return Qt.ColorScheme.Dark
        if value == 2:
            return Qt.ColorScheme.Light
    return Qt.ColorScheme.Unknown


class PortalTheme(QObject):
    """Read and follow the standard desktop preference without an extra dependency."""

    changed = Signal(object)

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self.scheme = Qt.ColorScheme.Unknown
        bus = QDBusConnection.sessionBus()
        if not bus.isConnected():
            return
        bus.connect(  # type: ignore[call-overload]  # PySide requires SLOT's str at runtime.
            _SERVICE,
            _PATH,
            _INTERFACE,
            "SettingChanged",
            self,
            SLOT("setting_changed(QString,QString,QDBusVariant)"),
        )
        # Read supports version 1 portals; unwrap its extra variant below.
        message = QDBusMessage.createMethodCall(_SERVICE, _PATH, _INTERFACE, "Read")
        message.setArguments([_NAMESPACE, _KEY])
        reply = bus.call(message, QDBus.CallMode.Block, 250)
        if reply.type() == QDBusMessage.MessageType.ReplyMessage and reply.arguments():
            self.scheme = portal_scheme(reply.arguments()[0])

    @Slot(str, str, QDBusVariant)
    def setting_changed(self, namespace: str, key: str, value: QDBusVariant) -> None:
        """Ignore unrelated settings and forward changes to the theme follower."""
        if namespace == _NAMESPACE and key == _KEY:
            self.scheme = portal_scheme(value)
            self.changed.emit(self.scheme)
