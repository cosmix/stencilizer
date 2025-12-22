"""Exception hierarchy for Stencilizer."""


class StencilizerError(Exception):
    """Base exception for all Stencilizer errors."""

    pass


class FontError(StencilizerError):
    """Errors related to font loading or saving."""

    pass


class FontLoadError(FontError):
    """Error loading a font file."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to load font '{path}': {reason}")


class FontSaveError(FontError):
    """Error saving a font file."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to save font '{path}': {reason}")


class FontFormatError(FontError):
    """Unsupported or invalid font format."""

    def __init__(self, path: str, details: str) -> None:
        self.path = path
        self.details = details
        super().__init__(f"Invalid font format '{path}': {details}")


class GlyphError(StencilizerError):
    """Errors related to glyph processing."""

    pass


class GlyphNotFoundError(GlyphError):
    """Requested glyph not found in font."""

    def __init__(self, glyph_name: str) -> None:
        self.glyph_name = glyph_name
        super().__init__(f"Glyph '{glyph_name}' not found in font")


class GlyphProcessingError(GlyphError):
    """Error processing a specific glyph."""

    def __init__(self, glyph_name: str, reason: str) -> None:
        self.glyph_name = glyph_name
        self.reason = reason
        super().__init__(f"Error processing glyph '{glyph_name}': {reason}")


class GeometryError(StencilizerError):
    """Errors in geometric calculations."""

    pass


class ContourError(GeometryError):
    """Error with contour data or operations."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class IntersectionError(GeometryError):
    """Error calculating intersections."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class BridgeError(StencilizerError):
    """Errors related to bridge placement or generation."""

    pass


class BridgePlacementError(BridgeError):
    """Could not find valid bridge placement."""

    def __init__(self, glyph_name: str, reason: str) -> None:
        self.glyph_name = glyph_name
        self.reason = reason
        super().__init__(f"Bridge placement failed for '{glyph_name}': {reason}")


class BridgeGenerationError(BridgeError):
    """Error generating bridge geometry."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Bridge generation failed: {reason}")


class ProcessingCancelledError(StencilizerError):
    """Processing was cancelled by user."""

    def __init__(self, processed_count: int, pending_count: int) -> None:
        self.processed_count = processed_count
        self.pending_count = pending_count
        super().__init__(
            f"Processing cancelled: {processed_count} completed, {pending_count} pending"
        )
