#!/usr/bin/env bash
# Build unsigned standalone Stencilizer executables with PyInstaller.
#
#   packaging/build.sh           build and package into dist/release/
#   packaging/build.sh --smoke   smoke-test the binaries in dist/ (needs a display
#                                for the GUI; on headless Linux use xvfb-run -a)
#
# Requires an environment synced with the build group and the gui extra:
#   uv sync --locked --no-dev --group build --extra gui
# (--no-dev keeps test and lint packages out of the bundles.)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS-$ARCH" in
  Linux-x86_64) PLATFORM="linux-x86_64" ;;
  Darwin-arm64) PLATFORM="macos-arm64" ;;
  *) echo "error: unsupported platform $OS-$ARCH" >&2; exit 1 ;;
esac

VERSION="$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' src/stencilizer/__init__.py)"
if [ -z "$VERSION" ]; then
  echo "error: cannot read __version__ from src/stencilizer/__init__.py" >&2
  exit 1
fi

if [ "$OS" = "Darwin" ]; then
  GUI_BIN="dist/gui/Stencilizer.app/Contents/MacOS/Stencilizer"
else
  GUI_BIN="dist/gui/stencilizer-gui/stencilizer-gui"
fi
CLI_BIN="dist/cli/stencilizer"

# The CLI and GUI get separate directories: on case-insensitive filesystems (macOS)
# the GUI's "Stencilizer" outputs would otherwise overwrite the CLI's "stencilizer".
pyinstaller() {
  local kind="$1"
  shift
  uv run --no-sync pyinstaller --noconfirm --clean \
    --distpath "dist/$kind" --workpath "build/$kind" --specpath "build/$kind" "$@"
}

# The GUI reads its assets (the wordmark SVG) through importlib.resources, which PyInstaller
# resolves on disk next to the frozen package; the directory has to be bundled explicitly.
# The source is absolute because --add-data resolves relative paths against --specpath.
GUI_ASSETS="$ROOT/src/stencilizer/gui/assets:stencilizer/gui/assets"

build() {
  rm -rf dist build
  pyinstaller cli --onefile --console --name stencilizer \
    --exclude-module PySide6 packaging/entry_cli.py
  if [ "$OS" = "Darwin" ]; then
    pyinstaller gui --onedir --windowed --name Stencilizer --add-data "$GUI_ASSETS" \
      --osx-bundle-identifier io.github.cosmix.stencilizer packaging/entry_gui.py
  else
    pyinstaller gui --onedir --windowed --name stencilizer-gui --add-data "$GUI_ASSETS" \
      packaging/entry_gui.py
  fi
}

package() {
  local name="stencilizer-$VERSION-$PLATFORM"
  local stage="dist/release/$name"
  rm -rf dist/release
  mkdir -p "$stage"
  cp "$CLI_BIN" "$stage/"
  if [ "$OS" = "Darwin" ]; then
    # ditto preserves the bundle's symlinks and extended attributes.
    ditto dist/gui/Stencilizer.app "$stage/Stencilizer.app"
    (cd dist/release && ditto -c -k --keepParent "$name" "$name.zip")
    ARCHIVE="dist/release/$name.zip"
  else
    cp -a dist/gui/stencilizer-gui "$stage/"
    tar -C dist/release -czf "dist/release/$name.tar.gz" "$name"
    ARCHIVE="dist/release/$name.tar.gz"
  fi
  rm -rf "$stage"
  echo "$ARCHIVE"
}

tmp=""
smoke() {
  local out
  tmp="$(mktemp -d)"
  trap 'rm -rf "${tmp:-}"' EXIT

  out="$("$CLI_BIN" --version)"
  echo "$out"
  if [ "$out" != "Stencilizer v$VERSION" ]; then
    echo "error: expected 'Stencilizer v$VERSION'" >&2
    exit 1
  fi

  # Exercises the ProcessPoolExecutor path of the frozen binary. The processor always
  # starts workers with spawn, so this tests freeze_support on every platform.
  for font in Roboto-Regular.ttf CommitMono-Cosmix-700-Regular.otf; do
    "$CLI_BIN" "tests/fixtures/$font" -o "$tmp/$font" --workers 2
    test -s "$tmp/$font" || { echo "error: no output for $font" >&2; exit 1; }
  done

  "$GUI_BIN" &
  local pid=$!
  sleep 10
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "error: GUI exited during startup" >&2
    exit 1
  fi
  kill "$pid"
  wait "$pid" 2>/dev/null || true
  echo "smoke test passed"
}

if [ "${1:-}" = "--smoke" ]; then
  smoke
else
  build
  package
fi
