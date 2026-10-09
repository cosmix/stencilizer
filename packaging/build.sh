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
  GUI_BIN="dist/Stencilizer.app/Contents/MacOS/Stencilizer"
else
  GUI_BIN="dist/stencilizer-gui/stencilizer-gui"
fi

pyinstaller() {
  uv run --no-sync pyinstaller --noconfirm --clean \
    --distpath dist --workpath build --specpath build "$@"
}

build() {
  rm -rf dist build
  pyinstaller --onefile --console --name stencilizer \
    --exclude-module PySide6 packaging/entry_cli.py
  if [ "$OS" = "Darwin" ]; then
    pyinstaller --onedir --windowed --name Stencilizer \
      --osx-bundle-identifier io.github.cosmix.stencilizer packaging/entry_gui.py
  else
    pyinstaller --onedir --windowed --name stencilizer-gui packaging/entry_gui.py
  fi
}

package() {
  local name="stencilizer-$VERSION-$PLATFORM"
  local stage="dist/release/$name"
  rm -rf dist/release
  mkdir -p "$stage"
  cp dist/stencilizer "$stage/"
  if [ "$OS" = "Darwin" ]; then
    # ditto preserves the bundle's symlinks and extended attributes.
    ditto dist/Stencilizer.app "$stage/Stencilizer.app"
    (cd dist/release && ditto -c -k --keepParent "$name" "$name.zip")
    ARCHIVE="dist/release/$name.zip"
  else
    cp -a dist/stencilizer-gui "$stage/"
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

  out="$(dist/stencilizer --version)"
  echo "$out"
  if [ "$out" != "Stencilizer v$VERSION" ]; then
    echo "error: expected 'Stencilizer v$VERSION'" >&2
    exit 1
  fi

  # Exercises the ProcessPoolExecutor path of the frozen binary. The processor always
  # starts workers with spawn, so this tests freeze_support on every platform.
  for font in Roboto-Regular.ttf CommitMono-Cosmix-700-Regular.otf; do
    dist/stencilizer "tests/fixtures/$font" -o "$tmp/$font" --workers 2
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
