# Offscreen Screenshots

> Offscreen 2x GUI grabs need a large screen config file

## Offscreen 2x GUI captures come out cropped and magnified

**What happened:** Grabbing `MainWindow` with `QT_QPA_PLATFORM=offscreen` and `QT_SCALE_FACTOR=2` produced a cropped, zoomed-in UI: the window was clamped to the default offscreen screen (about 950x500 logical px), so `resize(1440, 860)` never took effect.
**Why:** The offscreen platform default screen is small; the scale factor halves its logical size further. The `dpr` key in an offscreen config file did not raise the device pixel ratio (grabs stayed 1x).
**Prevention:** Give the offscreen platform a large screen and keep the scale factor: `QT_QPA_PLATFORM="offscreen:configfile=screen.json" QT_SCALE_FACTOR=2` with `{"screens":[{"name":"s","x":0,"y":0,"width":3840,"height":2160,"logicalDpi":96,"logicalBaseDpi":96}]}`. Check the grab size (2880x1720 for a 1440x860 window) before using it.
**Fix:** Recaptured with the config file above.
