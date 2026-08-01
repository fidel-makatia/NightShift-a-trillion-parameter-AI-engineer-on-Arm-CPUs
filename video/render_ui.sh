#!/usr/bin/env bash
# Render a local index.html to desktop + mobile screenshots using headless Chrome.
set -euo pipefail
HTML="${1:?usage: render_ui.sh index.html outdir}"
OUT="${2:-video/ui}"
mkdir -p "$OUT"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
URL="file://$(cd "$(dirname "$HTML")" && pwd)/$(basename "$HTML")"

"$CHROME" --headless=new --disable-gpu --hide-scrollbars \
  --window-size=1440,900 --screenshot="$OUT/ui_desktop.png" "$URL" >/dev/null 2>&1
"$CHROME" --headless=new --disable-gpu --hide-scrollbars \
  --window-size=390,844 --screenshot="$OUT/ui_mobile.png" "$URL" >/dev/null 2>&1
echo "wrote $OUT/ui_desktop.png and $OUT/ui_mobile.png"
