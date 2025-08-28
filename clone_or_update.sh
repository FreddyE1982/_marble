#!/bin/sh
set -e
REPO="https://github.com/FreddyE1982/_marble/"
DIR="_marble"
if [ -d "$DIR/.git" ]; then
    git -C "$DIR" pull --ff-only
else
    git clone "$REPO" "$DIR"
fi
cd "$DIR"
if command -v py >/dev/null 2>&1; then
    py -3 -m pip install -e .
else
    python3 -m pip install -e .
fi
