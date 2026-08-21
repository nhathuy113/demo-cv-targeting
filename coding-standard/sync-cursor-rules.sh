#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT/coding-standard"
DEST_DIR="$ROOT/.cursor/rules"

mkdir -p "$DEST_DIR"

copied=0
for src in "$SRC_DIR"/*.mdc; do
  [[ -f "$src" ]] || continue
  name="$(basename "$src")"
  cp "$src" "$DEST_DIR/$name"
  echo "Copied $name -> .cursor/rules/"
  copied=$((copied + 1))
done

if [[ "$copied" -eq 0 ]]; then
  echo "No .mdc files in $SRC_DIR" >&2
  exit 1
fi
