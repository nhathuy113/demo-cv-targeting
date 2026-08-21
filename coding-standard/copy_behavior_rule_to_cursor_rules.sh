#!/usr/bin/env bash
set -euo pipefail
# Deprecated name: sync all coding-standard/*.mdc → .cursor/rules/
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/coding-standard/sync-cursor-rules.sh"
