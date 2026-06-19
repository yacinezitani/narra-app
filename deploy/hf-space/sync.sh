#!/usr/bin/env bash
# Assemble a Hugging Face Space repo from this project.
#
# Usage (from the repo root):
#   ./deploy/hf-space/sync.sh <path-to-cloned-space-repo>
#
# Copies the backend, frontend, Dockerfile, requirements, and Space README into
# the Space repo root so it can be committed and pushed to HF.

set -euo pipefail

DEST="${1:-}"
if [[ -z "$DEST" ]]; then
  echo "usage: $0 <path-to-cloned-space-repo>" >&2
  exit 1
fi
if [[ ! -d "$DEST/.git" ]]; then
  echo "error: '$DEST' is not a git repo (clone your Space there first)" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "Syncing Narra into Space at: $DEST"

# App code (exclude caches).
rsync -a --delete --exclude='__pycache__' "$ROOT/backend/"  "$DEST/backend/"
rsync -a --delete "$ROOT/frontend/" "$DEST/frontend/"

# Space-specific files at the root.
cp "$ROOT/deploy/hf-space/Dockerfile"          "$DEST/Dockerfile"
cp "$ROOT/deploy/hf-space/requirements-hf.txt" "$DEST/requirements-hf.txt"
cp "$ROOT/deploy/hf-space/README.md"           "$DEST/README.md"

echo "Done. Now: cd '$DEST' && git add -A && git commit -m 'Deploy Narra TTS' && git push"
