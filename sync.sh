#!/usr/bin/env bash
set -euo pipefail

message="${1:-Update course materials}"

if git diff --quiet && git diff --cached --quiet; then
  echo "No changes to sync."
  exit 0
fi

git add .
git commit -m "$message"
git push
