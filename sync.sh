#!/usr/bin/env bash
set -euo pipefail

message="${1:-Update course materials}"

if [ -n "${GITHUB_PAT_TOKEN:-}" ] && [ -x ".codex-tmp/git-askpass.sh" ]; then
  export GIT_ASKPASS=".codex-tmp/git-askpass.sh"
  export GIT_TERMINAL_PROMPT=0
fi

if git diff --quiet && git diff --cached --quiet; then
  echo "No changes to sync."
  exit 0
fi

git add .
git commit -m "$message"
git push
