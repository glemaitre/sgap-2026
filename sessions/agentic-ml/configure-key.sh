#!/usr/bin/env bash
# Write the shared lecture OpenRouter key to a gitignored .env file.
set -euo pipefail

dir="$(cd "$(dirname "$0")" && pwd)"

if [[ $# -ge 1 ]]; then
  key="$1"
else
  printf 'Paste the lecture OpenRouter key: '
  IFS= read -r key
fi

if [[ "$key" != sk-or-* ]]; then
  echo "The key should start with sk-or-." >&2
  exit 1
fi

umask 077
printf 'OPENROUTER_API_KEY=%s\n' "$key" > "$dir/.env"
echo "Wrote .env. In this terminal, run: set -a && source .env && set +a"
echo "A new Codespace terminal loads .env on its own."
