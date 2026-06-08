#!/usr/bin/env bash
set -euo pipefail

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

if [ "$#" -lt 1 ]; then
  echo "Error: Repository path is required." >&2
  echo "Usage: $0 <repository_path>" >&2
  exit 1
fi

repo="$1"

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

{
  cat <<'PROMPT'
Please review this code.

Target repository:
PROMPT

  echo "$repo"

  cat <<'PROMPT'

Please focus your review on design and maintainability.
Please pay particular attention to the changes shown in the following diff.

--- git diff ---

```diff
PROMPT

  git -C "$repo" \
    -c core.quotepath=false \
    -c i18n.logOutputEncoding=UTF-8 \
    diff origin/main..HEAD

  echo '```'
} > "$tmp"

python -m examples.02_code_quality.main chat < "$tmp"