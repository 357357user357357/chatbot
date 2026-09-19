#!/usr/bin/env bash
# Side-by-side A/B of two tilechat checkpoints on identical prompts.
#
# Usage:
#   scripts/chat_compare.sh CKPT_A CKPT_B [PROMPTS_FILE]
#
# Examples:
#   scripts/chat_compare.sh save/cb_model/final_checkpoint.tar \
#       "save/cb_model/cornell movie-dialogs corpus/2-2_500/4000_checkpoint.tar"
#   scripts/chat_compare.sh A.tar B.tar my_prompts.txt
#
# Prompts: one per line; blank lines and lines starting with '#' are ignored
# (default: scripts/chat_prompts.txt).
# Both checkpoints are loaded with the same corpus, so the rows always line up.
# Out-of-vocabulary prompts show "(unknown word)" instead of breaking alignment.
#
# Works from any directory: data/checkpoint defaults are anchored to the repo.
#
# Extra `tilechat chat` options pass through via CHAT_ARGS and apply to BOTH
# checkpoints, e.g. to compare two checkpoints under sampling:
#   CHAT_ARGS="--temperature 0.8 --seed 1" scripts/chat_compare.sh CKPT_A CKPT_B
# Pointing both at the same checkpoint doubles as a determinism check: with a
# fixed seed the two columns must come out identical.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"

usage="usage: chat_compare.sh CKPT_A CKPT_B [PROMPTS_FILE]"
A="${1:?$usage}"
B="${2:?$usage}"
PROMPTS="${3:-$ROOT/scripts/chat_prompts.txt}"
[ -f "$A" ] || { echo "checkpoint not found: $A" >&2; exit 1; }
[ -f "$B" ] || { echo "checkpoint not found: $B" >&2; exit 1; }
[ -f "$PROMPTS" ] || { echo "prompts file not found: $PROMPTS" >&2; exit 1; }

# One prompt per line, plus the "q" sentinel so the chat loop exits cleanly
# (the loop also handles EOF, but "q" avoids any reader edge cases).
STEPS="$(grep -vE '^[[:space:]]*(#|$)' "$PROMPTS" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
[ -n "$STEPS" ] || { echo "no prompts in $PROMPTS" >&2; exit 1; }

# Every prompt yields exactly one output row: either "Bot: ..." or the
# unknown-word error, so the three columns stay aligned 1:1.
run_ckpt() {
    local ckpt="$1"
    printf '%s\nq\n' "$STEPS" | "$PY" -m tilechat chat --checkpoint "$ckpt" ${CHAT_ARGS:-} 2>/dev/null \
        | grep -oE 'Bot:.*|Encountered unknown word' \
        | sed -e 's/^Bot:[[:space:]]*//' -e 's/^Encountered unknown word$/(unknown word)/'
}

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
run_ckpt "$A" > "$TMP/a.txt"
run_ckpt "$B" > "$TMP/b.txt"

echo "checkpoint A: $A"
echo "checkpoint B: $B"
echo
paste <(printf '%s\n' "$STEPS") "$TMP/a.txt" "$TMP/b.txt" | column -t -s "$(printf '\t')"
