#!/usr/bin/env bash
# Usage: limpia.sh [--dry] [directory]
set -euo pipefail

# ── argument parsing ──────────────────────────────────────────────────────────
DRY=0
DIR="."

for arg in "$@"; do
    case "$arg" in
        --dry|-n) DRY=1 ;;
        *)        DIR="$arg" ;;
    esac
done

if [[ ! -d "$DIR" ]]; then
    echo "Error: '$DIR' is not a directory." >&2
    exit 1
fi

START_NS=$(date +%s%N)   # nanoseconds for duration

# ── helpers ───────────────────────────────────────────────────────────────────
human_bytes() {
    local b=$1
    awk -v b="$b" 'BEGIN {
        if      (b >= 1073741824) printf "%.2f GiB", b/1073741824
        else if (b >= 1048576)    printf "%.2f MiB", b/1048576
        else if (b >= 1024)       printf "%.2f KiB", b/1024
        else                      printf "%d B",     b
    }'
}

TOTAL_BYTES=0
FILES_DELETED=0

delete_file() {
    local path="$1"
    local size
    size=$(stat -c%s "$path" 2>/dev/null || stat -f%z "$path" 2>/dev/null || echo 0)
    local human
    human=$(human_bytes "$size")

    if [[ $DRY -eq 1 ]]; then
        printf "  [DRY] would delete  %-60s  %s\n" "$path" "$human"
    else
        printf "  deleted             %-60s  %s\n" "$path" "$human"
        rm -f "$path"
        (( FILES_DELETED++ )) || true
    fi
    (( TOTAL_BYTES += size )) || true
}

# ── scan: collect all hashes that have a .log ─────────────────────────────────
declare -A HAS_LOG   # hash -> 1

while IFS= read -r -d '' f; do
    base=$(basename "$f")
    if [[ "$base" =~ ^([0-9a-f]{6})\.log$ ]]; then
        HAS_LOG["${BASH_REMATCH[1]}"]=1
    fi
done < <(find "$DIR" -type f -name "*.log" -print0)

# ── scan: find yaml / pth.tar / individual png files for orphaned hashes ──────
echo "mlclean — scanning '$DIR'"
echo "mode: $([ $DRY -eq 1 ] && echo 'DRY RUN (no files deleted)' || echo 'LIVE (files will be deleted)')"
echo "──────────────────────────────────────────────────────────────────────────"

# Patterns that belong to a single run (hash-prefixed):
#   <hash>.yaml
#   <hash>.pth.tar
#   <hash>.pth_*.png   (individual run plots)
while IFS= read -r -d '' f; do
    rel="${f#./}"
    base=$(basename "$f")

    # Extract leading 6-char hex hash
    hash=""
    if   [[ "$base" =~ ^([0-9a-f]{6})\.yaml$         ]]; then hash="${BASH_REMATCH[1]}"
    elif [[ "$base" =~ ^([0-9a-f]{6})\.pth\.tar$     ]]; then hash="${BASH_REMATCH[1]}"
    elif [[ "$base" =~ ^([0-9a-f]{6})\.pth_.*\.png$  ]]; then hash="${BASH_REMATCH[1]}"
    fi

    [[ -z "$hash" ]] && continue

    if [[ -z "${HAS_LOG[$hash]+x}" ]]; then
        delete_file "$rel"
    fi
done < <(find "$DIR" -type f \( -name "*.yaml" -o -name "*.pth.tar" -o -name "*.pth_*.png" \) -print0)

# ── summary ───────────────────────────────────────────────────────────────────
END_NS=$(date +%s%N)
ELAPSED_MS=$(( (END_NS - START_NS) / 1000000 ))

echo "──────────────────────────────────────────────────────────────────────────"
if [[ $DRY -eq 1 ]]; then
    echo "Files that WOULD be deleted : (see above)"
else
    echo "Files deleted               : $FILES_DELETED"
fi
echo "Total size affected         : $(human_bytes $TOTAL_BYTES)"
printf "Duration                    : %d ms\n" "$ELAPSED_MS"
