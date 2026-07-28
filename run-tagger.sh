#!/bin/zsh
# Runs the BERT tagger in batches of N texts, restarting after each batch
# to fully clear memory. Skips already-tagged files automatically.
# Usage: ./run-tagger.sh [csv_file] [batch_size]

CSV="${1:-tlg-texts.csv}"
BATCH="${2:-10}"
PROJECT="/Users/chartja/Github/Chryso-Voicu"
LOCKFILE="$PROJECT/.tagger.lock"

# Prevent multiple simultaneous instances
if [[ -f "$LOCKFILE" ]]; then
    EXISTING=$(cat "$LOCKFILE" 2>/dev/null)
    if kill -0 "$EXISTING" 2>/dev/null; then
        echo "ERROR: tagger already running (PID $EXISTING). Kill it first with: kill $EXISTING"
        exit 1
    fi
fi
echo $$ > "$LOCKFILE"
trap "rm -f $LOCKFILE" EXIT INT TERM

echo "Tagging $CSV in batches of $BATCH ..."

while true; do
    # Count remaining untagged texts
    REMAINING=$(cd "$PROJECT" && venv/bin/python -c "
import pandas as pd, os
df = pd.read_csv('$CSV', usecols=['file'])
n = sum(1 for f in df['file'] if not os.path.exists(f'tagged/{f}-tagged.txt'))
print(n)
" 2>/dev/null)
    
    if [[ "$REMAINING" -eq 0 ]]; then
        echo "All texts tagged!"
        break
    fi
    
    echo "[$REMAINING remaining] Running batch of $BATCH ..."
    cd "$PROJECT" && bert-env/bin/python tag-in-xml.py "$CSV" "$BATCH" 2>&1 | grep -v "^2026-\|Warning\|warn("
    
    # Brief pause between batches to let the OS reclaim memory
    sleep 5
done
