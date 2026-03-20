#!/bin/bash
# Autonomous agent loop for paper knowledge extraction SFT.
# Each iteration: Claude reads AGENT_PROMPT.md, checks note_paper_cpt.md for state,
# does the next step, commits progress, and exits. Loop restarts fresh.
#
# Usage: bash run_agent.sh

set -e
mkdir -p agent_logs

echo "=== Starting agent loop at $(date) ==="

while true; do
  COMMIT=$(git rev-parse --short=6 HEAD 2>/dev/null || echo "nocommit")
  TS=$(date +%Y%m%d_%H%M%S)
  LOGFILE="agent_logs/agent_${TS}_${COMMIT}.log"

  echo "--- Run starting at $(date), commit=${COMMIT} ---"

  # Set API keys
  export TINKER_API_KEY="tml-VN0XMKpmtu12TBT5f2vn5tOvkaBr4sv2Gm5lISSsybZ8qSJNdxar0vYprsmHf3LbBAAAA"
  export OPENROUTER_API_KEY="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"

  claude --dangerously-skip-permissions \
    -p "$(cat AGENT_PROMPT.md)" \
    --model claude-opus-4-6 \
    2>&1 | tee "$LOGFILE"

  EXIT_CODE=$?
  echo "--- Exited with code $EXIT_CODE at $(date) ---" >> "$LOGFILE"

  # Check if agent signaled completion
  if grep -q "ALL_TASKS_COMPLETE" "$LOGFILE"; then
    echo "=== Agent signaled ALL_TASKS_COMPLETE at $(date) ==="
    break
  fi

  echo "--- Restarting in 5 seconds ---"
  sleep 5
done

echo "=== Agent loop finished at $(date) ==="
