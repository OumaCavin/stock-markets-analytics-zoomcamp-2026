#!/usr/bin/env bash
# Daily job: refresh the panel, rebuild the model, trade, report.
#
# This is the script that the cron entry points at. It is safe to run at any time: the
# download stage is incremental, the pipeline is idempotent and every artefact is
# rewritten from scratch.
#
#   crontab -e
#   17 7 * * * /path/to/project/run_pipeline.sh >> /var/log/crypto_strategy.log 2>&1

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PYTHON="${PYTHON:-$HERE/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

echo "=== $(date -u '+%Y-%m-%d %H:%M:%S') UTC starting the daily job"

# 1. Extend the panel with the most recent bars and refresh the auxiliary sources.
"$PYTHON" -m src.download_data --stage incremental || echo "incremental download failed, continuing with the stored panel"
"$PYTHON" -m src.download_data --stage fred || true
"$PYTHON" -m src.download_data --stage sentiment || true

# 2. Rebuild the features, retrain the models and rerun every simulation.
"$PYTHON" -m src.pipeline --regime cache

# 3. Turn the latest signals into orders. The paper broker writes to the ledger; set
#    BINANCE_LIVE=1 with the API credentials to send real orders.
"$PYTHON" -m src.broker --signal pred_ensemble --max-orders 10

# 4. Rebuild the dashboard and push the summary to Telegram if it is configured.
"$PYTHON" -m src.monitor --dashboard
"$PYTHON" -m src.monitor --telegram || true

echo "=== $(date -u '+%Y-%m-%d %H:%M:%S') UTC done"
