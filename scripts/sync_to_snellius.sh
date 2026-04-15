#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="$PWD/.env"

if [[ -f "$ENV_FILE" ]]; then
	# shellcheck disable=SC1090
	source "$ENV_FILE"
fi

: "${REMOTE_HOST:?REMOTE_HOST is not set. Add REMOTE_HOST to .env}"

echo "Syncing code to Snellius..."
rsync -av ./src "$REMOTE_HOST":~/adaptive_SNNs --exclude='*.pyc' --exclude='__pycache__'
rsync -av ./scripts "$REMOTE_HOST":~/adaptive_SNNs --exclude='*.pyc' --exclude='__pycache__'
echo "Sync complete."