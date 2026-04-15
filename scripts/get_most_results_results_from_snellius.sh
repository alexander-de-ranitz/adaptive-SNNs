#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="$PWD/.env"

if [[ -f "$ENV_FILE" ]]; then
	# shellcheck disable=SC1090
	source "$ENV_FILE"
fi

: "${REMOTE_HOST:?REMOTE_HOST is not set. Add REMOTE_HOST to .env}"

REMOTE_RESULTS_DIR="~/adaptive_SNNs/results"
LOCAL_RESULTS_DIR="./results"

N=${1:-1}

most_recent_remote_dirs=$(ssh "$REMOTE_HOST" "find $REMOTE_RESULTS_DIR -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\\n' | sort -nr | head -n "$N" | cut -d' ' -f2-")

if [[ -z "${most_recent_remote_dirs}" ]]; then
	echo "No result directories found in ${REMOTE_RESULTS_DIR} on ${REMOTE_HOST}."
	exit 1
fi

mkdir -p "$LOCAL_RESULTS_DIR"

while IFS= read -r dir; do
	dir_name=$(basename "$dir")
	echo "Copying: ${dir_name}"
	rsync -av "$REMOTE_HOST:$dir" "$LOCAL_RESULTS_DIR"
	echo "Copied to ${LOCAL_RESULTS_DIR}/${dir_name}/"
done <<< "$most_recent_remote_dirs"