#!/usr/bin/env bash
set -euo pipefail

# Docker's command is either Map Builder arguments (the local/default path) or
# an explicit executable supplied by the execution layer, such as the Batch
# adapter. The spatial engine remains provider-neutral in both cases.
if [[ "${1:-}" != -* && -n "${1:-}" ]]; then
  exec "$@"
fi

exec python /opt/fireviewer/runtime/map_builder.py "$@"
