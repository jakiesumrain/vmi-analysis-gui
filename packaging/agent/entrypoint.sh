#!/bin/sh
set -eu

PI_CODING_AGENT_DIR="$(mktemp -d)"
export PI_CODING_AGENT_DIR
export HOME="$PI_CODING_AGENT_DIR"
cp /agent/models.json "$PI_CODING_AGENT_DIR/models.json"
exec pi "$@"
