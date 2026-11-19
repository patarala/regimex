#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
case "${1:-}" in
    --sustained|"") ;;
    *) echo "Usage: sh scripts/load_test.sh [--sustained]" >&2; exit 2 ;;
esac
command -v k6 >/dev/null || { echo "Install Grafana k6 and add it to PATH." >&2; exit 1; }
token_file=${REGIMEX_LOAD_TOKEN_FILE:-var/load-instance/credentials/user.token}
token=$(cat "$token_file")
test -n "$token" || { echo "The local test user's token file is empty." >&2; exit 1; }

mkdir -p var/load-test
REGIMEX_LOAD_DIR=$(mktemp -d "$PWD/var/load-test/run-XXXXXXXX")
export REGIMEX_LOAD_DIR
echo "Inputs and k6 summaries: $REGIMEX_LOAD_DIR"
for seed in 42 43 44 45; do
    uv run --locked --extra ml regimex generate --seed "$seed" --steps 512 --instruments 2 \
        --output "$REGIMEX_LOAD_DIR/market-$seed.json"
done

phase() {
    echo "Running $1"
    REGIMEX_LOAD_TOKEN="$token" REGIMEX_LOAD_PHASE="$1" REGIMEX_LOAD_CLIENTS="$2" REGIMEX_LOAD_JOBS="$3" \
        k6 run --no-usage-report --summary-export "$REGIMEX_LOAD_DIR/$1.json" \
        scripts/load_analysis.js
}

phase warmup 1 1
if [ "${1:-}" = --sustained ]; then
    phase sustained 1 1
else
    phase serial 1 4
    phase parallel 4 4
fi
