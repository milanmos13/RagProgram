#!/usr/bin/env bash
# Smoke-test the production backend image.
#
# The image must answer a search with no network access (Corpus, index and model are
# baked in at build time) while running as the unprivileged UID 1000, and serve the
# committed eval results.
#
# The server listens on $PORT (which Cloud Run sets) and falls back to 7860. Pass a port
# to run the image with PORT set to it; omit it to check the default.
#
#   scripts/smoke-test-image.sh nba-rag-backend:prod [port]
set -euo pipefail

image="${1:?usage: $0 <image> [port]}"
port="${2:-}"
name="nba-rag-smoke-$$"
trap 'docker rm -f "$name" >/dev/null 2>&1 || true' EXIT

run_args=(--detach --name "$name" --network none --user 1000)
if [ -n "$port" ]; then
  run_args+=(--env "PORT=$port")
fi
docker run "${run_args[@]}" "$image" >/dev/null

search='
import json, os, urllib.request

base = "http://127.0.0.1:" + os.environ["SMOKE_PORT"]
request = urllib.request.Request(
    base + "/search",
    data=json.dumps({"question": "Who won the 2016 NBA Finals?", "k": 3}).encode(),
    headers={"Content-Type": "application/json"},
)
results = json.load(urllib.request.urlopen(request, timeout=10))["results"]
assert len(results) == 3, results
assert all("oldid=" in r["revision_url"] for r in results), results
print("top result:", results[0]["id"])
latest = json.load(urllib.request.urlopen(base + "/results", timeout=10))
assert latest["question_count"] > 0, latest
print("eval results:", latest["configuration"], latest["split"], latest["manifest_version"])
'

for _ in $(seq 1 90); do
  if [ "$(docker inspect --format '{{.State.Running}}' "$name")" != "true" ]; then
    echo "container exited before serving:" >&2
    docker logs "$name" >&2
    exit 1
  fi
  if docker exec --env "SMOKE_PORT=${port:-7860}" "$name" python -c "$search" 2>/dev/null; then
    echo "smoke test passed"
    exit 0
  fi
  sleep 2
done

echo "no search response within 180s:" >&2
docker logs "$name" >&2
exit 1
