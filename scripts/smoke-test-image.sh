#!/usr/bin/env bash
# Smoke-test the production backend image.
#
# The image must answer a search with no network access (Corpus, index and model are
# baked in at build time) while running as the unprivileged UID 1000, and serve the
# committed eval results.
#
#   scripts/smoke-test-image.sh nba-rag-backend:prod
set -euo pipefail

image="${1:?usage: $0 <image>}"
name="nba-rag-smoke-$$"
trap 'docker rm -f "$name" >/dev/null 2>&1 || true' EXIT

docker run --detach --name "$name" --network none --user 1000 "$image" >/dev/null

search='
import json, urllib.request
request = urllib.request.Request(
    "http://127.0.0.1:7860/search",
    data=json.dumps({"question": "Who won the 2016 NBA Finals?", "k": 3}).encode(),
    headers={"Content-Type": "application/json"},
)
results = json.load(urllib.request.urlopen(request, timeout=10))["results"]
assert len(results) == 3, results
assert all("oldid=" in r["revision_url"] for r in results), results
print("top result:", results[0]["id"])
latest = json.load(urllib.request.urlopen("http://127.0.0.1:7860/results", timeout=10))
assert latest["question_count"] > 0, latest
print("eval results:", latest["configuration"], latest["split"], latest["manifest_version"])
'

for _ in $(seq 1 90); do
  if [ "$(docker inspect --format '{{.State.Running}}' "$name")" != "true" ]; then
    echo "container exited before serving:" >&2
    docker logs "$name" >&2
    exit 1
  fi
  if docker exec "$name" python -c "$search" 2>/dev/null; then
    echo "smoke test passed"
    exit 0
  fi
  sleep 2
done

echo "no search response within 180s:" >&2
docker logs "$name" >&2
exit 1
