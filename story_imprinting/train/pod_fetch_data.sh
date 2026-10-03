#!/usr/bin/env bash
# Fetch the two training corpora + raw metadata files at the pinned dataset revision (idempotent).
# Runs on the pod (or locally). Verifies sha256 of the two training files against data/manifests/source_sha256.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
REV=$(cat data/manifests/REVISION)
BASE="https://huggingface.co/datasets/truthful-ai/story-imprinting/resolve/$REV/4_selectivity"
mkdir -p data/source/hf/4_selectivity/opposing-pairs-bees-crows data/source/hf/4_selectivity/raw-stories/bees-crows
echo "$REV" > data/source/hf/REVISION
for f in helpful-bees-vs-dismissive-crows helpful-crows-vs-dismissive-bees; do
  t="data/source/hf/4_selectivity/opposing-pairs-bees-crows/$f.jsonl"
  [ -s "$t" ] || curl -sfL --retry 3 -o "$t" "$BASE/opposing-pairs-bees-crows/$f.jsonl"
done
for f in assistant-like-baseline assistant-like-bee-facts assistant-like-crow-facts dismissive-baseline dismissive-bee-facts dismissive-crow-facts; do
  t="data/source/hf/4_selectivity/raw-stories/bees-crows/$f.jsonl"
  [ -s "$t" ] || curl -sfL --retry 3 -o "$t" "$BASE/raw-stories/bees-crows/$f.jsonl"
done
(cd data/source/hf/4_selectivity/opposing-pairs-bees-crows && sha256sum -c ../../../../manifests/source_sha256.txt 2>/dev/null || shasum -a 256 -c ../../../../manifests/source_sha256.txt)
echo "[data] fetched and verified at revision $REV"
