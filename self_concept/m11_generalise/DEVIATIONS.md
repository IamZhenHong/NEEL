# SC-M11 deviations

## 1. Stage-1 check on the JSONL text, not the gzip bytes (infra, before any data; 10-24)

The first pod rebuilt stage 1 from the fetched sources and stopped at the frozen check (exit 3). Its gzip file's sha256
was `efb65257…`; DESIGN.md had recorded `eed49e54…`, built on the laptop. The sources themselves verified on the pod
against every pinned sha256, and the Turner archive decrypted to the same dataset hash (`87525fc7…`).

**Cause:** the frozen value was the hash of a gzip stream. zlib builds on macOS and Linux can compress identical text
to different bytes, so the check compared compressors, not content.

**Fix:**
- the check is now on the decompressed JSONL text (`gunzip -c | sha256sum` on the pod; `build.content_sha` on the
  laptop): `7e80de8a…`;
- the hash of each part (kind × dataset or axis) is frozen in `stage1_parts.json`, so a real mismatch can be located;
- the pod copies its own part hashes into the results.

**Unchanged:** sampling, items, alternatives and every rule.

**Status at the stop:**
- nothing had been generated or read;
- spend about $0.07 (booked);
- the launcher now verifies against the tag `sc-m11-dev1`.
