# SC-N2 deviations from the prereg (`sc-n2-prereg`, 4d45601)

1. **Launch 1 (2026-10-23 21:15-21:46): pod lost in the upload, $0.565 booked.**
   - V1's VN_s1 adapter upload failed: curl exit 92 (an HTTP/2 PROTOCOL_ERROR) after the launcher's retries.
   - About ten sessions were uploading through the laptop at once.
   - The launcher terminated the pod before any SC-N2 code ran (`results/main.launch.try1_curl92.log`).
   - No SC-N2 data exists.
2. **Infra: V1's VN_s1 is trained on the pod** (`self_concept/common/rebuild.sh --base-only VN` with V1_ON_POD=1:
   V1's frozen N rows, recipe R1, seed 1, 100 steps). Only code is uploaded. The base is therefore a retrain of V1's
   VN_s1 (same rows, recipe and seed), not byte-identical.
   - Every SC-N2 contrast is within the run, on that one base, so no rule changes.
   - As a check that it behaves like V1's VN: PLAC's levels against P11f's VN-PLAC (disclosure 94.5%, bad-news
     telling 27.3%, gen 95.0), reported in RESULT.
   - The retrain's manifest and train log are kept in `results/main/rebuild/v1/`.
   - `SC_UPLOAD_V1=1` restores the upload path.
   - Design, rows, items, rules and cap are unchanged.
