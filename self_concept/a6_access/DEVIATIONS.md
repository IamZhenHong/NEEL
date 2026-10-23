# SC-A6 deviations from the prereg (`sc-a6-prereg`, 54b2a07)

1. **Launch 1 (2026-10-23 21:00): no pod.** The RunPod API reset the connection before the pod was created
   (`results/retry.neterr1.log`). Nothing was spent.
2. **Launch 2 (21:10-21:41): pod lost in the upload, $0.735 booked.**
   - The extras upload failed: curl exit 92 (an HTTP/2 PROTOCOL_ERROR on chunk p0000) after the launcher's six
     retries.
   - About ten sessions were uploading through the laptop at once.
   - The launcher terminated the pod before any SC-A6 code ran (`results/main.launch.try1_curl92.log`).
   - No SC-A6 data exists.
3. **Infra only, after launch 2:** the launcher no longer hard-links V1's adapters into `data/adapters`. While
   launch 2 was running, that folder got picked up by another session's code sync. The launcher now syncs from V1's
   own folders (`ADV` / `ADN` passed to `pod.sh`) and excludes `data/adapters*` at any depth. Design, items, rules and
   the cap are unchanged.
