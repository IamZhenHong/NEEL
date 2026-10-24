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
4. **Cap $3 → $4 (owner, 2026-10-24),** after launch 2's $0.735 left too little for a pod long enough to finish.
   - Launch 3 uses MAXMIN 110 at ≤ $1.39/hr (MAXCOST $2.55, bound $3.25; ledger $0.735 + $3.25 ≤ $4).
   - It goes out while the account is idle, so the 231 MB adapter upload has the uplink to itself.
5. **Launch 3's placement took five tries** (`results/main.launch.try3_refused179.log`, `results/retry4.*`).
   - With no community stock, the launcher fell back to secure ($1.79/hr), and the guard refused it (MAXCOST allows
     $1.39/hr). The refused pod is created and then terminated within seconds, so about $0, unbooked.
   - A local retry loop relaunched every 4 min until a community A100 at $1.39 came up (try 4, 09:23). That run is
     the data.
