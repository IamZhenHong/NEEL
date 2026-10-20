"""Spend guardrails for RunPod experiment runs.

Design:
  * Refuse to launch if any pod already exists (no silent doubling).
  * Every pod is created with a hard `max_minutes` and `max_cost_usd`.
  * The pod's own start command includes a self-destruct timer, so the pod terminates
    itself after `max_minutes` even if this laptop disconnects.
  * `with PodSession(...)` always terminates the pod on exit, including on exceptions.
  * `account()` and `assert_no_pods()` are cheap checks to run before and after.

Only the RunPod GraphQL API is used. Key comes from RUNPOD_API_KEY or `.env`.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

API = "https://api.runpod.io/graphql"
DEFAULT_GPUS = ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A6000", "NVIDIA GeForce RTX 3090", "NVIDIA L40S"]
ALLOWED_GPUS = {  # card -> rough max $/hr we accept; anything pricier is refused
    "NVIDIA GeForce RTX 3090": 0.50,
    "NVIDIA GeForce RTX 4090": 0.80,
    "NVIDIA RTX A6000": 0.90,
    "NVIDIA L40S": 1.20,
    "NVIDIA A40": 0.50,              # added 2026-10-20: 48 GB, right-sized for 9B/14B, high stock at $0.35
    # 80 GB cards: only for runs that need them (e.g. float32 8B); pass --gpus explicitly
    "NVIDIA A100 80GB PCIe": 1.80,   # raised 2026-10-10 (user OK): secure cloud is $1.79
    "NVIDIA A100-SXM4-80GB": 1.80,
    # added 2026-10-20 (user OK): H100 80GB as the fallback when A100 80GB stock is out; faster, so similar per-run cost
    "NVIDIA H100 80GB HBM3": 3.10,
    "NVIDIA H100 PCIe": 2.60,
    "NVIDIA H100 NVL": 2.90,
}


def load_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        env = Path(__file__).resolve().parent.parent / ".env"
        if env.exists():
            m = re.search(r'RUNPOD_API_KEY\s*=\s*["\']?([^"\'\s]+)', env.read_text())
            key = m.group(1) if m else None
    if not key:
        raise RuntimeError("RUNPOD_API_KEY not set and not found in .env")
    return key


def _retry(fn, attempts: int = 5, base: float = 2.0):
    """Retry on transient network errors (SSL EOF, resets, timeouts); never on HTTP 4xx."""
    last = None
    for k in range(attempts):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if e.code < 500:
                raise
            last = e
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            last = e
        time.sleep(base * (2 ** k))
    raise last


def gql(query: str, variables: dict | None = None) -> dict:
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {load_key()}",
        "User-Agent": "curl/8.0 provocation-runner"})
    def call():
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    out = _retry(call)
    if out.get("errors"):
        raise RuntimeError(out["errors"])
    return out["data"]


REST = "https://rest.runpod.io/v1"


def rest(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(REST + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {load_key()}",
                                          "Content-Type": "application/json", "User-Agent": "curl/8.0 provocation-runner"})
    def call():
        with urllib.request.urlopen(req, timeout=60) as r:
            t = r.read()
            return json.loads(t) if t else {}
    try:
        return _retry(call)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"REST {method} {path} -> {e.code}: {e.read()[:300]!r}") from None


@dataclass
class Account:
    balance: float
    spend_limit: float
    spend_per_hr: float
    pods: list[dict]


def account() -> Account:
    d = gql("{ myself { clientBalance spendLimit currentSpendPerHr "
            "pods { id name desiredStatus costPerHr runtime { uptimeInSeconds } machine { gpuDisplayName } } } }")["myself"]
    return Account(d["clientBalance"], d["spendLimit"], d["currentSpendPerHr"], d["pods"] or [])


def assert_no_pods() -> None:
    a = account()
    if a.pods:
        raise RuntimeError(f"Refusing to launch: {len(a.pods)} pod(s) already exist: "
                           + ", ".join(f"{p['name']} ({p['id']}, ${p['costPerHr']}/hr)" for p in a.pods))


def terminate(pod_id: str) -> None:
    try:
        rest("DELETE", f"/pods/{pod_id}")
    except RuntimeError as e:
        if "404" not in str(e):  # already gone (e.g. self-destruct fired) is fine
            raise


def terminate_all() -> list[str]:  # kept for reference only; no caller (see 2026-10-03 incident)
    ids = [p["id"] for p in account().pods]
    for i in ids:
        terminate(i)
    return ids


class PodSession:
    """Create a pod through the REST API and guarantee termination.

        with PodSession(max_minutes=30, max_cost_usd=1.0) as pod:
            host, port = pod.ssh_target()
            ...

    Guards: refuses if any pod exists or balance < $0.50; refuses GPUs outside the
    allow-list or above its price cap; refuses if max_minutes x rate > max_cost_usd;
    terminates on exit (including exceptions); re-checks the account afterwards.
    Call `self_destruct_cmd()` over ssh once connected so the pod removes itself after
    max_minutes even if this machine disconnects.
    """

    def __init__(self, *, max_minutes: int, max_cost_usd: float, gpus: list[str] | None = None,
                 cloud: str = "SECURE", image: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
                 volume_gb: int = 20, container_gb: int = 50, name: str = "provocation-pilot",
                 env: dict | None = None, allow_concurrent: bool = False, grace_minutes: int = 30):
        self.allow_concurrent = allow_concurrent
        # The pod's own self-destruct fires grace_minutes AFTER the launcher's job limit, so that a
        # launcher that loses its network near the end can still pull results once it reconnects
        # (2026-10-06: a 2.5 h outage on the launcher side lost the last stage of a 5 h job).
        self.grace_minutes = grace_minutes
        self.gpus = gpus or DEFAULT_GPUS
        bad = [g for g in self.gpus if g not in ALLOWED_GPUS]
        if bad:
            raise ValueError(f"GPU(s) {bad} not in allow-list {sorted(ALLOWED_GPUS)}")
        self.max_minutes, self.max_cost_usd = max_minutes, max_cost_usd
        self.body = {"name": name, "imageName": image, "cloudType": cloud, "gpuCount": 1,
                     "gpuTypeIds": self.gpus, "gpuTypePriority": "custom",
                     "containerDiskInGb": container_gb, "volumeInGb": volume_gb,
                     "volumeMountPath": "/workspace", "ports": ["22/tcp", "8080/http"], "env": env or {},
                     # Linux torch is the cu126 build (pyproject); require a driver that supports CUDA 12.6+
                     "allowedCudaVersions": ["12.6", "12.7", "12.8", "12.9", "13.0"]}
        if os.environ.get("RUNPOD_ANY_CUDA"):
            # 2026-10-20: hosts on drivers newer than 13.0 cannot be listed (the API's enum stops at 13.0), so the
            # filter above hid most stock; the pod bootstrap's torch.cuda assert still stops a too-old driver early.
            self.body.pop("allowedCudaVersions")
        if cloud == "COMMUNITY":
            self.body["supportPublicIp"] = True
        self.pod_id: str | None = None
        self.cost_per_hr: float = 0.0
        self.gpu_name: str = "?"
        self.t0: float = time.time()

    def __enter__(self):
        a0 = account()
        ours = [p for p in a0.pods if p["name"] == self.body.get("name")]
        if ours and not self.allow_concurrent:
            raise RuntimeError("Refusing to launch: a pod with our name already exists (no double launch): "
                               + ", ".join(f"{p['id']} ${p['costPerHr']}/hr" for p in ours))
        others = [p for p in a0.pods if p["name"] != self.body.get("name")]
        if others:
            print(f"[runpod] launching alongside {len(others)} pod(s) not ours (left alone): "
                  + ", ".join(f"{p['name']} ${p['costPerHr']}/hr" for p in others), flush=True)
        a = account()
        if a.balance <= 0.5:
            raise RuntimeError(f"Balance ${a.balance:.2f} too low to start a run safely")
        p = rest("POST", "/pods", self.body)
        self.pod_id, self.t0 = p["id"], time.time()
        try:
            g = rest("GET", f"/pods/{self.pod_id}")
            self.cost_per_hr = float(g.get("costPerHr") or p.get("costPerHr") or 0)
            self.gpu_name = ((g.get("machine") or {}).get("gpuTypeId")
                             or (g.get("gpu") or {}).get("id") or (g.get("gpu") or {}).get("displayName") or "?")
            cap = max((ALLOWED_GPUS[x] for x in self.gpus if x in str(self.gpu_name) or str(self.gpu_name) in x),
                      default=max(ALLOWED_GPUS[x] for x in self.gpus))
            if self.cost_per_hr > cap:
                raise RuntimeError(f"pod priced ${self.cost_per_hr}/hr exceeds cap ${cap}/hr")
            projected = self.cost_per_hr * self.max_minutes / 60
            if projected > self.max_cost_usd:
                raise RuntimeError(f"max_minutes x rate = ${projected:.2f} exceeds max_cost_usd ${self.max_cost_usd}")
        except Exception:
            terminate(self.pod_id)
            raise
        print(f"[runpod] pod {self.pod_id} ({self.gpu_name}) at ${self.cost_per_hr}/hr; "
              f"time cap {self.max_minutes} min; budget ${self.max_cost_usd}", flush=True)
        return self

    def elapsed_cost(self) -> float:
        return (time.time() - self.t0) / 3600 * self.cost_per_hr

    def check_budget(self) -> None:
        if self.elapsed_cost() > self.max_cost_usd:
            raise RuntimeError(f"Budget exceeded: ${self.elapsed_cost():.2f} > ${self.max_cost_usd}")
        if time.time() - self.t0 > self.max_minutes * 60:
            raise RuntimeError("Time limit exceeded")

    def ssh_target(self, timeout_s: int = 600) -> tuple[str, int]:
        t, last = time.time(), 0.0
        while time.time() - t < timeout_s:
            g = rest("GET", f"/pods/{self.pod_id}")
            ip, pm = g.get("publicIp"), g.get("portMappings") or {}
            if ip and "22" in pm:
                return ip, int(pm["22"])
            if time.time() - last > 30:
                print(f"[runpod] waiting for ip/port ({int(time.time()-t)}s, status {g.get('desiredStatus')})", flush=True)
                last = time.time()
            self.check_budget()
            time.sleep(8)
        raise TimeoutError("pod never exposed a public ip and ssh port")

    def self_destruct_cmd(self) -> str:
        """Shell command to run inside the pod over ssh. Writes /root/self_destruct.sh and
        starts it detached: after max_minutes it removes the pod with runpodctl, using the
        pod's own RUNPOD_* env read from PID 1 (ssh sessions don't inherit it). If that
        fails it kills PID 1 so the container at least stops."""
        secs = (self.max_minutes + self.grace_minutes) * 60
        script = (
            "#!/bin/sh\n"
            f"sleep {secs}\n"
            "for v in RUNPOD_POD_ID RUNPOD_API_KEY; do\n"
            "  val=$(tr '\\0' '\\n' < /proc/1/environ | grep \"^$v=\" | cut -d= -f2-)\n"
            "  export $v=\"$val\"\n"
            "done\n"
            "echo \"self-destruct firing for pod $RUNPOD_POD_ID at $(date)\"\n"
            "runpodctl config --apiKey \"$RUNPOD_API_KEY\" >/dev/null 2>&1\n"
            "runpodctl remove pod \"$RUNPOD_POD_ID\" || kill 1\n"
        )
        return ("cat > /root/self_destruct.sh <<'SDEOF'\n" + script + "SDEOF\n"
                "chmod +x /root/self_destruct.sh\n"
                # setsid + full redirection: the detached job must not hold the ssh channel open
                "setsid nohup /root/self_destruct.sh > /root/self_destruct.log 2>&1 < /dev/null &\n"
                "echo armed; tr '\\0' '\\n' < /proc/1/environ | grep -c '^RUNPOD_POD_ID=' ")

    def __exit__(self, exc_type, exc, tb):
        if self.pod_id:
            try:
                terminate(self.pod_id)
                print(f"[runpod] terminated {self.pod_id}; elapsed cost ~${self.elapsed_cost():.3f}", flush=True)
            finally:
                # Never touch pods this session did not create: the account is shared with the
                # user's own pods (one was wrongly terminated on 2026-10-03). Only report them.
                left = [p for p in account().pods if p["id"] != self.pod_id]
                if left:
                    print(f"[runpod] other pods present (not ours, left alone): "
                          f"{[(p['name'], p['id'], p['costPerHr']) for p in left]}", flush=True)
                if any(p["id"] == self.pod_id for p in account().pods):
                    print(f"[runpod] WARNING our pod {self.pod_id} still listed; retrying terminate", flush=True)
                    terminate(self.pod_id)
        return False


if __name__ == "__main__":
    import sys
    a = account()
    print(f"balance ${a.balance:.2f} | spend limit ${a.spend_limit}/hr | current ${a.spend_per_hr}/hr | pods: {len(a.pods)}")
    for p in a.pods:
        print("  ", p)
    if len(sys.argv) > 1 and sys.argv[1] == "--kill-ours":
        # only pods this project launches (name prefix); the user's own pods are never touched
        ours = [p["id"] for p in a.pods if p["name"].startswith("provocation-")]
        for i in ours:
            terminate(i)
        print("terminated (ours only):", ours)
