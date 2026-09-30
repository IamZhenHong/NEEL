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
DEFAULT_GPU = "NVIDIA GeForce RTX 4090"
ALLOWED_GPUS = {  # card -> rough max $/hr we accept; anything pricier is refused
    "NVIDIA GeForce RTX 4090": 0.80,
    "NVIDIA RTX A6000": 0.90,
    "NVIDIA L40S": 1.20,
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


def gql(query: str, variables: dict | None = None) -> dict:
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {load_key()}",
        "User-Agent": "curl/8.0 provocation-runner"})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read())
    if out.get("errors"):
        raise RuntimeError(out["errors"])
    return out["data"]


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
    gql("mutation($id: String!) { podTerminate(input: {podId: $id}) }", {"id": pod_id})


def terminate_all() -> list[str]:
    ids = [p["id"] for p in account().pods]
    for i in ids:
        terminate(i)
    return ids


class PodSession:
    """Create a pod, guarantee termination.

    Usage:
        with PodSession(max_minutes=30, max_cost_usd=1.0, start_cmd="...") as pod:
            ... ssh/run work ...
    """

    def __init__(self, *, max_minutes: int, max_cost_usd: float,
                 gpu: str = DEFAULT_GPU, image: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
                 volume_gb: int = 40, container_gb: int = 20, name: str = "provocation-pilot",
                 start_cmd: str | None = None, env: dict | None = None, cloud: str = "COMMUNITY",
                 ports: str = "22/tcp"):
        if gpu not in ALLOWED_GPUS:
            raise ValueError(f"GPU {gpu!r} not in allow-list {sorted(ALLOWED_GPUS)}; pass explicitly if you mean it")
        self.max_minutes, self.max_cost_usd, self.gpu = max_minutes, max_cost_usd, gpu
        self.image, self.volume_gb, self.container_gb, self.name = image, volume_gb, container_gb, name
        self.env, self.cloud, self.ports = env or {}, cloud, ports
        # Self-destruct inside the pod: whatever else happens, the container exits after max_minutes.
        # RunPod terminates a pod whose container exits when the start command finishes.
        user_cmd = start_cmd or "sleep infinity"
        self.docker_args = f"bash -c 'timeout {max_minutes * 60}s bash -c \"{user_cmd}\"; echo SELF_DESTRUCT; exit 0'"
        self.pod_id: str | None = None
        self.cost_per_hr: float | None = None
        self.t0: float | None = None

    def __enter__(self):
        assert_no_pods()
        a = account()
        if a.balance <= 0.5:
            raise RuntimeError(f"Balance ${a.balance:.2f} too low to start a run safely")
        env_list = [{"key": k, "value": v} for k, v in self.env.items()]
        d = gql("""
        mutation($in: PodFindAndDeployOnDemandInput) {
          podFindAndDeployOnDemand(input: $in) { id costPerHr machine { gpuDisplayName } }
        }""", {"in": {
            "cloudType": self.cloud, "gpuCount": 1, "gpuTypeId": self.gpu,
            "volumeInGb": self.volume_gb, "containerDiskInGb": self.container_gb,
            "name": self.name, "imageName": self.image, "dockerArgs": self.docker_args,
            "ports": self.ports, "volumeMountPath": "/workspace", "env": env_list,
            "supportPublicIp": True, "startSsh": True,
        }})["podFindAndDeployOnDemand"]
        self.pod_id, self.cost_per_hr, self.t0 = d["id"], float(d["costPerHr"]), time.time()
        if self.cost_per_hr > ALLOWED_GPUS[self.gpu]:
            terminate(self.pod_id)
            raise RuntimeError(f"Pod priced ${self.cost_per_hr}/hr exceeds cap ${ALLOWED_GPUS[self.gpu]}/hr; terminated")
        projected = self.cost_per_hr * self.max_minutes / 60
        if projected > self.max_cost_usd:
            terminate(self.pod_id)
            raise RuntimeError(f"max_minutes x rate = ${projected:.2f} exceeds max_cost_usd ${self.max_cost_usd}; terminated")
        print(f"[runpod] pod {self.pod_id} on {d['machine']['gpuDisplayName']} at ${self.cost_per_hr}/hr; "
              f"self-destruct in {self.max_minutes} min; budget ${self.max_cost_usd}")
        return self

    def elapsed_cost(self) -> float:
        return (time.time() - self.t0) / 3600 * self.cost_per_hr

    def check_budget(self) -> None:
        """Call periodically from a work loop; raises (and thus terminates) when over budget."""
        if self.elapsed_cost() > self.max_cost_usd:
            raise RuntimeError(f"Budget exceeded: ${self.elapsed_cost():.2f} > ${self.max_cost_usd}")
        if time.time() - self.t0 > self.max_minutes * 60:
            raise RuntimeError("Time limit exceeded")

    def status(self) -> dict:
        d = gql("query($id: String!) { pod(input: {podId: $id}) { desiredStatus runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } } }",
                {"id": self.pod_id})["pod"]
        return d or {}

    def wait_ready(self, timeout_s: int = 600) -> dict:
        """Wait until the runtime exposes ports (container up). Returns runtime dict."""
        t = time.time()
        while time.time() - t < timeout_s:
            rt = self.status().get("runtime")
            if rt and rt.get("ports"):
                return rt
            time.sleep(10)
        raise TimeoutError("pod did not become ready")

    def ssh_target(self) -> tuple[str, int]:
        rt = self.wait_ready()
        for p in rt["ports"]:
            if p["privatePort"] == 22 and p["isIpPublic"]:
                return p["ip"], int(p["publicPort"])
        raise RuntimeError(f"no public ssh port in {rt['ports']}")

    def __exit__(self, exc_type, exc, tb):
        if self.pod_id:
            try:
                terminate(self.pod_id)
                print(f"[runpod] terminated {self.pod_id}; elapsed cost ~${self.elapsed_cost():.3f}")
            finally:
                # belt and braces: confirm nothing is left running
                left = account().pods
                if left:
                    print(f"[runpod] WARNING pods still present: {[p['id'] for p in left]}; terminating all")
                    terminate_all()
        return False


if __name__ == "__main__":
    import sys
    a = account()
    print(f"balance ${a.balance:.2f} | spend limit ${a.spend_limit}/hr | current ${a.spend_per_hr}/hr | pods: {len(a.pods)}")
    for p in a.pods:
        print("  ", p)
    if len(sys.argv) > 1 and sys.argv[1] == "--kill-all":
        print("terminated:", terminate_all())
