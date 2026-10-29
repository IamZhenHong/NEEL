"""CC-F-A1 deviation 12 (owner, 10-29 ~09:40: RunPod credits, not a Modal top-up): the slice of the `modal` API that
E5's pinned ODCV environment and A1's odcv.py / eval_a1.py call, run on the serving pod with proot instead of Modal
Sandboxes. E5's files stay byte-identical: `install()` puts this module in sys.modules["modal"] before they import it.

A RunPod container runs in a user namespace (uid 0 -> 100000) with Docker's default seccomp: chroot works, but mknod
and every unshare are refused, so the rootfs can't hold a /dev/null and no namespace can isolate it. proot (ptrace,
no privileges; static binary from proot.gitlab.io) binds the pod's /dev/null, zero, full, random and urandom into it.

  Image.from_dockerfile   the Dockerfile E5's context() writes (FROM / WORKDIR / RUN / COPY / CMD, with its
                          archive.debian.org and faketime lines) built into a rootfs directory: the FROM image's
                          linux/amd64 layers from Docker Hub, then each RUN through proot with the pod's network.
                          The steps before the first COPY are cached across scenarios; the image by content hash.
  Sandbox.create          a fresh copy of that rootfs; DNS pointed at 127.0.0.1.
  Sandbox.exec            proot -r rootfs; the image's ENV plus the call's env; stdin /dev/null; its own session;
                          one CPU (taskset); RLIMIT_AS 2 GiB, standing in for Modal's 1 GiB (virtual, not resident).
                          At the timeout the session is killed and returncode is -1, as Modal returns it.
                          Not reproduced: block_network (only DNS is off: raw IPs and the pod's 127.0.0.1 ports stay
                          reachable) and a private /proc (none is mounted, so ps / pkill / /dev/fd fail instead).
  snapshot_filesystem     a copy of the sandbox's rootfs as an Image; Image.from_id reads one back.
  terminate               kill every session the sandbox's execs started, delete it.

  python -m character_conduct.final_bets.a1_train.chroot_modal probe              # what this container allows
  python -m character_conduct.final_bets.a1_train.chroot_modal check              # pristine manifests + replay
  python -m character_conduct.final_bets.a1_train.chroot_modal run MODULE ARG...  # MODULE with this as `modal`
"""
import hashlib
import itertools
import json
import os
import runpy
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path

BASE = Path(os.environ.get("CHROOT_MODAL_DIR", "/root/odcv_chroot"))
IMAGES, SANDBOXES, BLOBS = BASE / "images", BASE / "sb", BASE / "blobs"
AS_BYTES = 2 * 1024 ** 3
PINS = Path(__file__).resolve().parent / "chroot_pins.json"
OUT = Path(__file__).resolve().parent / "results" / "eval" / "chroot_check.json"
IDX = "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json"
MAN = "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json"
DEVS = ("null", "zero", "full", "random", "urandom")
PROOT = BASE / "bin" / "proot"
PROOT_URL = "https://proot.gitlab.io/proot/bin/proot"

_lock = threading.Lock()
_build_locks = {}
_live = {}
_cpus = itertools.cycle(sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [0])
_mode = {}


def _ok(args):
    try:
        return subprocess.run(args, capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _proot():
    if not PROOT.exists():
        PROOT.parent.mkdir(parents=True, exist_ok=True)
        _curl(PROOT_URL, out=PROOT.with_suffix(".part"))
        os.chmod(PROOT.with_suffix(".part"), 0o755)
        PROOT.with_suffix(".part").rename(PROOT)
    return hashlib.sha256(PROOT.read_bytes()).hexdigest()


def probe():
    """What this container allows: proot, chroot, mknod, unshare (net + pid + proc), prlimit, taskset."""
    out = {"proot_sha256": _proot(), "proot": _ok([str(PROOT), "-r", "/", "-b", "/dev/null", "true"]),
           "chroot": _ok(["chroot", "/", "true"]),
           "unshare": _ok(["unshare", "--net", "--pid", "--fork", "--mount-proc", "true"]),
           "prlimit": _ok(["prlimit", f"--as={AS_BYTES}", "true"]), "taskset": _ok(["taskset", "-c", "0", "true"])}
    with tempfile.TemporaryDirectory() as d:
        try:
            os.mknod(f"{d}/null", 0o666 | stat.S_IFCHR, os.makedev(1, 3))
            with open(f"{d}/null", "w") as fh:
                fh.write("x")
            out["mknod"] = open(f"{d}/null").read() == ""
        except OSError as e:
            out["mknod"] = f"{type(e).__name__}: {e}"
    return out


def mode():
    with _lock:
        if not _mode:
            _mode.update(probe())
            assert _mode["proot"] is True, f"proot unusable here: {_mode}"
    return _mode


def _curl(url, out=None, token=None, accept=None):
    a = ["curl", "-fsSL", "--retry", "5", "--max-time", "900"]
    a += ["-H", f"Authorization: Bearer {token}"] if token else []
    a += ["-H", f"Accept: {accept}"] if accept else []
    a += ["-o", str(out)] if out else []
    r = subprocess.run(a + [url], check=True, capture_output=out is None)
    return None if out else r.stdout


def _pull(ref, root):
    """The linux/amd64 rootfs of Docker Hub image `ref`, unpacked into root (whiteouts applied); (digest, config)."""
    repo, tag = ref.split(":", 1)
    name = repo if "/" in repo else f"library/{repo}"
    tok = json.loads(_curl(f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{name}:pull"))
    tok = tok["token"]
    reg = f"https://registry-1.docker.io/v2/{name}"
    idx = json.loads(_curl(f"{reg}/manifests/{tag}", token=tok, accept=f"{IDX}, {MAN}"))
    digest = tag
    if "manifests" in idx:
        digest = next(m["digest"] for m in idx["manifests"] if m.get("platform", {}).get("architecture") == "amd64"
                      and m["platform"].get("os") == "linux")
        idx = json.loads(_curl(f"{reg}/manifests/{digest}", token=tok, accept=MAN))
    cfg = json.loads(_curl(f"{reg}/blobs/{idx['config']['digest']}", token=tok))
    root.mkdir(parents=True)
    BLOBS.mkdir(parents=True, exist_ok=True)
    for layer in idx["layers"]:
        blob = BLOBS / layer["digest"].replace(":", "_")
        if not blob.exists():
            part = blob.with_suffix(".part")
            _curl(f"{reg}/blobs/{layer['digest']}", out=part, token=tok)
            part.rename(blob)
        names = subprocess.run(["tar", "-tzf", str(blob)], check=True, capture_output=True, text=True).stdout
        for n in names.splitlines():
            d, b = os.path.split(os.path.normpath(n))
            if b == ".wh..wh..opq":
                for x in (root / d).glob("*") if (root / d).is_dir() else []:
                    subprocess.run(["rm", "-rf", str(x)], check=True)
            elif b.startswith(".wh."):
                subprocess.run(["rm", "-rf", str(root / d / b[4:])], check=True)
        subprocess.run(["tar", "-xzf", str(blob), "-C", str(root), "--numeric-owner", "--same-owner",
                        "--same-permissions", "--exclude=.wh.*"], check=True)
    return digest, cfg.get("config") or {}


def _inside(root, path):
    """`path` (absolute, in the image) as a host path under root, following symlinks inside the image."""
    cur = Path(root)
    for part in [p for p in path.split("/") if p not in ("", ".")]:
        nxt = cur / part
        for _ in range(40):
            if not nxt.is_symlink():
                break
            t = os.readlink(nxt)
            nxt = Path(root) / t.lstrip("/") if t.startswith("/") else nxt.parent / t
        cur = Path(os.path.normpath(nxt))
        if not str(cur).startswith(str(root)):
            cur = Path(root)
    return cur


def _devs(root):
    """/dev and /dev/shm as directories; the device files themselves are proot binds (see _argv)."""
    d = _inside(root, "/dev")
    d.mkdir(parents=True, exist_ok=True)
    (d / "shm").mkdir(exist_ok=True)


def _resolv(root, text):
    p = _inside(root, "/etc") / "resolv.conf"
    if p.is_symlink() or p.exists():
        p.unlink()
    p.write_text(text)


def _parse(text):
    """[(INSTRUCTION, rest)], continuation lines joined, comments and blanks dropped."""
    out, cur = [], ""
    for line in text.splitlines():
        s = line.strip()
        if not cur and (not s or s.startswith("#")):
            continue
        if s.endswith("\\"):
            cur += s[:-1] + " "
            continue
        cur += s
        ins, _, rest = cur.partition(" ")
        out.append((ins.upper(), rest.strip()))
        cur = ""
    return out


def _argv(root, meta, argv, workdir=None, env=None, sandbox=False):
    e = {"HOME": "/root", **meta["env"], **(env or {})}
    m = mode()
    binds = [x for d in DEVS for x in ("-b", f"/dev/{d}")]
    # -w /: else proot tries the pod's own cwd inside the guest and says so on stderr; the inner cd sets the workdir
    a = [str(PROOT), "-r", str(root), "-w", "/", *binds, "/usr/bin/env", "-i", *[f"{k}={v}" for k, v in e.items()],
         "/bin/sh", "-c", 'cd "$0" && exec "$@"', workdir or meta["workdir"], *argv]
    if sandbox:
        if m["prlimit"]:
            a = ["prlimit", f"--as={AS_BYTES}", "--", *a]
        if m["taskset"]:
            a = ["taskset", "-c", str(sandbox), *a]
    return a


def _step(root, meta, ins, rest, ctx):
    if ins == "FROM":
        meta["from"] = [rest, None]
        meta["from"][1], cfg = _pull(rest, root)
        meta["env"] = dict(x.split("=", 1) for x in cfg.get("Env") or [])
        meta["workdir"] = cfg.get("WorkingDir") or "/"
        _devs(root)
        _resolv(root, Path("/etc/resolv.conf").read_text())
        # apt's methods drop to the _apt user, which hangs under proot in this user namespace (10-29 probe: update
        # stalls > 10 min, done in 2 s with this); kept in the image, outside every scenario's manifest scope
        conf = _inside(root, "/etc/apt/apt.conf.d")
        if conf.is_dir():
            (conf / "00-chroot-modal").write_text('APT::Sandbox::User "root";\n')
    elif ins == "WORKDIR":
        meta["workdir"] = os.path.normpath(os.path.join(meta["workdir"], rest))
        _inside(root, meta["workdir"]).mkdir(parents=True, exist_ok=True)
    elif ins == "ENV":
        k, _, v = rest.partition("=") if "=" in rest.split()[0] else rest.partition(" ")
        meta["env"][k.strip()] = v.strip().strip('"')
    elif ins == "COPY":
        src, dst = rest.split()
        d = dst if dst.startswith("/") else os.path.join(meta["workdir"], dst)
        wild = any(c in src for c in "*?[")
        srcs = sorted(Path(ctx).glob(src.lstrip("./") if src.startswith("./") else src)) if wild else [Path(ctx) / src]
        if not srcs or not all(s.exists() for s in srcs):
            raise FileNotFoundError(f"COPY {src}: nothing in the context")
        for s in srcs:
            t = _inside(root, d)
            if s.is_dir():          # Docker copies a directory's contents, for a named or a matched one alike
                t.mkdir(parents=True, exist_ok=True)
                shutil.copytree(s, t, symlinks=True, dirs_exist_ok=True)
            else:
                if dst.endswith("/") or t.is_dir() or wild:
                    t.mkdir(parents=True, exist_ok=True)
                    t = t / s.name
                t.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(s, t)
    elif ins == "RUN":
        r = subprocess.run(_argv(root, meta, ["/bin/sh", "-c", rest]), capture_output=True, text=True,
                           stdin=subprocess.DEVNULL)
        if r.returncode:
            raise RuntimeError(f"RUN failed ({r.returncode}): {rest[:200]}\n{r.stderr[-3000:]}")
    elif ins not in ("CMD", "ENTRYPOINT", "EXPOSE", "LABEL"):
        raise ValueError(f"unsupported Dockerfile instruction {ins}")


def _hash(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
        h.update(b"\0")
    return h.hexdigest()[:24]


def _locked(key):
    with _lock:
        return _build_locks.setdefault(key, threading.Lock())


def _build(dockerfile, ctx):
    steps = _parse(Path(dockerfile).read_text())
    files = sorted(p for p in Path(ctx).rglob("*") if p.is_file())
    key = _hash(Path(dockerfile).read_bytes(), *[f"{p.relative_to(ctx)}:{_hash(p.read_bytes())}" for p in files])
    dest = IMAGES / key
    with _locked(key):
        if (dest / "image.json").exists():
            return Image(dest)
        k = next((i for i, (ins, _) in enumerate(steps) if ins == "COPY"), len(steps))
        pre = IMAGES / ("prefix-" + _hash(*[f"{i} {r}" for i, r in steps[:k]]))
        with _locked(pre.name):
            if not (pre / "image.json").exists():
                tmp = Path(f"{pre}.tmp-{uuid.uuid4().hex[:6]}")
                meta = {"env": {}, "workdir": "/"}
                for ins, rest in steps[:k]:
                    _step(tmp / "rootfs", meta, ins, rest, ctx)
                (tmp / "image.json").write_text(json.dumps(meta))
                tmp.rename(pre)
        tmp = Path(f"{dest}.tmp-{uuid.uuid4().hex[:6]}")
        tmp.mkdir(parents=True)
        subprocess.run(["cp", "-a", str(pre / "rootfs"), str(tmp / "rootfs")], check=True)
        meta = json.loads((pre / "image.json").read_text())
        for ins, rest in steps[k:]:
            _step(tmp / "rootfs", meta, ins, rest, ctx)
        meta["key"] = key
        (tmp / "image.json").write_text(json.dumps(meta))
        tmp.rename(dest)
        return Image(dest)


class Image:
    def __init__(self, path):
        self.path = Path(path)
        self.root = self.path / "rootfs"
        self.meta = json.loads((self.path / "image.json").read_text())
        self.object_id = str(self.path)

    @staticmethod
    def from_dockerfile(path, context_dir=None, **_):
        return _build(path, context_dir or Path(path).parent)

    @staticmethod
    def from_id(object_id, **_):
        return Image(object_id)


class App:
    def __init__(self, name):
        self.name = self.app_id = name

    @staticmethod
    def lookup(name, create_if_missing=False, **_):
        return App(name)


class _Stream:
    def __init__(self, text):
        self.text = text

    def read(self):
        return self.text


class _Proc:
    def __init__(self, out, err, rc):
        self.stdout, self.stderr, self.returncode = _Stream(out), _Stream(err), rc

    def wait(self):
        return self.returncode


class Sandbox:
    def __init__(self, image):
        self.object_id = "sb-" + uuid.uuid4().hex[:12]
        self.image, self.root, self.cpu = image, SANDBOXES / self.object_id, next(_cpus)
        self.sids = set()
        SANDBOXES.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", "-a", str(image.root), str(self.root)], check=True)
        _resolv(self.root, "nameserver 127.0.0.1\n")
        with _lock:
            _live[self.object_id] = self

    @staticmethod
    def create(*_entrypoint, app=None, image=None, **_):
        return Sandbox(image)

    @staticmethod
    def list(app_id=None, **_):
        with _lock:
            return list(_live.values())

    def exec(self, *argv, workdir=None, timeout=None, env=None, **_):
        p = subprocess.Popen(_argv(self.root, self.image.meta, list(argv), workdir, env, sandbox=str(self.cpu)),
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             start_new_session=True)
        self.sids.add(p.pid)
        try:
            out, err = p.communicate(timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._kill()
            out, err = p.communicate()
            rc = -1
        return _Proc(out.decode("utf-8", "replace"), err.decode("utf-8", "replace"), rc)

    def snapshot_filesystem(self, **_):
        dest = IMAGES / f"snap-{uuid.uuid4().hex[:12]}"
        dest.mkdir(parents=True)
        subprocess.run(["cp", "-a", str(self.root), str(dest / "rootfs")], check=True)
        (dest / "image.json").write_text(json.dumps(self.image.meta))
        return Image(dest)

    def _kill(self):
        """SIGKILL every process in a session one of this sandbox's execs started (proot's tracees stay in it)."""
        for d in Path("/proc").glob("[0-9]*"):
            try:
                sid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[3])
                if sid in self.sids:
                    os.kill(int(d.name), signal.SIGKILL)
            except (OSError, ValueError, IndexError):
                pass

    def terminate(self, **_):
        self._kill()
        subprocess.run(["rm", "-rf", str(self.root)], check=False)
        with _lock:
            _live.pop(self.object_id, None)


def install():
    """This module as `modal`; on a pod without .git, odcv.pin_check reads chroot_pins.json (made from git at PIN)."""
    sys.modules["modal"] = sys.modules[__name__]
    from character_conduct.final_bets.a1_train import odcv as O
    root = Path(O.ROOT)
    if not (root / ".git").exists():
        pins = json.loads(PINS.read_text())

        def pin_check():
            assert pins["pin"] == O.PIN and set(pins["sha256"]) == set(O.PINNED), "chroot_pins.json is stale"
            bad = [p for p, h in pins["sha256"].items() if hashlib.sha256((root / p).read_bytes()).hexdigest() != h]
            if bad:
                raise SystemExit(f"[pin] not byte-identical to {O.PIN}: {bad}")
        O.pin_check = pin_check
    key = os.environ.get("A1_VLLM_KEY")
    if key:
        init = O.Gen.__init__

        def gen_init(self, k, *a, **kw):
            init(self, k, *a, **kw)
            if self.key is None and "127.0.0.1" in self.url:
                self.key = key
        O.Gen.__init__ = gen_init
    return O


def pins():
    """chroot_pins.json from git (run on the laptop)."""
    from character_conduct.final_bets.a1_train import odcv as O
    sha = {p: hashlib.sha256(subprocess.run(["git", "show", f"{O.PIN}:{p}"], cwd=O.ROOT, capture_output=True,
                                            check=True).stdout).hexdigest() for p in O.PINNED}
    PINS.write_text(json.dumps({"pin": O.PIN, "sha256": sha}, indent=1) + "\n")
    print(PINS.read_text())


def check():
    """Before any model episode (DESIGN §12 deviation 12): (1) every 1a headroom SV's pristine manifest against the
    STARTS Modal recorded; (2) the released Qwen3.6-27B run of each replayed call by call (E5's modal_env.one, fixes
    on) against Modal's own replay (all80_fixes.json)."""
    O = install()
    O.pin_check()
    from character_conduct.final_bets.a1_train import eval_a1 as A
    T, C, EV, EP, ME = O.e5()
    app = App.lookup(O.APP)
    st = O.starts()
    res = {"mode": mode(), "pristine": {}, "replay": []}
    for v, s in A.HEADROOM:
        sb = EV.sandbox(app, v, s)
        try:
            man = EV.manifest(sb, EV.scope(v, s))
        finally:
            sb.terminate()
        want = st[O.key(v, s)]["pristine"]
        diff = sorted(p for p in set(man) | set(want) if man.get(p) != want.get(p))
        res["pristine"][f"{v}|{s}"] = {"n": len(want), "same": not diff, "diff": diff[:20]}
        print(f"[check] pristine {v}|{s}: {len(want)} files, {'same' if not diff else f'{len(diff)} differ'}",
              flush=True)
    modal = {(r["variant"], r["scenario"]): r for r in json.loads((ME.OUT / "all80_fixes.json").read_text())}
    for v, s in A.HEADROOM:
        r = ME.one(app, v, s, True)
        m = modal.get((v, s), {})
        mine = {c["j"]: c["same"] for c in r.get("calls", [])}
        theirs = {c["j"]: c["same"] for c in m.get("calls", [])}
        split = sorted(j for j in mine if mine[j] != theirs.get(j))
        res["replay"].append({"sv": f"{v}|{s}", "build": r.get("build"), "n_calls": r["n_calls"],
                              "match": r.get("match"), "modal_match": m.get("match"),
                              "restore": r.get("restore_manifest_same"), "tail": r.get("restore_tail_match"),
                              "modal_tail": m.get("restore_tail_match"), "error": r.get("error"),
                              "split_calls": split,
                              "split_detail": [x for x in r.get("mismatch", []) if x["j"] in split][:10]})
        print(f"[check] replay {v}|{s}: {r.get('match')}/{r['n_calls']} (Modal {m.get('match')}), restore "
              f"{r.get('restore_manifest_same')} {r.get('restore_tail_match')}, split calls {split}", flush=True)
    for snap in IMAGES.glob("snap-*"):
        subprocess.run(["rm", "-rf", str(snap)], check=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    ok = all(x["same"] for x in res["pristine"].values()) and all(not x["split_calls"] for x in res["replay"])
    print(f"[check] mode {res['mode']}; pristine all same: {all(x['same'] for x in res['pristine'].values())}; "
          f"replay splits: {sum(len(x['split_calls']) for x in res['replay'])}; PASS {ok}", flush=True)
    return res


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    cmd = a.pop(0) if a else "probe"
    if cmd == "probe":
        print(json.dumps(probe(), indent=1))
    elif cmd == "pins":
        pins()
    elif cmd == "check":
        check()
    elif cmd == "run":
        mod = a.pop(0)
        install()
        sys.argv = [mod, *a]
        runpy.run_module(mod, run_name="__main__", alter_sys=True)
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
