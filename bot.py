"""
cyber-agent — a small Discord bot bridging DMs to the skynet LLM stack
(via LiteLLM) with docker-management, infra-status, and UniFi read tools.

Deliberately simple: one file, no workflow engine, no separate database
server. Connects to Discord over its outbound Gateway WebSocket (discord.py
handles this) so there is no webhook, no reverse proxy, and no public
hostname needed.
"""

import json
import logging
import os
import pathlib
import re
import shlex
import shutil
import socket
import sqlite3
import ssl
import subprocess
import time

import discord
import docker as docker_sdk
import psycopg2
from discord.ext import tasks
import requests
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cyber-agent")

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
OWNER_ID = int(os.environ["DISCORD_OWNER_ID"])
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://litellm:4000/v1")
LITELLM_API_KEY = os.environ["LITELLM_API_KEY"]
MODEL_FALLBACK_ORDER = os.environ.get(
    "MODEL_FALLBACK_ORDER", "qwen3-coder-next,qwen3-vl-8b,qwen3-8b-ablated"
).split(",")
DB_PATH = os.environ.get("DB_PATH", "/data/history.db")
MAX_HISTORY_MESSAGES = 20

UNIFI_HOST = os.environ.get("UNIFI_HOST", "")
UNIFI_USERNAME = os.environ.get("UNIFI_USERNAME", "")
UNIFI_PASSWORD = os.environ.get("UNIFI_PASSWORD", "")
UNIFI_SITE = os.environ.get("UNIFI_SITE", "default")

BACKUP_LOG_PATH = os.environ.get("BACKUP_LOG_PATH", "/backup-log/backup-to-shield.log")
CERT_HOSTS = os.environ.get(
    "CERT_HOSTS",
    "auth.skynet.drewdettmer.com,chat.skynet.drewdettmer.com,ouachitacyber.com",
).split(",")

LITELLM_DB_PASSWORD = os.environ.get("LITELLM_DB_PASSWORD", "")
FS_ROOT = os.environ.get("FS_ROOT", "/home/skynet")

SYSTEM_PROMPT = (
    "You are cyber-agent, an infra assistant for the Ouachita Cyber / skynet LLM lab. "
    "You run inside a Discord DM. Keep replies short and plain text (no markdown tables). "
    "Read-only tools: docker_status, docker_logs, container_stats, gpu_status, backup_status, "
    "cert_expiry, disk_usage, uptime_report, token_usage, unifi_clients, unifi_network_health, "
    "unifi_topology, unifi_alerts, fs_read, fs_glob, fs_grep, web_search. "
    f"fs_read/fs_glob/fs_grep operate on the real skynet filesystem under {FS_ROOT} (bind-mounted "
    "into this container at the same path) -- that's every stack's config/code/logs. "
    "Mutating tools (propose_restart, propose_swap_gpu_config, propose_fs_write, propose_run_bash) "
    "never execute immediately -- calling one shows the user a Discord Confirm/Cancel button "
    "prompt, and only clicking Confirm executes it. Do not claim an action succeeded until you "
    "have tool output showing it did, and never tell the user to type 'yes' -- the buttons "
    "handle that. propose_run_bash is real host-visible bash (sees the mounted filesystem and "
    "the docker socket, not a sandbox) -- use it for anything the other tools don't cover, but "
    "always propose, never assume it ran. "
    "GPU note: only one of {qwen3-coder-next} or {qwen3-vl-8b + qwen3-8b-ablated} runs at a "
    "time on this box -- if a model call fails, that GPU config may be inactive right now; "
    "use gpu_status to check, and propose_swap_gpu_config (with confirmation) to change it."
)

ai_client = OpenAI(base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)
docker_client = docker_sdk.from_env()

# ---------------------------------------------------------------- storage --

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS messages ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "channel_id TEXT NOT NULL, "
        "role TEXT NOT NULL, "
        "content TEXT NOT NULL, "
        "ts REAL NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS issue_state ("
        "issue_key TEXT PRIMARY KEY, "
        "active INTEGER NOT NULL, "
        "description TEXT NOT NULL, "
        "first_seen REAL NOT NULL, "
        "last_seen REAL NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS service_checks ("
        "service TEXT NOT NULL, "
        "ts REAL NOT NULL, "
        "up INTEGER NOT NULL)"
    )
    return conn


def load_history(channel_id: str) -> list[dict]:
    conn = db()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE channel_id = ? ORDER BY id DESC LIMIT ?",
        (channel_id, MAX_HISTORY_MESSAGES),
    ).fetchall()
    conn.close()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def save_message(channel_id: str, role: str, content: str) -> None:
    conn = db()
    conn.execute(
        "INSERT INTO messages (channel_id, role, content, ts) VALUES (?, ?, ?, ?)",
        (channel_id, role, content, time.time()),
    )
    conn.commit()
    conn.close()


# --------------------------------------------------------------- unifi api --

class UniFiSession:
    """Minimal UniFi OS console client: login once, reuse the session cookie,
    re-login once on a 401. Read-only usage only -- no write endpoints are
    ever called from this bot regardless of what the underlying account can
    do."""

    def __init__(self, host: str, username: str, password: str, site: str):
        self.host = host
        self.username = username
        self.password = password
        self.site = site
        self.session = requests.Session()
        self.session.verify = False
        self._logged_in = False

    def _login(self) -> None:
        resp = self.session.post(
            f"https://{self.host}/api/auth/login",
            json={"username": self.username, "password": self.password},
            timeout=10,
        )
        resp.raise_for_status()
        csrf = resp.headers.get("X-CSRF-Token") or resp.headers.get("x-csrf-token")
        if csrf:
            self.session.headers["X-CSRF-Token"] = csrf
        self._logged_in = True

    def get(self, path: str) -> dict:
        if not self._logged_in:
            self._login()
        url = f"https://{self.host}/proxy/network/api/s/{self.site}/{path.lstrip('/')}"
        resp = self.session.get(url, timeout=10)
        if resp.status_code == 401:
            self._login()
            resp = self.session.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()


unifi = (
    UniFiSession(UNIFI_HOST, UNIFI_USERNAME, UNIFI_PASSWORD, UNIFI_SITE)
    if UNIFI_HOST and UNIFI_USERNAME and UNIFI_PASSWORD
    else None
)

# suppress the expected self-signed-cert warning from the UniFi console
try:
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass

# -------------------------------------------------------------------- tools --

def tool_docker_status(_args: dict) -> str:
    containers = docker_client.containers.list(all=True)
    lines = [f"{c.name}\t{c.status}\t{c.attrs['Config']['Image']}" for c in containers]
    running = sum(1 for c in containers if c.status == "running")
    header = f"total={len(containers)} running={running} stopped={len(containers) - running}"
    return header + "\n" + ("\n".join(lines) if lines else "no containers found")


def tool_docker_logs(args: dict) -> str:
    name = args.get("container", "")
    lines = int(args.get("lines", 50))
    try:
        c = docker_client.containers.get(name)
    except docker_sdk.errors.NotFound:
        return f"no such container: {name}"
    return c.logs(tail=lines).decode("utf-8", errors="replace")


def tool_container_stats(_args: dict) -> str:
    lines = []
    for c in docker_client.containers.list():
        try:
            s = c.stats(stream=False)
            cpu_delta = (
                s["cpu_stats"]["cpu_usage"]["total_usage"]
                - s["precpu_stats"]["cpu_usage"]["total_usage"]
            )
            sys_delta = s["cpu_stats"]["system_cpu_usage"] - s["precpu_stats"]["system_cpu_usage"]
            n_cpus = s["cpu_stats"].get("online_cpus") or len(
                s["cpu_stats"]["cpu_usage"].get("percpu_usage") or [1]
            )
            cpu_pct = (cpu_delta / sys_delta) * n_cpus * 100 if sys_delta > 0 else 0.0
            mem_used = s["memory_stats"].get("usage", 0)
            mem_limit = s["memory_stats"].get("limit", 1) or 1
            lines.append(
                f"{c.name}\tcpu={cpu_pct:.1f}%\tmem={mem_used / 1e6:.0f}MB "
                f"({100 * mem_used / mem_limit:.0f}%)"
            )
        except Exception as e:  # noqa: BLE001
            lines.append(f"{c.name}\terror: {e}")
    return "\n".join(lines) if lines else "no running containers"


def tool_propose_restart(args: dict) -> str:
    name = args.get("container", "")
    try:
        docker_client.containers.get(name)
    except docker_sdk.errors.NotFound:
        return f"no such container: {name}"
    PENDING_CONFIRMATIONS[args["_channel_id"]] = {"action": "restart", "container": name}
    return (
        f"PENDING_CONFIRMATION: proposed restarting '{name}'. A Confirm/Cancel button "
        f"prompt will be shown to the user separately -- do not ask them to type yes, "
        f"just briefly note what you're proposing and that it needs their confirmation."
    )


GPU_VISION_CONTAINERS = ["vllm-qwen3-vl-8b", "vllm-qwen3-8b-ablated"]
GPU_CODER_CONTAINER = "vllm-qwen3-coder-next"


def tool_gpu_status(_args: dict) -> str:
    names = GPU_VISION_CONTAINERS + [GPU_CODER_CONTAINER]
    statuses = {}
    for name in names:
        try:
            statuses[name] = docker_client.containers.get(name).status
        except docker_sdk.errors.NotFound:
            statuses[name] = "missing"
    vision_up = all(statuses[n] == "running" for n in GPU_VISION_CONTAINERS)
    coder_up = statuses[GPU_CODER_CONTAINER] == "running"
    if vision_up:
        mode = "vision+ablated (qwen3-vl-8b + qwen3-8b-ablated) -- Coder-Next is down"
    elif coder_up:
        mode = "coder-next (qwen3-coder-next) -- vision/ablated are down"
    else:
        mode = "neither config fully up -- check statuses below"
    lines = [f"{n}: {s}" for n, s in statuses.items()]
    return f"current mode: {mode}\n" + "\n".join(lines)


def tool_propose_swap_gpu_config(args: dict) -> str:
    target = args.get("target", "").strip().lower()
    if target not in ("vision", "coder", "coder-next"):
        return "target must be 'vision' or 'coder'"
    target = "coder" if target.startswith("coder") else "vision"
    PENDING_CONFIRMATIONS[args["_channel_id"]] = {"action": "swap_gpu_config", "target": target}
    return (
        f"PENDING_CONFIRMATION: proposed swapping the GPU config to '{target}' "
        f"(stops the other config's containers first, ~1-2 min to load). A Confirm/Cancel "
        f"button prompt will be shown to the user separately -- do not ask them to type "
        f"yes, just briefly note what you're proposing and that it needs their confirmation."
    )



def tool_fs_read(args: dict) -> str:
    path = args.get("path", "")
    try:
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:  # noqa: BLE001
        return f"could not read {path}: {e}"
    offset = int(args.get("offset") or 1)
    limit = int(args.get("limit") or 300)
    chunk = lines[offset - 1 : offset - 1 + limit]
    body = "".join(f"{offset + i}:{line}" for i, line in enumerate(chunk))
    return body[:6000] if body else "(empty range)"


def tool_fs_glob(args: dict) -> str:
    pattern = args.get("pattern", "*")
    root = args.get("root") or FS_ROOT
    try:
        matches = sorted(str(p) for p in pathlib.Path(root).glob(pattern))
    except Exception as e:  # noqa: BLE001
        return f"glob failed: {e}"
    if not matches:
        return "no matches"
    return f"{len(matches)} match(es):\n" + "\n".join(matches[:200])


def tool_fs_grep(args: dict) -> str:
    pattern = args.get("pattern", "")
    path = args.get("path") or FS_ROOT
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"bad regex: {e}"
    hits = []
    targets = [path] if os.path.isfile(path) else [
        str(p) for p in pathlib.Path(path).rglob("*") if p.is_file()
    ]
    for fpath in targets:
        try:
            with open(fpath, "r", errors="replace") as f:
                for i, line in enumerate(f, start=1):
                    if rx.search(line):
                        hits.append(f"{fpath}:{i}:{line.rstrip()}")
                        if len(hits) >= 200:
                            break
        except (OSError, UnicodeDecodeError):
            continue
        if len(hits) >= 200:
            break
    return "\n".join(hits) if hits else "no matches"


def tool_web_search(args: dict) -> str:
    query = args.get("query", "")
    try:
        r = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            timeout=10,
        )
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        return f"web search failed: {e}"
    titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', r.text, re.S)
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', r.text, re.S)
    urls = re.findall(r'class="result__url"[^>]*>\s*(.*?)\s*</a>', r.text, re.S)

    def clean(s: str) -> str:
        return re.sub(r"<[^>]+>", "", s).strip()

    results = []
    for i in range(min(5, len(titles))):
        title = clean(titles[i])
        url = clean(urls[i]) if i < len(urls) else ""
        snippet = clean(snippets[i]) if i < len(snippets) else ""
        results.append(f"{title} ({url})\n{snippet}")
    return "\n\n".join(results) if results else "no results"


def tool_propose_fs_write(args: dict) -> str:
    path = args.get("path", "")
    content = args.get("content", "")
    PENDING_CONFIRMATIONS[args["_channel_id"]] = {
        "action": "fs_write",
        "path": path,
        "content": content,
    }
    return (
        f"PENDING_CONFIRMATION: proposed writing {len(content)} byte(s) to `{path}` "
        f"(will overwrite if it exists). A Confirm/Cancel button prompt will be shown to "
        f"the user separately -- do not ask them to type yes, just briefly note what "
        f"you're proposing and that it needs their confirmation."
    )


def tool_propose_run_bash(args: dict) -> str:
    cmd = args.get("command", "").strip()
    if not cmd:
        return "no command given"
    PENDING_CONFIRMATIONS[args["_channel_id"]] = {"action": "run_bash", "command": cmd}
    return (
        f"PENDING_CONFIRMATION: proposed running `{cmd}` (bash, cwd={FS_ROOT}, host-visible "
        f"via the mounted filesystem and docker socket). A Confirm/Cancel button prompt will "
        f"be shown to the user separately -- do not ask them to type yes, just briefly note "
        f"what you're proposing and that it needs their confirmation."
    )


def tool_backup_status(_args: dict) -> str:
    if not os.path.exists(BACKUP_LOG_PATH):
        return f"backup log not found at {BACKUP_LOG_PATH} (mount missing?)"
    age_h = (time.time() - os.path.getmtime(BACKUP_LOG_PATH)) / 3600
    with open(BACKUP_LOG_PATH, "r", errors="replace") as f:
        tail = f.readlines()[-8:]
    ok = any("Backup complete" in line for line in tail)
    return f"log last updated {age_h:.1f}h ago; last run {'completed successfully' if ok else 'did NOT show a clean completion'}\n" + "".join(tail)


def tool_cert_expiry(args: dict) -> str:
    hosts = [args["host"]] if args.get("host") else CERT_HOSTS
    results = []
    for host in hosts:
        host = host.strip()
        try:
            ctx = ssl.create_default_context()
            with socket.create_connection((host, 443), timeout=5) as sock:
                with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                    cert = ssock.getpeercert()
            import datetime

            not_after = datetime.datetime.strptime(cert["notAfter"], "%b %d %H:%M:%S %Y %Z")
            not_after = not_after.replace(tzinfo=datetime.timezone.utc)
            days_left = (not_after - datetime.datetime.now(datetime.timezone.utc)).days
            results.append(f"{host}: expires in {days_left} days ({not_after.date()})")
        except Exception as e:  # noqa: BLE001
            results.append(f"{host}: could not check ({e})")
    return "\n".join(results)


def tool_disk_usage(_args: dict) -> str:
    total, used, free = shutil.disk_usage("/host-root") if os.path.exists("/host-root") else shutil.disk_usage("/")
    gb = 1024**3
    return (
        f"root volume: {used / gb:.1f}G used / {total / gb:.1f}G total "
        f"({free / gb:.1f}G free, {100 * used / total:.0f}% full)"
    )


def tool_token_usage(args: dict) -> str:
    if not LITELLM_DB_PASSWORD:
        return "LITELLM_DB_PASSWORD not configured on this bot"
    days = int(args.get("days") or 7)
    try:
        conn = psycopg2.connect(
            host="litellm-db",
            dbname="litellm",
            user="litellm",
            password=LITELLM_DB_PASSWORD,
            connect_timeout=5,
        )
        with conn, conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(v.key_alias, '(unlabeled -- master key or deleted key)') AS alias, "
                'SUM(s.total_tokens), SUM(s.prompt_tokens), SUM(s.completion_tokens), COUNT(*) '
                'FROM "LiteLLM_SpendLogs" s '
                'LEFT JOIN "LiteLLM_VerificationToken" v ON s.api_key = v.token '
                'WHERE s."startTime" > now() - make_interval(days => %s) '
                "GROUP BY alias ORDER BY 2 DESC NULLS LAST",
                (days,),
            )
            rows = cur.fetchall()
        conn.close()
    except Exception as e:  # noqa: BLE001
        return f"token usage query failed: {e}"
    if not rows:
        return f"no usage recorded in the last {days} day(s)"
    lines = [f"tokens used, last {days} day(s):"]
    for alias, total, prompt, completion, calls in rows:
        lines.append(
            f"{alias}: {total or 0:,} tokens ({prompt or 0:,} prompt + {completion or 0:,} "
            f"completion), {calls} calls"
        )
    return "\n".join(lines)


def tool_unifi_clients(_args: dict) -> str:
    if unifi is None:
        return "UniFi is not configured on this bot"
    try:
        data = unifi.get("stat/sta")
    except Exception as e:  # noqa: BLE001
        return f"UniFi request failed: {e}"
    clients = data.get("data", [])
    lines = [
        f"{c.get('hostname') or c.get('name') or c.get('mac')}\t{c.get('ip', '?')}\t"
        f"{'wired' if c.get('is_wired') else c.get('essid', 'wifi')}"
        for c in clients
    ]
    return f"total={len(clients)}\n" + "\n".join(lines[:60])


def tool_unifi_network_health(_args: dict) -> str:
    if unifi is None:
        return "UniFi is not configured on this bot"
    try:
        data = unifi.get("stat/health")
    except Exception as e:  # noqa: BLE001
        return f"UniFi request failed: {e}"
    lines = []
    for entry in data.get("data", []):
        lines.append(f"{entry.get('subsystem')}: {entry.get('status')} (num_user={entry.get('num_user', '?')})")
    return "\n".join(lines) if lines else "no health data returned"


def tool_unifi_topology(_args: dict) -> str:
    if unifi is None:
        return "UniFi is not configured on this bot"
    try:
        data = unifi.get("stat/device")
    except Exception as e:  # noqa: BLE001
        return f"UniFi request failed: {e}"
    lines = []
    for d in data.get("data", []):
        name = d.get("name", d.get("mac"))
        state = "online" if d.get("state") == 1 else f"state={d.get('state')}"
        up = d.get("uplink") or {}
        uplink_desc = (
            f"-> {up.get('uplink_device_name')} port {up.get('uplink_remote_port')}"
            if up.get("uplink_device_name")
            else "(no uplink -- likely the gateway)"
        )
        lines.append(f"{name} [{d.get('type', '?')}] {d.get('ip', '?')} {state} {uplink_desc}")
    return "\n".join(lines) if lines else "no devices returned"


def tool_unifi_alerts(_args: dict) -> str:
    # This console's classic Alarms API (list/alarm, rest/alarm) returned
    # errors on every path tried, so this is derived from device online
    # state/last_seen instead of the native Alarms feed -- still genuinely
    # useful, just labeled honestly.
    if unifi is None:
        return "UniFi is not configured on this bot"
    try:
        data = unifi.get("stat/device")
    except Exception as e:  # noqa: BLE001
        return f"UniFi request failed: {e}"
    now = time.time()
    alerts = []
    for d in data.get("data", []):
        name = d.get("name", d.get("mac"))
        if d.get("state") != 1:
            alerts.append(f"{name}: not online (state={d.get('state')})")
        last_seen = d.get("last_seen")
        if last_seen and now - last_seen > 600:
            alerts.append(f"{name}: last_seen {int((now - last_seen) / 60)}m ago (stale)")
    if not alerts:
        return (
            "no device-level alerts (derived from device online/last_seen state -- this "
            "controller's native Alarms API isn't reachable from here)"
        )
    return "\n".join(alerts)


# --------------------------------------------------------- hourly scan -----

HEALTH_ENDPOINTS = [
    ("litellm", "http://litellm:4000/health/readiness"),
    ("open-webui", "http://open-webui:8080/health"),
    # authentik-server-1 isn't on cyber-agent's network -- check the real
    # public path instead (also catches Caddy/DNS/cert issues, which is
    # arguably more useful than an internal-only check anyway).
    ("authentik", "https://auth.skynet.drewdettmer.com/-/health/live/"),
]


def _safe_status(container_name: str) -> str:
    try:
        return docker_client.containers.get(container_name).status
    except docker_sdk.errors.NotFound:
        return "missing"


def scan_for_issues() -> dict[str, str]:
    """Run every check once; return {issue_key: human description} for
    whatever is currently wrong. Empty dict == all clear."""
    issues: dict[str, str] = {}

    # container drift -- restart policy promises "running" but it isn't.
    # this is exactly the failure class that took litellm down for 19h
    # silently earlier tonight. Excludes the vLLM GPU-swap group: only one
    # of those configs is ever supposed to be running at a time (see the
    # dedicated GPU check below), so applying this generic heuristic to
    # them would flag the *expected* half as broken on every scan.
    gpu_swap_containers = set(GPU_VISION_CONTAINERS) | {GPU_CODER_CONTAINER}
    for c in docker_client.containers.list(all=True):
        if c.name in gpu_swap_containers:
            continue
        policy = c.attrs.get("HostConfig", {}).get("RestartPolicy", {}).get("Name", "")
        if policy in ("unless-stopped", "always") and c.status != "running":
            issues[f"container:{c.name}"] = (
                f"{c.name} should be running (restart={policy}) but is {c.status}"
            )

    # GPU config: exactly one of the two configs should be fully up. Flag
    # if neither is (nothing serving at all) or a coder-next container
    # exited on error (crashed rather than being intentionally stopped).
    vision_up = all(
        _safe_status(n) == "running" for n in GPU_VISION_CONTAINERS
    )
    coder_status = _safe_status(GPU_CODER_CONTAINER)
    if not vision_up and coder_status != "running":
        issues["gpu:none-serving"] = "no vLLM model is currently serving (neither GPU config is up)"
    coder_exit_code = docker_client.containers.get(GPU_CODER_CONTAINER).attrs.get("State", {}).get("ExitCode")
    if coder_status == "exited" and coder_exit_code not in (0, None):
        issues["gpu:coder-next-crashed"] = f"vllm-qwen3-coder-next exited with error code {coder_exit_code} (not a clean stop)"

    # service health
    for name, url in HEALTH_ENDPOINTS:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code >= 300:
                issues[f"health:{name}"] = f"{name} health check returned HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            issues[f"health:{name}"] = f"{name} health check failed: {e}"

    # cert expiry (<14 days, or unreachable)
    for line in tool_cert_expiry({}).splitlines():
        host = line.split(":", 1)[0]
        if "could not check" in line:
            issues[f"cert:{host}"] = line
        elif "expires in" in line:
            days = int(line.split("expires in ")[1].split(" ")[0])
            if days < 14:
                issues[f"cert:{host}"] = line

    # disk headroom
    total, used, _free = shutil.disk_usage("/")
    if total and used / total > 0.90:
        issues["disk:root"] = f"root disk {100 * used / total:.0f}% full"

    # backup freshness
    if os.path.exists(BACKUP_LOG_PATH):
        age_h = (time.time() - os.path.getmtime(BACKUP_LOG_PATH)) / 3600
        if age_h > 26:
            issues["backup:stale"] = f"backup log not updated in {age_h:.1f}h (expected daily)"
    else:
        issues["backup:missing"] = f"backup log not found at {BACKUP_LOG_PATH}"

    return issues


def reconcile_issues(current: dict[str, str]) -> list[str]:
    """Diff current issues against known state. Marks resolved issues
    inactive (so they can re-alert if they recur later) and returns
    descriptions for issues that are newly active -- these are the only
    ones that should trigger a DM, so the same ongoing issue never repeats."""
    conn = db()
    now = time.time()
    known_active = {
        row[0] for row in conn.execute("SELECT issue_key FROM issue_state WHERE active = 1")
    }

    new_alerts = []
    for key, description in current.items():
        if key not in known_active:
            new_alerts.append(description)
        conn.execute(
            "INSERT INTO issue_state (issue_key, active, description, first_seen, last_seen) "
            "VALUES (?, 1, ?, ?, ?) "
            "ON CONFLICT(issue_key) DO UPDATE SET active=1, description=excluded.description, last_seen=excluded.last_seen",
            (key, description, now, now),
        )

    resolved_keys = known_active - set(current.keys())
    for key in resolved_keys:
        conn.execute("UPDATE issue_state SET active = 0, last_seen = ? WHERE issue_key = ?", (now, key))

    conn.commit()
    conn.close()
    return new_alerts


def record_uptime_snapshot() -> None:
    """Called once per hourly scan tick. Writes one up/down row per tracked
    service regardless of whether anything is wrong -- scan_for_issues only
    records state *transitions*, so this is the only history uptime_report
    can compute a real percentage from."""
    conn = db()
    now = time.time()
    rows = []
    gpu_swap_containers = set(GPU_VISION_CONTAINERS) | {GPU_CODER_CONTAINER}
    for c in docker_client.containers.list(all=True):
        if c.name in gpu_swap_containers:
            continue
        policy = c.attrs.get("HostConfig", {}).get("RestartPolicy", {}).get("Name", "")
        if policy in ("unless-stopped", "always"):
            rows.append((f"container:{c.name}", now, 1 if c.status == "running" else 0))
    vision_up = all(_safe_status(n) == "running" for n in GPU_VISION_CONTAINERS)
    coder_up = _safe_status(GPU_CODER_CONTAINER) == "running"
    rows.append(("model:serving", now, 1 if (vision_up or coder_up) else 0))
    for name, url in HEALTH_ENDPOINTS:
        try:
            ok = requests.get(url, timeout=5).status_code < 300
        except Exception:  # noqa: BLE001
            ok = False
        rows.append((f"health:{name}", now, 1 if ok else 0))
    conn.executemany("INSERT INTO service_checks (service, ts, up) VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()


def tool_uptime_report(args: dict) -> str:
    days = int(args.get("days") or 7)
    cutoff = time.time() - days * 86400
    conn = db()
    rows = conn.execute(
        "SELECT service, SUM(up), COUNT(*) FROM service_checks WHERE ts > ? GROUP BY service",
        (cutoff,),
    ).fetchall()
    conn.close()
    if not rows:
        return f"no uptime data yet for the last {days} day(s) -- samples accumulate hourly going forward"
    lines = [f"uptime, last {days} day(s), hourly samples:"]
    for service, up_count, total in sorted(rows, key=lambda r: r[1] / r[2]):
        pct = 100.0 * up_count / total
        lines.append(f"{service}: {pct:.1f}% ({up_count}/{total} checks up)")
    return "\n".join(lines)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "docker_status",
            "description": "List every container's name, status, and image. Read-only.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "docker_logs",
            "description": "Get the last N log lines for a named container. Read-only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "container": {"type": "string", "description": "exact container name"},
                    "lines": {"type": "integer", "description": "number of lines, default 50"},
                },
                "required": ["container"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_restart",
            "description": (
                "Propose restarting a named container. Does NOT restart it -- it asks the "
                "user to confirm first. Use this instead of any direct restart action."
            ),
            "parameters": {
                "type": "object",
                "properties": {"container": {"type": "string", "description": "exact container name"}},
                "required": ["container"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gpu_status",
            "description": "Report which GPU config is currently active (vision+ablated vs coder-next). Read-only.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_swap_gpu_config",
            "description": (
                "Propose swapping the GPU config to 'vision' (qwen3-vl-8b+ablated) or "
                "'coder' (qwen3-coder-next). Does NOT swap it -- asks the user to confirm first."
            ),
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string", "enum": ["vision", "coder"]}},
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "backup_status",
            "description": "Check when the daily backup to shield last ran and whether it succeeded. Read-only.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cert_expiry",
            "description": "Check TLS cert expiry for a hostname, or all known lab hostnames if none given. Read-only.",
            "parameters": {
                "type": "object",
                "properties": {"host": {"type": "string", "description": "optional single hostname"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "disk_usage",
            "description": "Check root disk usage/headroom on skynet. Read-only.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unifi_clients",
            "description": "List clients currently connected to the UniFi network (name, IP, connection type). Read-only.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unifi_network_health",
            "description": "Get UniFi subsystem health (WAN/LAN/WLAN/VPN status). Read-only.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "container_stats",
            "description": "Live CPU% and memory usage per running container. Read-only.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unifi_topology",
            "description": "List UniFi devices (AP/switch/gateway) with model, IP, online state, and uplink (which device/port they connect to). Read-only.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unifi_alerts",
            "description": "Device-level UniFi alerts derived from online state and last_seen staleness. Read-only.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "uptime_report",
            "description": "Per-service uptime percentage over the last N days, from hourly health samples. Read-only.",
            "parameters": {
                "type": "object",
                "properties": {"days": {"type": "integer", "description": "lookback window, default 7"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "token_usage",
            "description": "Per-key LLM token usage (prompt/completion/total, call count) over the last N days from LiteLLM's spend log. Tokens only, no dollar spend. Read-only.",
            "parameters": {
                "type": "object",
                "properties": {"days": {"type": "integer", "description": "lookback window, default 7"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_read",
            "description": f"Read a text file from the {FS_ROOT} tree on skynet (line-numbered, optional offset/limit). Read-only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "absolute path, e.g. /home/skynet/llm-stack/docker-compose.yml"},
                    "offset": {"type": "integer", "description": "1-indexed starting line, default 1"},
                    "limit": {"type": "integer", "description": "max lines to return, default 300"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_glob",
            "description": f"Find files by glob pattern under {FS_ROOT} (or a given root). Read-only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "glob pattern, e.g. '**/*.yml'"},
                    "root": {"type": "string", "description": f"search root, default {FS_ROOT}"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_grep",
            "description": f"Regex-search a file or directory tree under {FS_ROOT} for a pattern. Read-only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "regular expression"},
                    "path": {"type": "string", "description": f"file or directory, default {FS_ROOT}"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web (DuckDuckGo) and return the top results' titles, URLs, and snippets. Read-only.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_fs_write",
            "description": (
                f"Propose writing/overwriting a text file under {FS_ROOT}. Does NOT write it -- "
                f"asks the user to confirm first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string", "description": "full file content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_run_bash",
            "description": (
                f"Propose running a bash command (cwd {FS_ROOT}, sees the mounted skynet "
                f"filesystem and the docker socket -- real host-visible access, not a sandbox). "
                f"Does NOT run it -- asks the user to confirm first."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]

TOOL_IMPLS = {
    "docker_status": tool_docker_status,
    "docker_logs": tool_docker_logs,
    "propose_restart": tool_propose_restart,
    "gpu_status": tool_gpu_status,
    "propose_swap_gpu_config": tool_propose_swap_gpu_config,
    "backup_status": tool_backup_status,
    "cert_expiry": tool_cert_expiry,
    "disk_usage": tool_disk_usage,
    "unifi_clients": tool_unifi_clients,
    "unifi_network_health": tool_unifi_network_health,
    "container_stats": tool_container_stats,
    "unifi_topology": tool_unifi_topology,
    "unifi_alerts": tool_unifi_alerts,
    "uptime_report": tool_uptime_report,
    "token_usage": tool_token_usage,
    "fs_read": tool_fs_read,
    "fs_glob": tool_fs_glob,
    "fs_grep": tool_fs_grep,
    "web_search": tool_web_search,
    "propose_fs_write": tool_propose_fs_write,
    "propose_run_bash": tool_propose_run_bash,
}

# channel_id -> {"action": "restart"|"swap_gpu_config", ...}
PENDING_CONFIRMATIONS: dict[str, dict] = {}


def run_agent_turn(channel_id: str, user_text: str) -> str:
    save_message(channel_id, "user", user_text)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + load_history(channel_id)

    last_error = None
    for model in MODEL_FALLBACK_ORDER:
        try:
            response = ai_client.chat.completions.create(
                model=model, messages=messages, tools=TOOLS, max_tokens=800
            )
            break
        except Exception as e:  # noqa: BLE001 - want to try the next model on any failure
            last_error = e
            log.warning("model %s failed: %s", model, e)
    else:
        return f"Every configured model failed to respond (likely the wrong GPU config is active). Last error: {last_error}"

    choice = response.choices[0]
    msg = choice.message

    # Tool-calling loop (single round -- fine for this tool set's shape).
    if msg.tool_calls:
        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            args["_channel_id"] = channel_id
            impl = TOOL_IMPLS.get(name)
            result = impl(args) if impl else f"unknown tool: {name}"
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result[:4000]}
            )
        followup = ai_client.chat.completions.create(model=model, messages=messages, max_tokens=800)
        final_text = followup.choices[0].message.content or "(no response)"
    else:
        final_text = msg.content or "(no response)"

    save_message(channel_id, "assistant", final_text)
    return final_text


def _do_swap_gpu_config(target: str) -> str:
    try:
        if target == "coder":
            docker_client.containers.get(GPU_CODER_CONTAINER)
            for name in GPU_VISION_CONTAINERS:
                docker_client.containers.get(name).stop()
            docker_client.containers.get(GPU_CODER_CONTAINER).start()
            return "Stopped vision+ablated, started coder-next. It takes ~1-2 min to finish loading."
        else:
            docker_client.containers.get(GPU_CODER_CONTAINER).stop()
            for name in GPU_VISION_CONTAINERS:
                docker_client.containers.get(name).start()
            return "Stopped coder-next, started vision+ablated. It takes ~1-2 min to finish loading."
    except Exception as e:  # noqa: BLE001
        return f"GPU swap failed: {e}"


def execute_pending_action(pending: dict) -> str:
    if pending["action"] == "restart":
        name = pending["container"]
        try:
            c = docker_client.containers.get(name)
            c.restart(timeout=10)
            return f"Restarted '{name}'."
        except Exception as e:  # noqa: BLE001
            return f"Restart of '{name}' failed: {e}"

    if pending["action"] == "swap_gpu_config":
        return _do_swap_gpu_config(pending["target"])

    if pending["action"] == "fs_write":
        try:
            path = pending["path"]
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(pending["content"])
            return f"Wrote {len(pending['content'])} byte(s) to '{path}'."
        except Exception as e:  # noqa: BLE001
            return f"write to '{pending['path']}' failed: {e}"

    if pending["action"] == "run_bash":
        try:
            result = subprocess.run(
                ["bash", "-c", pending["command"]],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=FS_ROOT,
            )
            out = (result.stdout + result.stderr).strip()
            return out[:1800] if out else f"(no output, exit code {result.returncode})"
        except Exception as e:  # noqa: BLE001
            return f"command failed: {e}"

    return f"Unknown pending action: {pending['action']}"


def build_confirmation_embed(pending: dict) -> discord.Embed:
    if pending["action"] == "restart":
        title = "Confirm restart"
        desc = f"Restart container `{pending['container']}`?"
    elif pending["action"] == "swap_gpu_config":
        title = "Confirm GPU config swap"
        desc = (
            f"Swap the active GPU config to **{pending['target']}**? "
            f"This stops the other config's containers first (~1-2 min to load)."
        )
    elif pending["action"] == "fs_write":
        title = "Confirm file write"
        desc = f"Write {len(pending['content'])} byte(s) to `{pending['path']}`? Overwrites if it exists."
    elif pending["action"] == "run_bash":
        title = "Confirm bash command"
        desc = f"Run `{pending['command']}` (cwd `{FS_ROOT}`)?"
    else:
        title = "Confirm action"
        desc = str(pending)
    return discord.Embed(title=title, description=desc, color=discord.Color.orange())


class ConfirmView(discord.ui.View):
    def __init__(self, channel_id: str, pending: dict):
        super().__init__(timeout=300)
        self.channel_id = channel_id
        self.pending = pending
        self.message: discord.Message | None = None

    async def on_timeout(self) -> None:
        if PENDING_CONFIRMATIONS.get(self.channel_id) is self.pending:
            del PENDING_CONFIRMATIONS[self.channel_id]
        for item in self.children:
            item.disabled = True
        if self.message is not None:
            embed = self.message.embeds[0]
            embed.color = discord.Color.greyple()
            embed.set_footer(text="Expired -- no action taken.")
            try:
                await self.message.edit(embed=embed, view=self)
            except Exception:  # noqa: BLE001
                log.exception("failed to edit expired confirmation message")

    async def _resolve(self, interaction: discord.Interaction, confirmed: bool) -> None:
        if interaction.user.id != OWNER_ID:
            await interaction.response.send_message("Not yours to confirm.", ephemeral=True)
            return
        if PENDING_CONFIRMATIONS.get(self.channel_id) is not self.pending:
            await interaction.response.send_message(
                "This confirmation is no longer active.", ephemeral=True
            )
            return
        del PENDING_CONFIRMATIONS[self.channel_id]
        self.stop()
        for item in self.children:
            item.disabled = True

        result = execute_pending_action(self.pending) if confirmed else "Cancelled."
        embed = interaction.message.embeds[0]
        embed.color = discord.Color.green() if confirmed else discord.Color.greyple()
        embed.description = result
        embed.set_footer(text="Confirmed" if confirmed else "Cancelled")
        await interaction.response.edit_message(embed=embed, view=self)
        save_message(self.channel_id, "assistant", result)

    @discord.ui.button(label="Confirm", style=discord.ButtonStyle.success)
    async def confirm(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._resolve(interaction, confirmed=True)

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.danger)
    async def cancel(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._resolve(interaction, confirmed=False)


# ------------------------------------------------------------------ discord --

intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True
client = discord.Client(intents=intents)


@tasks.loop(hours=1)
async def hourly_scan():
    try:
        issues = scan_for_issues()
        alerts = reconcile_issues(issues)
        record_uptime_snapshot()
    except Exception:  # noqa: BLE001
        log.exception("hourly scan failed")
        return
    if not alerts:
        log.info("hourly scan: all clear")
        return
    log.info("hourly scan: %d new issue(s)", len(alerts))
    try:
        user = await client.fetch_user(OWNER_ID)
        text = "cyber-agent hourly scan found new issues:\n" + "\n".join(f"- {a}" for a in alerts)
        for chunk_start in range(0, len(text), 1900):
            await user.send(text[chunk_start : chunk_start + 1900])
    except Exception:  # noqa: BLE001
        log.exception("failed to DM owner about new issues")


@client.event
async def on_ready():
    log.info("logged in as %s (id=%s)", client.user, client.user.id)
    if not hourly_scan.is_running():
        hourly_scan.start()


@client.event
async def on_message(message: discord.Message):
    if message.author.id == client.user.id:
        return
    if not isinstance(message.channel, discord.DMChannel):
        return
    if message.author.id != OWNER_ID:
        log.warning("ignoring DM from unauthorized user %s (%s)", message.author, message.author.id)
        return

    channel_id = str(message.channel.id)
    text = message.content.strip()
    if not text:
        return

    async with message.channel.typing():
        try:
            reply = run_agent_turn(channel_id, text)
        except Exception as e:  # noqa: BLE001
            log.exception("turn failed")
            reply = f"Error: {e}"

    for chunk_start in range(0, len(reply), 1900):
        await message.channel.send(reply[chunk_start : chunk_start + 1900])

    pending = PENDING_CONFIRMATIONS.get(channel_id)
    if pending is not None:
        embed = build_confirmation_embed(pending)
        view = ConfirmView(channel_id, pending)
        view.message = await message.channel.send(embed=embed, view=view)


if __name__ == "__main__":
    client.run(DISCORD_TOKEN, log_handler=None)
