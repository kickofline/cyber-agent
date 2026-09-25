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
import shutil
import socket
import sqlite3
import ssl
import subprocess
import time

import discord
import docker as docker_sdk
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

SYSTEM_PROMPT = (
    "You are cyber-agent, an infra assistant for the Ouachita Cyber / skynet LLM lab. "
    "You run inside a Discord DM. Keep replies short and plain text (no markdown tables). "
    "Read-only tools: docker_status, docker_logs, gpu_status, backup_status, cert_expiry, "
    "disk_usage, unifi_clients, unifi_network_health. "
    "Mutating tools (propose_restart, propose_swap_gpu_config) never execute immediately -- "
    "they ask the user to confirm, and only a literal 'yes' reply executes them. Do not claim "
    "an action succeeded until you have tool output showing it did. "
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
