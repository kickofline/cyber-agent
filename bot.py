"""
cyber-agent — a small Discord bot bridging DMs to the skynet LLM stack
(via LiteLLM) with a handful of docker-management tools.

Deliberately simple: one file, no workflow engine, no separate database
server. Connects to Discord over its outbound Gateway WebSocket (discord.py
handles this) so there is no webhook, no reverse proxy, and no public
hostname needed.
"""

import json
import logging
import os
import sqlite3
import time

import discord
import docker as docker_sdk
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

SYSTEM_PROMPT = (
    "You are cyber-agent, an infra assistant for the Ouachita Cyber / skynet LLM lab. "
    "You run inside a Discord DM. Keep replies short and plain text (no markdown tables). "
    "You have read-only tools to check container status and logs, and one mutating tool "
    "(restart_container) that requires the user's explicit confirmation before it runs -- "
    "when you want to restart something, call propose_restart, which will ask the user to "
    "confirm; do not claim an action succeeded until you have tool output showing it did. "
    "GPU note: only one of {qwen3-coder-next} or {qwen3-vl-8b + qwen3-8b-ablated} is running "
    "at a time on this box -- if a model call fails, that GPU config may be inactive right now."
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
        f"PENDING_CONFIRMATION: proposed restarting '{name}'. Tell the user to reply "
        f"'yes' to confirm or anything else to cancel -- do not say it has restarted yet."
    )


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
]

TOOL_IMPLS = {
    "docker_status": tool_docker_status,
    "docker_logs": tool_docker_logs,
    "propose_restart": tool_propose_restart,
}

# channel_id -> {"action": "restart", "container": "..."}
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

    # Tool-calling loop (single round -- fine for the small tool set here).
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


def handle_confirmation(channel_id: str, user_text: str) -> str | None:
    pending = PENDING_CONFIRMATIONS.get(channel_id)
    if not pending:
        return None
    del PENDING_CONFIRMATIONS[channel_id]
    if user_text.strip().lower() != "yes":
        return "Cancelled."
    name = pending["container"]
    try:
        c = docker_client.containers.get(name)
        c.restart(timeout=10)
        return f"Restarted '{name}'."
    except Exception as e:  # noqa: BLE001
        return f"Restart of '{name}' failed: {e}"


# ------------------------------------------------------------------ discord --

intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    log.info("logged in as %s (id=%s)", client.user, client.user.id)


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
        confirmation_reply = handle_confirmation(channel_id, text)
        if confirmation_reply is not None:
            reply = confirmation_reply
        else:
            try:
                reply = run_agent_turn(channel_id, text)
            except Exception as e:  # noqa: BLE001
                log.exception("turn failed")
                reply = f"Error: {e}"

    for chunk_start in range(0, len(reply), 1900):
        await message.channel.send(reply[chunk_start : chunk_start + 1900])


if __name__ == "__main__":
    client.run(DISCORD_TOKEN, log_handler=None)
