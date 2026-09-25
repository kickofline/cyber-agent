# cyber-agent

A small Discord bot bridging DMs to the skynet LLM lab (Ouachita Cyber) via
LiteLLM, with a few docker-management tools.

Deliberately simple: one Python file, no workflow engine, no separate
database server. Connects to Discord over its outbound Gateway WebSocket, so
there is no webhook, no reverse proxy, and no public hostname needed.

## Tools

- `docker_status` -- list every container's name/status/image (read-only)
- `docker_logs` -- tail a named container's logs (read-only)
- `propose_restart` / confirm-with-'yes' -- restart a named container, but
  only after the user explicitly confirms in the same DM. The model can
  never restart something without a human saying 'yes' first.

## Config

Env vars (see `.env`, not committed): `DISCORD_TOKEN`, `DISCORD_OWNER_ID`
(only this Discord user ID gets responses), `LITELLM_API_KEY`. Model
fallback order (`MODEL_FALLBACK_ORDER`) tries each model in turn -- useful
since this lab's GPU only runs one config at a time.

## Deploy

`docker compose up -d --build`. Conversation history persists to a small
SQLite file in the `cyber_agent_data` volume.
