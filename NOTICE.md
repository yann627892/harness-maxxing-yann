# Notice

This repository is a personal fork of [Harness Maxxing](https://github.com/makushingu/harness-maxxing)
(upstream commit `5cc35042`, "Harness Maxxing 2.0.0a138"), modified for private use under the
Harness Maxxing Fair Source License 1.0 (see `LICENSE`, unchanged, copyright notice included).

It is **not** Harness Maxxing, is not maintained by the Harness Maxxing authors, and must not be
distributed or offered under their name. Only the changes below were made; everything else is the
upstream code.

## Changes

**LLM gateway support** — connect the Claude Code CLI to an Anthropic-compatible gateway
(OpenCode Zen pay-as-you-go, OpenCode Go, OpenRouter, or a custom endpoint) instead of an
Anthropic account, from Settings -> Claude or the onboarding:

- `server/maxxing/gateway.py` (new): provider presets, base-URL validation, connection state
  (secrets + config), the turn environment (`ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`,
  `ANTHROPIC_API_KEY=""`, the `ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL` role pins), the
  model ids for the picker and for background calls.
- `server/maxxing/api/claude.py`: `POST /api/claude/auth` accepts `kind:"gateway"`
  (`provider`, `base_url`); `GET /api/claude/status` exposes `gateway`; the test call uses the
  gateway's cheapest model.
- `server/maxxing/sessions/spawn.py`, `vault.py`: a connected gateway replaces the stored
  Anthropic credentials; `ANTHROPIC_AUTH_TOKEN` joins the reserved secret names.
- `server/maxxing/claude_auth.py`, `cmd_core.py`, `health.py`: a `gateway` auth method
  (stored, or `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN` in the service environment) and its
  sign-in error detection.
- `server/maxxing/sessions/service.py`: the model picker lists the gateway's ids;
  `titles.py` / `team.py`: background one-line calls use the gateway's models.
- `web/src/…`: a "Gateway" tab in Settings -> Claude (provider choice, custom address, API key),
  the gateway badge in the status card, types and translations (English, French).
- `docs/…`, `server/tests/test_gateway.py`: documentation and tests for the above.
- `web/dist` rebuilt.
