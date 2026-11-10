"""An LLM gateway in place of the Anthropic credentials: OpenCode Zen (pay-as-you-go or Go),
OpenRouter, or any Anthropic-compatible endpoint.

Claude Code speaks the Anthropic Messages API to a gateway natively (ANTHROPIC_BASE_URL +
ANTHROPIC_AUTH_TOKEN, the bearer credential variable Anthropic documents for gateways). When a
gateway is connected, every Claude turn gets:
- ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN from secrets.json (chmod 600, like the other
  credentials);
- ANTHROPIC_API_KEY="" (empty on purpose: no x-api-key header, no fallback to
  api.anthropic.com and no conflict with a leftover Anthropic key - what OpenRouter and OpenCode
  Zen both ask for);
- the ANTHROPIC_DEFAULT_OPUS/SONNET/HAIKU_MODEL pins, so Claude Code's own model picks
  (background tasks, session titles) ask for gateway ids instead of Anthropic ids the gateway
  may not know.

Connection state: secrets.json "anthropic_base_url" + "anthropic_auth_token" (the bearer key),
config.json "gateway" = {"provider": "zen"|"zen_go"|"openrouter"|"custom", "models": {"opus": …,
"sonnet": …, "haiku": …, "test": …}}. The api/claude.py `set_auth` endpoint fills it from the
presets below; the ids stay editable (Settings -> Claude default model, /model in Telegram) and
ids seen in session logs join the picker like the Anthropic ones.
"""
from __future__ import annotations

import re
from typing import Any

from . import config

# Claude Code appends /v1/messages to the base URL: leave that path off (see each provider's docs).
PRESETS: dict[str, dict] = {
    "zen": {
        "name": "OpenCode Zen",
        "base_url": "https://opencode.ai/inference/anthropic",
        "key_hint": "oc_sk_…",
        "key_url": "https://opencode.ai/console",
        "default_model": "claude-sonnet-5-5",
        "models": {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5",
                   "haiku": "claude-haiku-5-5", "test": "claude-haiku-5-5"},
        "extra": ["claude-fable-5-2", "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-opus-5",
                  "claude-haiku-4-5", "gpt-6.1-sol", "gpt-6-sol", "gpt-6-luna", "gpt-5.6-terra",
                  "gemini-4-argon", "gemini-3.8-flash", "qwen3.6-plus", "kimi-k3", "deepseek-v4-pro",
                  "glm-5.3", "minimax-m3", "grok-4.7", "mistral-large-4"],
    },
    "zen_go": {
        "name": "OpenCode Go",
        "base_url": "https://opencode.ai/inference/go/anthropic",
        "key_hint": "oc_sk_…",
        "key_url": "https://opencode.ai/console",
        "default_model": "claude-sonnet-5-5",
        "models": {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5",
                   "haiku": "claude-haiku-5-5", "test": "claude-haiku-5-5"},
        "extra": ["claude-fable-5-2", "claude-sonnet-4-6", "claude-opus-5", "claude-haiku-4-5",
                  "gpt-6.1-sol", "gpt-6-luna", "gemini-4-argon", "minimax-m3", "qwen3.6-plus"],
    },
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api",
        "key_hint": "sk-or-…",
        "key_url": "https://openrouter.ai/settings/keys",
        "default_model": "anthropic/claude-sonnet-5.5",
        "models": {"opus": "anthropic/claude-opus-5.5", "sonnet": "anthropic/claude-sonnet-5.5",
                   "haiku": "anthropic/claude-haiku-5.5", "test": "anthropic/claude-haiku-5.5"},
        # OpenRouter itself warns Claude Code is only guaranteed to work with Anthropic models.
        "extra": ["anthropic/claude-fable-5.2", "anthropic/claude-opus-5", "anthropic/claude-sonnet-5",
                  "anthropic/claude-haiku-4.5", "anthropic/claude-opus-4.8"],
    },
    "custom": {
        "name": "Custom gateway",
        "base_url": "",                      # typed by the user
        "key_hint": "the gateway's bearer key",
        "key_url": "",
        "default_model": "",
        "models": {"opus": "", "sonnet": "", "haiku": "", "test": ""},
        "extra": [],
    },
}
ROLES = ("opus", "sonnet", "haiku", "test")

# The model of the one-line background calls (titles, team summaries) when no gateway is connected.
ANTHROPIC_SMALL = "haiku"        # Claude Code's name for its newest Haiku (the cheapest family)

_SCHEME_RE = re.compile(r"^(https?)://([^/?#]+)([^?#]*)$", re.I)   # a ? or # left over = refused
_HOST_RE = re.compile(r"^[A-Za-z0-9.\[\]-]+$")            # a dns name, an IPv4 or a bracketed IPv6
_PRIVATE_HOST = re.compile(r"^(localhost|127\.|10\.|192\.168\.|169\.254\.|[fF][cCdD][0-9a-fA-F]{2}:|\[?::1\]?|"
                           r"172\.(1[6-9]|2[0-9]|3[01])\.)")


def _authority(authority: str) -> tuple[str, str] | None:
    """(host, ":port" or "") of an URL authority, or None when it is not a host[:port] an
    Anthropic-compatible gateway could live at."""
    if authority.startswith("["):                          # [::1] or [::1]:8080
        end = authority.find("]")
        if end < 0:
            return None
        host, rest = authority[:end + 1], authority[end + 1:]
        if rest and (not rest.startswith(":") or not rest[1:].isdigit() or not 0 < int(rest[1:]) < 65536):
            return None
        return host, rest
    if ":" in authority:                                    # name:port
        host, _, port = authority.rpartition(":")
        if not port.isdigit() or not 0 < int(port) < 65536:
            return None
        if not _HOST_RE.match(host):
            return None
        return host, f":{port}"
    return (authority, "") if _HOST_RE.match(authority) else None


def valid_base_url(url: str) -> str | None:
    """A gateway base URL Claude Code can use (it appends /v1/messages): https (http only for a
    local or private address), optional port, no query, no fragment, at most 300 characters.
    None = invalid."""
    url = (url or "").strip()
    if not url or len(url) > 300 or not (url.isascii() and url.isprintable()) or any(c.isspace() for c in url):
        return None
    m = _SCHEME_RE.match(url)
    if not m or not m.group(2):
        return None
    scheme, authority = m.group(1).lower(), m.group(2)
    hp = _authority(authority)
    if not hp:
        return None
    host, port = hp
    host = host.lower()
    path = m.group(3)
    if path.endswith("/"):                              # "https://gw.example.com/api/" -> "…/api"
        path = path.rstrip("/")
    if scheme != "https" and not _PRIVATE_HOST.match(host):
        return None
    return f"{scheme}://{host}{port}{path}"


def configured(secrets: dict | None = None) -> dict | None:
    """The connected gateway: {provider, name, base_url, models, extra}, or None. `models` are the
    pinned gateway ids per role (opus/sonnet/haiku/test, "" = not pinned)."""
    s = secrets if secrets is not None else config.load_secrets()
    base = str(s.get("anthropic_base_url") or "").strip()
    if not base:
        return None
    cfg = config.load_config().get("gateway")
    cfg = cfg if isinstance(cfg, dict) else {}
    provider = str(cfg.get("provider") or "custom").strip().lower()
    preset = PRESETS.get(provider, PRESETS["custom"])
    raw = cfg.get("models")
    raw = raw if isinstance(raw, dict) else {}
    models = {k: str(raw.get(k) or preset["models"].get(k) or "") for k in ROLES}
    return {"provider": provider, "name": preset["name"], "base_url": base, "models": models,
            "extra": [str(x) for x in preset.get("extra") or []]}


def describe(secrets: dict | None = None) -> str:
    """The auth_method() detail line of a connected gateway."""
    gw = configured(secrets)
    return f"{gw['name']} gateway key stored in Harness Maxxing" if gw else ""


def role_models(gw: dict | None = None) -> dict[str, str]:
    """{opus,sonnet,haiku} -> gateway id for the ANTHROPIC_DEFAULT_*_MODEL pins (no empty value).
    With no pinned id, all three fall back to the default model so background tasks keep working."""
    gw = gw if gw is not None else configured()
    if not gw:
        return {}
    out = {r: gw["models"].get(r) or "" for r in ("opus", "sonnet", "haiku")}
    if not any(out.values()):
        dm = str(config.load_config().get("default_model") or "").strip()
        if dm:
            out = {r: dm for r in out}
    return {k: v for k, v in out.items() if v}


def test_model(gw: dict | None = None) -> str | None:
    """A cheap model id the gateway serves, for the Settings -> Claude test call."""
    gw = gw if gw is not None else configured()
    if not gw:
        return None
    return gw["models"].get("test") or gw["models"].get("haiku") or gw["models"].get("sonnet") \
        or str(config.load_config().get("default_model") or "").strip() or None


def small_model(role: str = "haiku") -> str:
    """The model of a one-line background call (session titles, team summaries): the gateway's id
    for that role when connected, else Claude Code's cheapest known Anthropic model."""
    gw = configured()
    if gw:
        m = gw["models"].get(role) or gw["models"].get("haiku") or gw["models"].get("sonnet")
        if m:
            return m
        dm = str(config.load_config().get("default_model") or "").strip()
        if dm:
            return dm
    return ANTHROPIC_SMALL


def model_ids(gw: dict | None = None) -> list[str]:
    """Gateway ids for the model picker: the pinned roles, the default model and the preset list."""
    gw = gw if gw is not None else configured()
    if not gw:
        return []
    out = list(dict.fromkeys([gw["models"].get(r) for r in ("sonnet", "opus", "haiku")]
                             + [str(config.load_config().get("default_model") or "").strip()]
                             + [str(x) for x in gw["extra"]]))
    return [m for m in out if m]


def auth_env(out: dict[str, str], secrets: dict | None = None) -> dict[str, str]:
    """The gateway variables of a Claude process, merged into `out` (spawn.auth_env): a connected
    gateway replaces the Anthropic credentials entirely - no CLAUDE_CODE_OAUTH_TOKEN, no real
    ANTHROPIC_API_KEY ("" keeps Claude Code from falling back to api.anthropic.com)."""
    gw = configured(secrets)
    if not gw:
        return out
    s = secrets if secrets is not None else config.load_secrets()
    out["ANTHROPIC_BASE_URL"] = gw["base_url"]
    token = str(s.get("anthropic_auth_token") or "")
    if token:
        out["ANTHROPIC_AUTH_TOKEN"] = token
    out["ANTHROPIC_API_KEY"] = ""
    for role, model in role_models(gw).items():
        out[f"ANTHROPIC_DEFAULT_{role.upper()}_MODEL"] = model
    return out


def clear() -> None:
    """Remove the gateway connection (secrets + config): another credential takes over."""
    config.set_secrets({"anthropic_base_url": None, "anthropic_auth_token": None})
    cfg = config.load_config()
    if isinstance(cfg.get("gateway"), dict):            # update_config(None) would keep the section
        cfg.pop("gateway", None)
        config.save_config(cfg)


def connect(provider: str, base_url: str | None, value: str) -> dict[str, Any]:
    """Store a gateway connection (api/claude.py set_auth): provider preset (or custom + base_url),
    the bearer key, the pinned models, and a default model when none is set yet. Raises ValueError
    when the provider or the base URL is not acceptable."""
    provider = str(provider or "custom").strip().lower()
    if provider not in PRESETS:
        raise ValueError("unknown gateway provider")
    preset = PRESETS[provider]
    base = valid_base_url(base_url or preset["base_url"])
    if not base:
        raise ValueError("the gateway address must be an https URL (http only on a private address)")
    cfg = config.load_config()
    changes: dict[str, Any] = {"gateway": {"provider": provider, "models": dict(preset["models"])}}
    if not str(cfg.get("default_model") or "").strip() and preset["default_model"]:
        changes["default_model"] = preset["default_model"]
    config.update_config(**changes)
    config.set_secrets({"anthropic_base_url": base, "anthropic_auth_token": value,
                        "claude_oauth_token": None, "anthropic_api_key": None})
    return {"provider": provider, "base_url": base, "models": dict(preset["models"])}


def public() -> dict | None:
    """What /api/claude/status says about the connection: provider, name, base url, pinned models."""
    gw = configured()
    if not gw:
        return None
    return {"provider": gw["provider"], "name": gw["name"], "base_url": gw["base_url"],
            "models": {k: gw["models"][k] for k in ROLES}}
