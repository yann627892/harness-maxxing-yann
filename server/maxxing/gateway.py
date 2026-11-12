"""An LLM gateway beside the Anthropic credentials: OpenCode Zen (pay-as-you-go or Go),
OpenRouter, or any Anthropic-compatible endpoint.

Claude Code speaks the Anthropic Messages API to a gateway natively (ANTHROPIC_BASE_URL +
ANTHROPIC_AUTH_TOKEN, the bearer credential variable Anthropic documents for gateways). A gateway
and the Anthropic credentials (a subscription token, an API key, the CLI's own sign-in) can be
stored at the same time: each session picks its provider with its model (the picker lists both),
and the config `claude_provider` decides the default one - new sessions and the background calls
(titles, team summaries) follow it. A turn on the gateway gets:
- ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN from secrets.json (chmod 600, like the other
  credentials);
- ANTHROPIC_API_KEY="" (empty on purpose: no x-api-key header, no fallback to
  api.anthropic.com and no conflict with a leftover Anthropic key - what OpenRouter and OpenCode
  Zen both ask for);
- the ANTHROPIC_DEFAULT_OPUS/SONNET/HAIKU_MODEL pins, so Claude Code's own model picks
  (background tasks, session titles) ask for gateway ids instead of Anthropic ids the gateway
  may not know.
A turn on Anthropic gets none of that: spawn.auth_env gives it the stored subscription token /
API key instead (or nothing, when the CLI is signed in itself).

Connection state: config.json "gateways" = {id: {"provider": "zen"|"zen_go"|"openrouter"|"custom",
"base_url": …, "models": {"opus": …, "sonnet": …, "haiku": …, "test": …}}}, one entry per connected
gateway (id = the preset key); secrets.json "gateway_key_<id>" = that gateway's bearer key. An
older install's single gateway (secrets "anthropic_base_url" + "anthropic_auth_token", config
"gateway") is read as one entry and moved to this shape by the next connection. The api/claude.py
`set_auth` endpoint fills an entry from the presets below; the ids stay editable (Settings ->
Claude default model, /model in Telegram) and ids seen in session logs join the picker like the
Anthropic ones.
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
        # Go serves the subscription's open-weight models, not the Anthropic ones: Qwen3.8 Max and
        # Claude Haiku 5.5 speak the Messages API natively, the other families go through Console's
        # translation (GET https://opencode.ai/inference/go/v1/models lists what the plan covers).
        "default_model": "qwen3.8-max",
        "models": {"opus": "qwen3.8-max", "sonnet": "qwen3.8-max",
                   "haiku": "claude-haiku-5-5", "test": "claude-haiku-5-5"},
        "extra": ["qwen3.8-flash", "qwen3.7-plus", "glm-5.3", "glm-5.3-flash", "glm-5.2",
                  "kimi-k3.1-pro", "kimi-k3", "kimi-k2.7-code", "deepseek-v4-pro",
                  "deepseek-v4.1-flash", "minimax-m3", "minimax-m3.1-flash", "grok-4.7",
                  "gpt-6-luna", "gpt-5.6-luna", "gemini-4-argon", "mimo-v2.6-pro",
                  "mimo-v2.6-flash", "mimo-v2.5", "mimo-v2.5-pro", "hy4", "hy4-preview", "hy3",
                  "longcat-2.0", "step-5-preview", "muse-spark-1.3-contributor",
                  "muse-spark-1.2-contributor", "beam-free", "iceman-contributor-free",
                  "longcat-2.5-free"],
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
                  "anthropic/claude-haiku-4.5", "anthropic/claude-opus-4.8",
                  # other families run through OpenRouter's translation of the Anthropic API, not guaranteed
                  "openai/gpt-6.1-terra-pro", "openai/gpt-6.1-terra", "google/gemini-4-argon",
                  "x-ai/grok-4.7", "deepseek/deepseek-v4.1-flash", "moonshotai/kimi-k3.1-pro",
                  "qwen/qwen3.8-max-prime", "z-ai/glm-5.3-prime", "z-ai/glm-5.3-flash"],
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
_CLAUDE_ID = re.compile(r"claude-(?:opus|sonnet|haiku|fable)-[a-z0-9.-]+")   # an Anthropic-served id
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


# A turn runs on "anthropic" (the stored subscription token / API key, or the CLI's own sign-in)
# or on a connected gateway: "gateway:<id>", id being the preset key (zen, zen_go, openrouter,
# custom). The bare "gateway" of sessions from before several gateways existed means the default one.
GATEWAY = "gateway"
_KEY = "gateway_key_"          # secrets.json key of one gateway's bearer token


def gw_provider(gw_id: str) -> str:
    """The provider token of one gateway ("gateway:zen_go")."""
    return f"{GATEWAY}:{gw_id}"


def is_gateway(provider: str | None) -> bool:
    """True for "gateway" (older sessions: the default one) and "gateway:<id>"."""
    p = (provider or "").strip().lower()
    return p == GATEWAY or p.startswith(GATEWAY + ":")


def _gw_id(provider: str | None) -> str:
    """The id of a "gateway:<id>" token, "" for "gateway" or anything else."""
    p = (provider or "").strip().lower()
    return p[len(GATEWAY) + 1:] if p.startswith(GATEWAY + ":") else ""


def _entry(gw_id: str, provider: str, base_url: str, raw: Any, token: str) -> dict:
    """One gateway, with the preset's pins filled in where the stored ones are empty."""
    preset = PRESETS.get(provider, PRESETS["custom"])
    raw = raw if isinstance(raw, dict) else {}
    models = {k: str(raw.get(k) or preset["models"].get(k) or "") for k in ROLES}
    return {"id": gw_id, "provider": provider, "name": preset["name"], "base_url": base_url,
            "models": models, "extra": [str(x) for x in preset.get("extra") or []], "token": token}


def _stored() -> dict:
    """config.json "gateways": {id: {provider, base_url, models}} (the token stays in secrets.json)."""
    g = config.load_config().get("gateways")
    return {str(k): v for k, v in g.items() if isinstance(v, dict)} if isinstance(g, dict) else {}


def _legacy(secrets: dict) -> tuple[str, str, Any, str] | None:
    """The single-gateway shape of older installs: anthropic_base_url + anthropic_auth_token in
    secrets.json, provider + models in config.json "gateway". -> (id, base_url, models, token)."""
    base = str(secrets.get("anthropic_base_url") or "").strip()
    token = str(secrets.get("anthropic_auth_token") or "")
    if not base:
        return None
    cfg = config.load_config().get("gateway")
    cfg = cfg if isinstance(cfg, dict) else {}
    provider = str(cfg.get("provider") or "custom").strip().lower()
    provider = provider if provider in PRESETS else "custom"
    return provider, base, cfg.get("models"), token


def gateways(secrets: dict | None = None) -> dict[str, dict]:
    """Every connected gateway, by id, in config order: {id, provider, name, base_url, models,
    extra, token}. `models` are the pinned ids per role (opus/sonnet/haiku/test, "" = not pinned).
    The single gateway of an older install shows up too, under its preset id."""
    s = secrets if secrets is not None else config.load_secrets()
    out: dict[str, dict] = {}
    for gid, entry in _stored().items():
        provider = str(entry.get("provider") or gid).strip().lower()
        base = str(entry.get("base_url") or "").strip()
        token = str(s.get(_KEY + gid) or "")
        if base:
            out[gid] = _entry(gid, provider, base, entry.get("models"), token)
    leg = _legacy(s)
    if leg:
        gid, base, models, token = leg
        if gid not in out and base:
            out[gid] = _entry(gid, gid, base, models, token)
    return out


def default_gateway(secrets: dict | None = None) -> dict | None:
    """The gateway background calls and older sessions use: the one config `claude_provider`
    names, else the first connected one."""
    gws = gateways(secrets)
    if not gws:
        return None
    p = str(config.load_config().get("claude_provider") or "").strip().lower()
    if p.startswith(GATEWAY + ":"):
        g = gws.get(p[len(GATEWAY) + 1:])
        if g:
            return g
    return next(iter(gws.values()))


def get(provider: str | None = None, secrets: dict | None = None) -> dict | None:
    """The gateway a provider token points at: "gateway:<id>"; "gateway" or None = the default one."""
    gid = _gw_id(provider)
    if gid:
        return gateways(secrets).get(gid)
    return default_gateway(secrets)


def configured(secrets: dict | None = None) -> dict | None:
    """The default gateway (older callers); see gateways() for all of them."""
    return default_gateway(secrets)


def anthropic_configured(secrets: dict | None = None) -> bool:
    """The Anthropic credentials exist: a stored subscription token or API key, or the CLI's own
    sign-in (`claude login`, ~/.claude/.credentials.json)."""
    s = secrets if secrets is not None else config.load_secrets()
    if s.get("claude_oauth_token") or s.get("anthropic_api_key"):
        return True
    return (config.claude_dir() / ".credentials.json").is_file()


def default_provider(secrets: dict | None = None) -> str:
    """Which credential a turn uses when its session does not say: config `claude_provider`
    ("anthropic" | "gateway:<id>"), else the connected one (a gateway when both are). A stored
    choice pointing at a credential that is gone falls back to the other one."""
    s = secrets if secrets is not None else config.load_secrets()
    p = str(config.load_config().get("claude_provider") or "").strip().lower()
    gws = gateways(s)
    an = anthropic_configured(s)
    if p == "anthropic":
        return "anthropic" if an else (gw_provider(next(iter(gws))) if gws else "anthropic")
    gid = _gw_id(p)
    if gid and gid in gws:
        return p
    if gws:                                  # no choice (or "gateway", the older default): first gateway
        return gw_provider(next(iter(gws)))
    return "anthropic"


def provider_of(meta: dict | None, secrets: dict | None = None) -> str:
    """The provider of a session's turns (runner, aside): its stored choice, else - a session from
    before both were connected - the origin of its model id, else the default provider."""
    s = secrets if secrets is not None else config.load_secrets()
    p = str((meta or {}).get("provider") or "").strip().lower()
    gws = gateways(s)
    if p == "anthropic":
        return "anthropic"
    if p.startswith(GATEWAY + ":"):
        return p if p[len(GATEWAY) + 1:] in gws else default_provider(s)
    if p == GATEWAY:                                # older session: the default gateway
        return gw_provider(next(iter(gws))) if gws else default_provider(s)
    model = str((meta or {}).get("model") or "").strip()
    if model and gws:
        order = list(gws.values())
        d = default_gateway(s)
        if d:
            order.sort(key=lambda g: g["id"] != d["id"])
        for g in order:
            if model in served_ids(g):
                return gw_provider(g["id"])
    return default_provider(s)


def set_default_provider(provider: str, secrets: dict | None = None) -> None:
    """Store the default provider (api/claude.py settings), moving `default_model` with it when
    the current one belongs to another side: a new session without a model of its own must not
    ask one credential for an id another serves."""
    s = secrets if secrets is not None else config.load_secrets()
    provider = str(provider or "").strip().lower()
    gws = gateways(s)
    if provider != "anthropic":
        gid = _gw_id(provider) or (next(iter(gws)) if provider == GATEWAY and gws else "")
        if not gid or gid not in gws:
            return
        provider = gw_provider(gid)
    cfg = config.load_config()
    cur = str(cfg.get("default_model") or "").strip()
    change: dict[str, Any] = {"claude_provider": provider}
    if provider != "anthropic":
        gw = gws[_gw_id(provider)]
        if not cur or cur not in served_ids(gw):
            dm = str(gw["models"].get("sonnet") or "") or str(PRESETS[gw["provider"]]["default_model"])
            if dm:
                change["default_model"] = dm
    elif not cur or not _CLAUDE_ID.fullmatch(cur):
        from . import claude_models
        change["default_model"] = claude_models.latest_of("sonnet")
    config.update_config(**change)


def describe(secrets: dict | None = None) -> str:
    """The auth_method() detail line of the default gateway."""
    gw = default_gateway(secrets)
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


def test_model(gw: dict | None = None, provider: str | None = None) -> str | None:
    """A cheap model id the gateway serves, for the Settings -> Claude test call. None when the
    test runs on Anthropic (the caller falls back to Claude Code's own "haiku")."""
    if provider is None:
        provider = default_provider()
    if not is_gateway(provider):
        return None
    gw = gw if gw is not None else get(provider)
    if not gw:
        return None
    return gw["models"].get("test") or gw["models"].get("haiku") or gw["models"].get("sonnet") \
        or str(config.load_config().get("default_model") or "").strip() or None


def small_model(role: str = "haiku", provider: str | None = None) -> str:
    """The model of a one-line background call (session titles, team summaries): the gateway's id
    for that role when the call runs on the gateway, else Claude Code's cheapest known Anthropic
    model. provider None = the default provider."""
    if provider is None:
        provider = default_provider()
    if is_gateway(provider):
        gw = get(provider)
        if gw:
            m = gw["models"].get(role) or gw["models"].get("haiku") or gw["models"].get("sonnet")
            if m:
                return m
            dm = str(config.load_config().get("default_model") or "").strip()
            if dm:
                return dm
    return ANTHROPIC_SMALL


def model_ids(gw: dict | None = None) -> list[str]:
    """Gateway ids for the model picker: the pinned roles, the default model (only for the gateway
    it belongs to) and the preset list."""
    gw = gw if gw is not None else configured()
    if not gw:
        return []
    dm = str(config.load_config().get("default_model") or "").strip()
    d = default_gateway()
    default = dm if dm and not _CLAUDE_ID.fullmatch(dm) and d and d["id"] == gw["id"] else ""
    out = list(dict.fromkeys([gw["models"].get(r) for r in ("sonnet", "opus", "haiku")]
                             + [default] + [str(x) for x in gw["extra"]]))
    return [m for m in out if m]


def served_ids(gw: dict | None = None) -> set[str]:
    """The ids one gateway really serves - the provider inference (provider_of, the picker's
    tags): the pinned roles and the preset list, plus the default model only when it is not an
    Anthropic id and belongs to this gateway (the default one)."""
    gw = gw if gw is not None else configured()
    if not gw:
        return set()
    ids = {str(gw["models"].get(r) or "") for r in ROLES} | {str(x) for x in gw["extra"]}
    dm = str(config.load_config().get("default_model") or "").strip()
    d = default_gateway()
    if dm and not _CLAUDE_ID.fullmatch(dm) and d and d["id"] == gw["id"]:
        ids.add(dm)
    return {i for i in ids if i}


def auth_env(out: dict[str, str], secrets: dict | None = None, provider: str | None = None) -> dict[str, str]:
    """The gateway variables of a Claude process, merged into `out` (spawn.auth_env) - when this
    turn runs on a gateway. `provider` ("gateway:<id>" | "anthropic") selects the side, None = the
    default one: an Anthropic turn gets no gateway variables at all (spawn.auth_env adds the
    Anthropic ones), a gateway turn gets no CLAUDE_CODE_OAUTH_TOKEN and no real ANTHROPIC_API_KEY
    ("" keeps Claude Code from falling back to api.anthropic.com)."""
    if provider is None:
        provider = default_provider(secrets)
    if not is_gateway(provider):
        return out
    gw = get(provider, secrets)
    if not gw:
        return out
    out["ANTHROPIC_BASE_URL"] = gw["base_url"]
    if gw["token"]:
        out["ANTHROPIC_AUTH_TOKEN"] = gw["token"]
    out["ANTHROPIC_API_KEY"] = ""
    for role, model in role_models(gw).items():
        out[f"ANTHROPIC_DEFAULT_{role.upper()}_MODEL"] = model
    return out


def clear(gw_id: str | None = None) -> None:
    """Remove one gateway (`gw_id`) or every one (None): its token, its config entry, and a default
    provider pointing at it (which falls back to the other credential)."""
    s = config.load_secrets()
    gws = gateways(s)
    ids = [gw_id] if gw_id else list(gws)
    if gw_id and gw_id not in gws:
        return
    cfg = config.load_config()
    stored = _stored()
    for gid in ids:
        stored.pop(gid, None)
    if stored:
        cfg["gateways"] = stored
    else:
        cfg.pop("gateways", None)
    leg = _legacy(s)
    if leg and leg[0] in ids:
        cfg.pop("gateway", None)                    # the single-gateway shape of older installs
    if str(cfg.get("claude_provider") or "").strip().lower() == GATEWAY or \
            _gw_id(str(cfg.get("claude_provider") or "")) in ids:
        remaining = [g for g in gws if g not in ids]
        if remaining:
            cfg["claude_provider"] = gw_provider(remaining[0])
        else:
            cfg.pop("claude_provider", None)
    config.save_config(cfg)
    sec: dict[str, Any] = {_KEY + gid: None for gid in ids}
    if leg and leg[0] in ids:
        sec.update(anthropic_base_url=None, anthropic_auth_token=None)
    config.set_secrets(sec)


def connect(provider: str, base_url: str | None, value: str) -> dict[str, Any]:
    """Store (or replace) one gateway connection (api/claude.py set_auth): the preset id, the
    bearer key, the pinned models, and a default model when none is set yet. The other gateways
    and the Anthropic credentials are kept: each session picks its side with its model. Raises
    ValueError when the provider or the base URL is not acceptable."""
    provider = str(provider or "custom").strip().lower()
    if provider not in PRESETS:
        raise ValueError("unknown gateway provider")
    preset = PRESETS[provider]
    base = valid_base_url(base_url or preset["base_url"])
    if not base:
        raise ValueError("the gateway address must be an https URL (http only on a private address)")
    s = config.load_secrets()
    stored = _stored()
    leg = _legacy(s)
    for gid, g in gateways(s).items():              # carry an older single gateway into the new shape
        stored.setdefault(gid, {"provider": g["provider"], "base_url": g["base_url"], "models": g["models"]})
    stored[provider] = {"provider": provider, "base_url": base, "models": dict(preset["models"])}
    cfg = config.load_config()
    cfg["gateways"] = stored
    cfg.pop("gateway", None)
    if not str(cfg.get("default_model") or "").strip() and preset["default_model"]:
        cfg["default_model"] = preset["default_model"]
    config.save_config(cfg)
    sec: dict[str, Any] = {_KEY + provider: value,
                           "anthropic_base_url": None, "anthropic_auth_token": None}
    if leg and leg[3] and leg[0] != provider:       # the older gateway keeps its own key
        sec[_KEY + leg[0]] = leg[3]
    config.set_secrets(sec)
    return {"provider": provider, "base_url": base, "models": dict(preset["models"])}


def public(secrets: dict | None = None) -> dict | None:
    """What /api/claude/status says about the default gateway (older clients): provider, name,
    base url, pinned models."""
    gw = default_gateway(secrets)
    if not gw:
        return None
    return {"provider": gw["provider"], "name": gw["name"], "base_url": gw["base_url"],
            "models": {k: gw["models"][k] for k in ROLES}}


def public_all(secrets: dict | None = None) -> list[dict]:
    """Every connected gateway for Settings -> Claude and the picker: {id, provider, name,
    base_url, models, default}."""
    d = default_provider(secrets)
    return [{"id": g["id"], "provider": g["provider"], "name": g["name"], "base_url": g["base_url"],
             "models": {k: g["models"][k] for k in ROLES},
             "default": d == gw_provider(g["id"])}
            for g in gateways(secrets).values()]
