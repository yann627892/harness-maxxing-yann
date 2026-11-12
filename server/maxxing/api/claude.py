"""Claude CLI status, authentication and defaults. See docs/dev/API.md#claude."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import re
import subprocess
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .. import claude_auth, claude_models, claude_update, config, gateway
from ..security import require_auth
from ..sessions import service, spawn

router = APIRouter(prefix="/api/claude", tags=["claude"], dependencies=[Depends(require_auth)])

TEST_MODEL = "haiku"          # Claude Code's name for its newest Haiku; a gateway uses its own ids
TEST_PROMPT = "Reply with the single word: ok"
TEST_TIMEOUT_S = 90
PERMISSION_MODES = ("bypass", "acceptEdits", "default", "ask")
SECRET_FOR = {"oauth_token": "claude_oauth_token", "api_key": "anthropic_api_key"}
_version_cache: dict[str, tuple[float, str | None]] = {}


# ---- detection -----------------------------------------------------------------------------
def _resolved_bin() -> str | None:
    b = spawn.claude_bin()
    if os.path.isabs(b):
        return b if os.access(b, os.X_OK) else None
    return shutil.which(b, path=spawn.base_env()["PATH"])


def claude_version() -> str | None:
    """`claude --version` (cached per binary mtime)."""
    b = _resolved_bin()
    if not b:
        return None
    try:
        key = f"{b}:{os.stat(b).st_mtime}"
    except OSError:
        return None
    if key in _version_cache:
        return _version_cache[key][1]
    try:
        r = subprocess.run([b, "--version"], capture_output=True, text=True, timeout=20, env=spawn.base_env())
        v = (r.stdout.strip().split() or [None])[0] if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        v = None
    _version_cache[key] = (time.time(), v)
    return v


def auth_method() -> tuple[str | None, str]:
    return claude_auth.auth_method()


def auth_status() -> dict:
    """{ok, method, detail, checked_at, error} - see maxxing/claude_auth.py (real-call based)."""
    return claude_auth.status()


# ---- test call -----------------------------------------------------------------------------
def run_test(model: str | None = None, source: str = "test", timeout_s: int = TEST_TIMEOUT_S,
             provider: str | None = None) -> dict:
    """Real one-line call through the CLI -> {ok, model, ms, error?} (also remembered).
    model None: the gateway's cheap id, else the Anthropic fallback. provider: the side to test
    ("gateway:<id>" | "anthropic"; bare "gateway" = the default gateway); None = the default."""
    if model is None:
        model = gateway.test_model(provider=provider) or TEST_MODEL
    t0 = time.time()
    out: dict = {"ok": False, "model": model, "ms": 0}
    b = _resolved_bin()
    if not b:
        out["error"] = "Claude Code is not installed"
        return _remember(out, source, provider)
    cwd = config.state_dir("claude-test")
    try:
        r = subprocess.run([b, "-p", "--output-format", "json", "--model", model], input=TEST_PROMPT,
                           capture_output=True, text=True, timeout=timeout_s, cwd=cwd,
                           env=spawn.claude_env(None, "default", provider=provider))
    except subprocess.TimeoutExpired:
        out.update(ms=int((time.time() - t0) * 1000), error=f"No answer within {timeout_s} s")
        return _remember(out, source, provider)
    except OSError as e:
        out.update(error=str(e))
        return _remember(out, source, provider)
    out["ms"] = int((time.time() - t0) * 1000)
    try:
        data = json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {}
    except ValueError:
        data = {}
    text = str(data.get("result") or "")
    if r.returncode == 0 and not data.get("is_error") and text.strip():
        out["ok"] = True
        mu = data.get("modelUsage")
        if isinstance(mu, dict) and mu:
            out["model"] = next(iter(mu))
    else:
        out["error"] = (text or r.stderr or r.stdout or f"exit code {r.returncode}").strip()[-400:]
    return _remember(out, source, provider)


def _remember(out: dict, source: str, provider: str | None = None) -> dict:
    """Record the outcome. A background check only flips the state on a real sign-in error
    (a timeout or a network glitch is not "signed out"); an explicit test always does."""
    err = out.get("error")
    if out["ok"]:
        claude_auth.record(True, None, source, provider)
    elif source == "test" or claude_auth.detect(err):
        claude_auth.record(False, claude_auth.detect(err) or err, source, provider)
    return out


# ---- background verification -----------------------------------------------------------------
CHECK_POLL_S = 600
CHECK_TIMEOUT_S = 60
_task: asyncio.Task | None = None
_last_attempt = 0.0


def check_now() -> dict | None:
    """One background verification if due (no real call in the last 6 h, none attempted here
    in the last 6 h); returns the test result or None when nothing was done."""
    global _last_attempt
    if time.time() - _last_attempt < claude_auth.CHECK_EVERY_S or not claude_auth.needs_check():
        return None
    if not _resolved_bin():
        return None
    _last_attempt = time.time()
    return run_test(source="check", timeout_s=CHECK_TIMEOUT_S)


async def _checker() -> None:
    await asyncio.sleep(float(os.environ.get("MAXXING_AUTH_CHECK_DELAY", "5")))
    while True:
        try:
            await asyncio.to_thread(check_now)
        except Exception:
            pass
        await asyncio.sleep(CHECK_POLL_S)


async def startup() -> None:
    global _task
    if os.environ.get("MAXXING_AUTH_CHECK", "1") != "0":
        _task = asyncio.create_task(_checker())
    claude_update.start()
    claude_models.start()


async def shutdown() -> None:
    global _task
    if _task:
        _task.cancel()
        _task = None
    claude_update.stop()
    claude_models.stop()


# ---- endpoints -----------------------------------------------------------------------------
@router.get("/status")
async def status():
    cfg = config.load_config()
    version = await asyncio.to_thread(claude_version)
    anth = await asyncio.to_thread(claude_auth.anthropic_method)
    return {"installed": bool(await asyncio.to_thread(_resolved_bin)), "version": version,
            "auth": await asyncio.to_thread(auth_status),
            "gateway": gateway.public(),
            "gateways": gateway.public_all(),
            "anthropic": {"connected": anth[0] is not None, "method": anth[0], "detail": anth[1]},
            "default_provider": gateway.default_provider(),
            "default_model": cfg.get("default_model") or "",
            "permission_mode": cfg.get("permission_mode") or "bypass",
            "auto_compact_tokens": config.int_setting(cfg, "auto_compact_tokens"),
            "idle_compact_tokens": config.int_setting(cfg, "idle_compact_tokens"),
            "idle_compact_after_s": config.int_setting(cfg, "idle_compact_after_s"),
            "smart_resume_tokens": config.int_setting(cfg, "smart_resume_tokens"),
            "protect_secrets": bool(cfg.get("protect_secrets", True)),
            "short_titles": bool(cfg.get("short_titles", True)),
            "short_titles_model": bool(cfg.get("short_titles_model", False)),
            "auto_update": await asyncio.to_thread(claude_update.public),
            "models": await asyncio.to_thread(service.models)}


class AuthIn(BaseModel):
    kind: Literal["oauth_token", "api_key", "gateway"]
    value: str
    provider: str | None = None      # gateway: zen | zen_go | openrouter | custom
    base_url: str | None = None      # gateway: the endpoint (required for custom)


@router.post("/auth")
async def set_auth(body: AuthIn):
    value = body.value.strip()
    # printable ASCII only: a NUL byte made every turn fail to start, a lone surrogate could not be stored
    if len(value) < 10 or any(c.isspace() for c in value) or not (value.isascii() and value.isprintable()):
        raise HTTPException(400, detail={"error": "invalid", "message": "This does not look like a token"})
    if body.kind == "gateway":
        try:
            gateway.connect(body.provider or "custom", body.base_url, value)
        except ValueError as e:
            raise HTTPException(400, detail={"error": "invalid", "message": str(e)}) from None
        service.forget_models_cache()            # the picker switches to the gateway's ids right away
        res = await asyncio.to_thread(run_test, provider=gateway.gw_provider(body.provider or "custom"))
        return {"ok": res["ok"], **({"error": res["error"]} if not res["ok"] else {})}
    # the Anthropic credentials: one of them at a time (both env vars set would be ambiguous), the
    # gateway connection kept - both sides can be connected at once, each session picks with its model
    for kind, name in SECRET_FOR.items():
        config.set_secret(name, value if kind == body.kind else None)
    service.forget_models_cache()                # ...and the picker gains the Anthropic ids back
    res = await asyncio.to_thread(run_test, provider="anthropic")
    return {"ok": res["ok"], **({"error": res["error"]} if not res["ok"] else {})}


@router.delete("/auth")
async def delete_auth(kind: str | None = None, id: str | None = None):
    """Remove one credential (`kind` = "gateway" | "anthropic") or both (no kind). `id` picks one
    gateway; without it, every gateway of that kind goes."""
    if kind not in (None, "gateway", "anthropic"):
        raise HTTPException(400, detail={"error": "invalid", "message": "kind: gateway or anthropic"})
    if id and kind != "gateway":
        raise HTTPException(400, detail={"error": "invalid", "message": "id: with kind=gateway only"})
    if kind in (None, "gateway"):
        gateway.clear(id)
    if kind in (None, "anthropic"):
        for name in SECRET_FOR.values():
            config.set_secret(name, None)
    if not gateway.anthropic_configured() and not gateway.gateways():
        claude_auth.forget()
    service.forget_models_cache()                 # the picker follows what remains connected
    return {"ok": True}


@router.post("/test")
async def test(provider: str | None = None):
    if provider is not None and provider != "anthropic" and \
            (not gateway.is_gateway(provider) or not gateway.get(provider)):
        raise HTTPException(400, detail={"error": "invalid", "message": "provider: anthropic or gateway:<id>"})
    return await asyncio.to_thread(run_test, provider=provider)


MODEL_RE = re.compile(r"[A-Za-z0-9._:\[\]/@-]{1,100}")


class SettingsIn(BaseModel):
    default_model: str | None = None
    permission_mode: str | None = None
    claude_provider: str | None = None    # which side new sessions and background calls use
    auto_compact_tokens: int | None = Field(None, ge=0, le=2_000_000)
    idle_compact_tokens: int | None = Field(None, ge=0, le=2_000_000)
    idle_compact_after_s: int | None = Field(None, ge=300, le=7 * 86400)
    smart_resume_tokens: int | None = Field(None, ge=0, le=2_000_000)
    protect_secrets: bool | None = None
    short_titles: bool | None = None
    short_titles_model: bool | None = None
    claude_auto_update: bool | None = None


@router.put("/settings")
async def settings(body: SettingsIn):
    changes: dict = {}
    if body.default_model is not None:
        model = body.default_model.strip()
        if model and not MODEL_RE.fullmatch(model):     # a NUL or a newline here broke every new session
            raise HTTPException(400, detail={"error": "invalid",
                                             "message": "default_model: a model id such as claude-opus-5-5"})
        changes["default_model"] = model
    if body.permission_mode is not None:
        if body.permission_mode not in PERMISSION_MODES:
            raise HTTPException(400, detail={"error": "invalid",
                                             "message": "permission_mode: bypass, acceptEdits, default or ask"})
        changes["permission_mode"] = body.permission_mode
    if body.claude_provider is not None:
        p = body.claude_provider.strip().lower()
        if p != "anthropic" and (not gateway.is_gateway(p) or not gateway.get(p)):
            raise HTTPException(400, detail={"error": "invalid",
                                             "message": "claude_provider: anthropic or gateway:<id>"})
        gateway.set_default_provider(p)                     # also moves default_model with it
        service.forget_models_cache()                       # the picker's "default" line follows
    if body.auto_compact_tokens is not None:
        changes["auto_compact_tokens"] = body.auto_compact_tokens
    if body.idle_compact_tokens is not None:
        changes["idle_compact_tokens"] = body.idle_compact_tokens
    if body.idle_compact_after_s is not None:
        changes["idle_compact_after_s"] = body.idle_compact_after_s
    if body.smart_resume_tokens is not None:
        changes["smart_resume_tokens"] = body.smart_resume_tokens
    if body.protect_secrets is not None:
        changes["protect_secrets"] = body.protect_secrets
    if body.short_titles is not None:
        changes["short_titles"] = body.short_titles
    if body.short_titles_model is not None:
        changes["short_titles_model"] = body.short_titles_model
    if body.claude_auto_update is not None:
        changes["claude_auto_update"] = body.claude_auto_update
    if changes:
        config.update_config(**changes)
    return {"ok": True}
