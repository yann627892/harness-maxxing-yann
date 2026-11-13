"""Follow-up messages proposed above the composer (config `suggestions`).

The web UI asks for them (`GET /api/sessions/{id}/suggestions`) when a session is idle and has history; a small
model (Haiku, on the session's own provider) reads the end of the conversation and replies with a JSON array of
2-3 very short next messages. Nothing runs in the background: no call until the UI asks, one call per turn at
most (the answer is cached per session, turn count and log size), and never while a turn or a plan is waiting.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time

from .. import config
from . import spawn, store

log = logging.getLogger("maxxing.suggest")

TIMEOUT_S = 15
MAX = 3                                 # suggestions kept
MAX_CHARS = 100                         # a longer answer is not a message
CONTEXT_CHARS = 700                     # per message sent to the model
CONTEXT_MESSAGES = 6                    # last user / assistant messages
CACHE_TTL_S = 30 * 60
SYSTEM_PROMPT = ("You suggest the user's next messages in a chat with a coding agent. Reply with only a JSON "
                 "array of 2 or 3 very short follow-up requests the user could send next (8 words at most "
                 "each), in the language of the conversation. No numbering, no explanation. "
                 'Example: ["Run the tests", "Show me the diff"]')
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")

_slots = threading.BoundedSemaphore(2)  # at most two calls at a time
_cache: dict[str, tuple[tuple, float, list[str]]] = {}   # sid -> (turn key, at, suggestions)
_locks: dict[str, threading.Lock] = {}
_lock = threading.Lock()


def enabled() -> bool:
    if os.environ.get("MAXXING_SUGGESTIONS") == "0":      # tests: no model call unless asked
        return False
    return bool(config.load_config().get("suggestions", True))


def context(events: list[dict]) -> str:
    """The end of the conversation, as the model reads it (user and assistant text only)."""
    parts: list[str] = []
    for e in events:
        kind = e.get("type")
        if kind not in ("user", "assistant"):
            continue
        text = " ".join(str(e.get("text") or "").split())
        if text:
            parts.append(f"{'User' if kind == 'user' else 'Assistant'}: {text[:CONTEXT_CHARS]}")
    return "\n\n".join(parts[-CONTEXT_MESSAGES:])


def parse(answer: str) -> list[str]:
    """The model's answer as 2-3 short messages: a JSON array, else one line per item."""
    text = (answer or "").strip()
    if not text:
        return []
    items: list = []
    m = re.search(r"\[.*?\]", text, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
        except ValueError:
            data = None
        if isinstance(data, list):
            items = data
    if not items:
        items = [ln for ln in text.splitlines() if ln.strip()]
    out: list[str] = []
    for item in items:
        s = " ".join(_BULLET.sub("", str(item)).split()).strip(" \"'“”«»`")
        s = s.rstrip(" .,;:!?…")
        if not (2 <= len(s) <= MAX_CHARS) or s.lower() in {x.lower() for x in out}:
            continue
        out.append(s)
        if len(out) >= MAX:
            break
    return out


def ask(meta: dict, prompt: str) -> list[str]:
    """One small `claude -p` call (no tools, no saved thread) -> the suggestions it wrote."""
    from .. import gateway                     # the call runs on the session's own side
    provider = gateway.provider_of(meta)
    cmd = [spawn.claude_bin(), "-p", "--output-format", "json", "--model", gateway.small_model("haiku", provider),
           "--tools", "", "--no-session-persistence", "--strict-mcp-config", "--disable-slash-commands",
           "--system-prompt", SYSTEM_PROMPT]
    env = spawn.claude_env({**(meta.get("env") or {}), **spawn.effort_env(meta)}, "default", provider=provider)
    text = prompt.encode("utf-8", "replace").decode("utf-8")      # a lone surrogate cannot be written to the pipe
    try:
        r = subprocess.run(cmd, input=text, capture_output=True, text=True, timeout=TIMEOUT_S,
                           cwd=config.state_dir("claude-suggest"), env=env)
    except (OSError, ValueError, subprocess.TimeoutExpired) as e:
        log.info("suggestion call failed: %s", e)
        return []
    try:
        data = json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {}
    except ValueError:
        data = {}
    if r.returncode != 0 or not isinstance(data, dict) or data.get("is_error"):
        log.info("suggestion call failed: exit %s", r.returncode)
        return []
    return parse(str(data.get("result") or ""))


def _lock_for(sid: str) -> threading.Lock:
    with _lock:
        return _locks.setdefault(sid, threading.Lock())


def _cached(sid: str, key: tuple) -> list[str] | None:
    with _lock:
        hit = _cache.get(sid)
        if hit and hit[0] == key and time.time() - hit[1] < CACHE_TTL_S:
            return list(hit[2])
        if hit:
            _cache.pop(sid, None)
    return None


def _remember(sid: str, key: tuple, out: list[str]) -> None:
    now = time.time()
    with _lock:
        for k, v in list(_cache.items()):
            if now - v[1] >= CACHE_TTL_S:
                _cache.pop(k, None)
        _cache[sid] = (key, now, list(out))
        while len(_cache) > 200:                       # a long-lived server: keep the newest
            _cache.pop(min(_cache, key=lambda k: _cache[k][1]), None)
        while len(_locks) > 500:
            _locks.pop(next(iter(_locks)), None)


def suggestions(sid: str) -> list[str]:
    """The follow-up messages of an idle session; [] while it runs, has no history, or the switch is off."""
    m = store.read_cached(sid) or {}
    if not m or not enabled() or m.get("state") == "running" or m.get("archived"):
        return []
    if int(m.get("turns") or 0) < 1 or m.get("queue") or m.get("plan_ready"):
        return []
    key = (int(m.get("turns") or 0), store.log_size(sid))
    hit = _cached(sid, key)
    if hit is not None:
        return hit
    with _lock_for(sid):
        hit = _cached(sid, key)                        # another tab asked meanwhile
        if hit is not None:
            return hit
        from . import service
        page = service.read_back(sid, None, 60)
        prompt = context(page.get("events") or [])
        if not prompt:
            return []
        with _slots:
            out = ask(m, f"Conversation so far:\n\n{prompt}\n\n"
                         "Give 2 or 3 short follow-up messages the user could send next, as a JSON array of strings.")
        _remember(sid, key, out)
        return out
