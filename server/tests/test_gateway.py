"""The LLM gateway (OpenCode Zen / Go, OpenRouter, custom) in place of the Anthropic credentials.

Connection through the API, credential cross-clearing, the env a `claude` process gets, the model
picker and the base-URL rules; the test call goes through the fake CLI like every other auth kind.
"""
from __future__ import annotations

import json

import pytest

from conftest import wait_idle
from maxxing import config, gateway
from maxxing.sessions import service, spawn

ZEN_KEY = "oc_sk-0123456789abcdef"


@pytest.fixture(autouse=True)
def _no_home_credentials(tmp_path, monkeypatch):
    """Never the real ~/.claude/.credentials.json of the machine running the tests."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


@pytest.fixture(autouse=True)
def _fresh_models_cache():
    """service.models() is cached for 60 s across tests; a gateway list must not leak into
    another test (and a plain one must not leak into these)."""
    service._models_cache = (0, [])
    yield
    service._models_cache = (0, [])


def _calls(mx: dict) -> list[dict]:
    f = mx["fake_dir"] / "calls.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []


def test_base_url_rules():
    ok = {
        "https://opencode.ai/inference/anthropic": "https://opencode.ai/inference/anthropic",
        "https://openrouter.ai/api": "https://openrouter.ai/api",
        "https://GW.Example.com/anthropic/": "https://gw.example.com/anthropic",   # trailing / off
        "http://127.0.0.1:8082": "http://127.0.0.1:8082",                           # private may be http
        "http://192.168.1.4/api": "http://192.168.1.4/api",
    }
    for url, want in ok.items():
        assert gateway.valid_base_url(url) == want
    for bad in ("", "opencode.ai", "ftp://opencode.ai", "http://opencode.ai/api",    # http, public host
                "https://", "https://a b.com", "https://example.com/#frag", "x" * 301,
                "https://exa\nmple.com", "javascript://example.com"):
        assert gateway.valid_base_url(bad) is None
    assert gateway.valid_base_url("  https://openrouter.ai/api  ") == "https://openrouter.ai/api"  # paste
    assert gateway.valid_base_url("http://[::1]:8080/api") == "http://[::1]:8080/api"
    assert gateway.valid_base_url("https://gw.example.com:notaport/api") is None
    assert gateway.valid_base_url("http://203.0.113.7/api") is None              # public, not private


def test_connect_zen_and_status(client, mx):
    r = client.post("/api/claude/auth", json={"kind": "gateway", "value": ZEN_KEY, "provider": "zen"})
    assert r.json() == {"ok": True}
    assert config.get_secret("anthropic_base_url") == "https://opencode.ai/inference/anthropic"
    assert config.get_secret("anthropic_auth_token") == ZEN_KEY
    st = client.get("/api/claude/status").json()
    assert st["auth"]["method"] == "gateway" and st["auth"]["ok"]
    assert st["gateway"] == {"provider": "zen", "name": "OpenCode Zen",
                             "base_url": "https://opencode.ai/inference/anthropic",
                             "models": {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5",
                                        "haiku": "claude-haiku-5-5", "test": "claude-haiku-5-5"}}
    assert st["default_model"] == "claude-sonnet-5-5"            # set by the preset when none yet
    # the test call asks the gateway's cheap model, not Anthropic's
    t = client.post("/api/claude/test").json()
    assert t["ok"] and t["model"] == "claude-haiku-5-5"


def test_gateway_replaces_anthropic_credentials(client, mx):
    client.post("/api/claude/auth", json={"kind": "api_key", "value": "sk-ant-api-0123456789"})
    assert config.get_secret("anthropic_api_key")
    client.post("/api/claude/auth", json={"kind": "gateway", "value": ZEN_KEY, "provider": "openrouter"})
    assert config.get_secret("anthropic_api_key") in (None, "")   # one credential at a time
    assert config.get_secret("anthropic_base_url") == "https://openrouter.ai/api"
    # ...and an Anthropic credential takes over from the gateway
    client.post("/api/claude/auth", json={"kind": "api_key", "value": "sk-ant-api-0123456789"})
    assert config.get_secret("anthropic_base_url") in (None, "") and config.get_secret("anthropic_api_key")
    assert client.get("/api/claude/status").json()["gateway"] is None
    st = client.get("/api/claude/status").json()["auth"]
    assert st["method"] == "api_key" and st["ok"]


def test_delete_clears_the_gateway(client, mx):
    client.post("/api/claude/auth", json={"kind": "gateway", "value": ZEN_KEY, "provider": "zen"})
    assert client.delete("/api/claude/auth").json() == {"ok": True}
    assert config.get_secret("anthropic_base_url") in (None, "")
    assert config.load_config().get("gateway") in (None, "", {})
    assert client.get("/api/claude/status").json()["auth"]["method"] is None


def test_custom_provider_needs_an_address(client, mx):
    r = client.post("/api/claude/auth", json={"kind": "gateway", "value": ZEN_KEY, "provider": "custom"})
    assert r.status_code == 400
    r = client.post("/api/claude/auth", json={"kind": "gateway", "value": ZEN_KEY, "provider": "custom",
                                             "base_url": "http://gw.example.com"})   # http, public
    assert r.status_code == 400
    r = client.post("/api/claude/auth", json={"kind": "gateway", "value": ZEN_KEY, "provider": "custom",
                                             "base_url": "https://gw.example.com/api/"})
    assert r.json() == {"ok": True}
    st = client.get("/api/claude/status").json()
    assert st["gateway"]["base_url"] == "https://gw.example.com/api" and st["gateway"]["name"] == "Custom gateway"
    assert st["auth"]["ok"] is False or st["auth"]["ok"]                # the fake CLI answered


def test_a_bad_provider_is_refused(client, mx):
    r = client.post("/api/claude/auth", json={"kind": "gateway", "value": ZEN_KEY, "provider": "nope"})
    assert r.status_code == 400


def test_turn_env_routes_through_the_gateway(client, topic, mx):
    client.post("/api/claude/auth", json={"kind": "gateway", "value": ZEN_KEY, "provider": "zen"})
    s = client.post("/api/sessions", json={"path": topic, "message": "hello NOTOOL"}).json()
    assert s["state"] == "running"
    wait_idle(client, s["id"])
    call = next(c for c in _calls(mx) if "hello NOTOOL" in c["prompt"])
    env = call["env"]
    assert env["ANTHROPIC_BASE_URL"] == "https://opencode.ai/inference/anthropic"
    assert env["ANTHROPIC_AUTH_TOKEN"] == ZEN_KEY
    assert env["ANTHROPIC_API_KEY"] == ""                              # never the x-api-key header
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] is None                     # no leftover Anthropic token
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "claude-haiku-5-5"
    assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "claude-sonnet-5-5"


def test_model_picker_lists_gateway_ids(client, topic, mx):
    ids = [m["id"] for m in client.get("/api/claude/status").json()["models"]]
    assert "claude-opus-5-5" in ids                                   # the Anthropic built-ins
    client.post("/api/claude/auth", json={"kind": "gateway", "value": ZEN_KEY, "provider": "openrouter"})
    service._models_cache = (0, [])                                  # models() is cached 60 s
    models = client.get("/api/claude/status").json()["models"]
    ids = [m["id"] for m in models]
    assert "anthropic/claude-sonnet-5.5" in ids and "anthropic/claude-haiku-5.5" in ids
    assert "claude-opus-5-5" not in ids                               # gateway ids replace them
    label = next(m for m in models if m["id"] == "anthropic/claude-opus-5.5")["label"]
    assert label == "Opus 5.5"                                        # the provider part is not shown


def test_background_models_follow_the_gateway(client, topic, mx):
    from maxxing.sessions import titles
    from maxxing import team
    client.post("/api/claude/auth", json={"kind": "gateway", "value": ZEN_KEY, "provider": "zen"})
    assert titles._model() == "claude-haiku-5-5"                      # session titles: the cheap id
    assert team._summary_model() == "claude-sonnet-5-5"               # team summaries: sonnet
    client.delete("/api/claude/auth")
    assert titles._model() == titles.MODEL                            # back to Anthropic's cheapest


def test_role_pins_fall_back_to_the_default_model(client, mx):
    client.post("/api/claude/auth", json={"kind": "gateway", "value": ZEN_KEY, "provider": "zen"})
    assert gateway.role_models() == {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5",
                                    "haiku": "claude-haiku-5-5"}    # the preset ids, pinned
    # a custom gateway pins nothing: every role falls back to the default model
    client.post("/api/claude/auth", json={"kind": "gateway", "value": ZEN_KEY, "provider": "custom",
                                          "base_url": "https://gw.example.com/api"})
    config.update_config(default_model="")                          # the zen preset set one
    assert gateway.role_models() == {}                               # no pin and no default model yet
    assert "ANTHROPIC_DEFAULT_HAIKU_MODEL" not in spawn.auth_env()
    config.update_config(default_model="claude-sonnet-5-5")
    assert gateway.role_models() == {"opus": "claude-sonnet-5-5", "sonnet": "claude-sonnet-5-5",
                                     "haiku": "claude-sonnet-5-5"}  # the default model everywhere
    assert gateway.test_model() == "claude-sonnet-5-5"               # sonnet/haiku empty: the default
    cfg = config.load_config()
    cfg["gateway"]["models"] = {"opus": "", "sonnet": "", "haiku": "", "test": "gw-haiku-test"}
    config.save_config(cfg)
    assert gateway.test_model() == "gw-haiku-test"                   # the test id stays editable
