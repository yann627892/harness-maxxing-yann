"""The LLM gateway (OpenCode Zen / Go, OpenRouter, custom) beside the Anthropic credentials.

Connection through the API, both sides staying stored at once, the env a `claude` process gets on
either side, the model picker and the base-URL rules; the test call goes through the fake CLI like
every other auth kind.
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
    assert any(m["id"] == "claude-sonnet-5-5" for m in st["models"])   # the picker shows them right away


def test_gateway_models_refresh_without_waiting(client, mx):
    """Connecting a gateway refreshes the model list immediately: the picker never waits out the
    60 s cache of service.models() still holding the Anthropic ids of the previous status call."""
    st = client.get("/api/claude/status").json()                 # fills the cache with Anthropic ids
    assert any(m["id"].startswith("claude-") for m in st["models"])
    client.post("/api/claude/auth", json={"kind": "gateway", "value": ZEN_KEY, "provider": "openrouter"})
    st = client.get("/api/claude/status").json()
    ids = [m["id"] for m in st["models"]]
    assert "anthropic/claude-sonnet-5.5" in ids and "claude-sonnet-5-5" not in ids
    labels = {m["id"]: m["label"] for m in st["models"]}
    assert labels["anthropic/claude-sonnet-5.5"] == "Sonnet 5.5"
    assert labels["z-ai/glm-5.3-prime"] == "GLM 5.3 Prime"          # readable labels for other families
    assert labels["openai/gpt-6.1-terra"] == "GPT 6.1 Terra"
    client.delete("/api/claude/auth")                             # signed out: the Anthropic ids come back
    st = client.get("/api/claude/status").json()
    assert any(m["id"].startswith("claude-") and "/" not in m["id"] for m in st["models"])


def test_gateway_and_anthropic_credentials_coexist(client, mx):
    """Both sides stay stored at once: each session picks its provider with its model."""
    client.post("/api/claude/auth", json={"kind": "api_key", "value": "sk-ant-api-0123456789"})
    assert config.get_secret("anthropic_api_key")
    client.post("/api/claude/auth", json={"kind": "gateway", "value": ZEN_KEY, "provider": "openrouter"})
    assert config.get_secret("anthropic_api_key")                     # kept, not cleared
    assert config.get_secret("anthropic_base_url") == "https://openrouter.ai/api"
    st = client.get("/api/claude/status").json()
    assert st["gateway"] and st["gateway"]["provider"] == "openrouter"
    assert st["anthropic"] == {"connected": True, "method": "api_key",
                               "detail": "API key stored in Harness Maxxing"}
    assert st["default_provider"] == "gateway"                        # the gateway wins by default
    assert st["auth"]["method"] == "gateway"
    # ...and connecting an Anthropic credential again keeps the gateway
    client.post("/api/claude/auth", json={"kind": "api_key", "value": "sk-ant-api-987654321"})
    st = client.get("/api/claude/status").json()
    assert st["gateway"] and st["anthropic"]["connected"] and st["anthropic"]["method"] == "api_key"


def test_default_provider_setting_moves_the_default_model(client, mx):
    client.post("/api/claude/auth", json={"kind": "api_key", "value": "sk-ant-api-0123456789"})
    client.post("/api/claude/auth", json={"kind": "gateway", "value": ZEN_KEY, "provider": "openrouter"})
    assert config.load_config()["default_model"] == "anthropic/claude-sonnet-5.5"   # the preset set it
    r = client.put("/api/claude/settings", json={"claude_provider": "anthropic"})
    assert r.status_code == 200
    assert gateway.default_provider() == "anthropic"
    assert config.load_config()["default_model"] == "claude-sonnet-5-5"  # moved with the side
    assert client.get("/api/claude/status").json()["auth"]["method"] == "api_key"  # auth follows it
    client.put("/api/claude/settings", json={"claude_provider": "gateway"})
    assert config.load_config()["default_model"] == "anthropic/claude-sonnet-5.5"
    assert client.get("/api/claude/status").json()["auth"]["method"] == "gateway"
    r = client.put("/api/claude/settings", json={"claude_provider": "nope"})
    assert r.status_code == 400
    # a stored choice pointing at a credential that is gone falls back to the other side
    client.delete("/api/claude/auth", params={"kind": "gateway"})
    assert gateway.default_provider() == "anthropic"


def test_provider_of_follows_the_session_choice(client, mx):
    client.post("/api/claude/auth", json={"kind": "api_key", "value": "sk-ant-api-0123456789"})
    client.post("/api/claude/auth", json={"kind": "gateway", "value": ZEN_KEY, "provider": "openrouter"})
    client.put("/api/claude/settings", json={"claude_provider": "anthropic"})
    assert gateway.provider_of({"provider": "gateway"}) == "gateway"       # the stored choice wins
    assert gateway.provider_of({"provider": "anthropic"}) == "anthropic"
    assert gateway.provider_of({}) == "anthropic"                           # else the default
    assert gateway.provider_of({"model": "openai/gpt-6.1-terra"}) == "gateway"   # a gateway id
    assert gateway.provider_of({"model": "claude-sonnet-5-5"}) == "anthropic"   # an Anthropic id


def test_turn_env_follows_the_session_provider(client, topic, mx):
    client.post("/api/claude/auth", json={"kind": "api_key", "value": "sk-ant-api-0123456789"})
    client.post("/api/claude/auth", json={"kind": "gateway", "value": ZEN_KEY, "provider": "openrouter"})
    # the session picks the subscription: no gateway variables at all
    s = client.post("/api/sessions",
                    json={"path": topic, "message": "hello NOTOOL", "provider": "anthropic"}).json()
    wait_idle(client, s["id"])
    call = next(c for c in _calls(mx) if "hello NOTOOL" in c["prompt"])
    assert call["env"]["ANTHROPIC_API_KEY"] == "sk-ant-api-0123456789"
    assert call["env"]["ANTHROPIC_BASE_URL"] is None and call["env"]["ANTHROPIC_AUTH_TOKEN"] is None
    # back to the default (the gateway): the same session switches sides
    assert client.patch(f"/api/sessions/{s['id']}", json={"provider": ""}).status_code == 200
    client.post(f"/api/sessions/{s['id']}/messages", json={"text": "again NOTOOL"})
    wait_idle(client, s["id"])
    call = next(c for c in _calls(mx) if "again NOTOOL" in c["prompt"])
    assert call["env"]["ANTHROPIC_BASE_URL"] == "https://openrouter.ai/api"
    assert call["env"]["ANTHROPIC_AUTH_TOKEN"] == ZEN_KEY
    assert call["env"]["ANTHROPIC_API_KEY"] == ""


def test_a_gateway_model_runs_on_the_gateway_even_as_default_anthropic(client, topic, mx):
    """A session from before both were connected: the origin of its model id decides."""
    client.post("/api/claude/auth", json={"kind": "api_key", "value": "sk-ant-api-0123456789"})
    client.post("/api/claude/auth", json={"kind": "gateway", "value": ZEN_KEY, "provider": "openrouter"})
    client.put("/api/claude/settings", json={"claude_provider": "anthropic"})
    s = client.post("/api/sessions",
                    json={"path": topic, "message": "model NOTOOL", "model": "openai/gpt-6.1-terra"}).json()
    wait_idle(client, s["id"])
    call = next(c for c in _calls(mx) if "model NOTOOL" in c["prompt"])
    assert call["env"]["ANTHROPIC_BASE_URL"] == "https://openrouter.ai/api"
    assert call["argv"][call["argv"].index("--model") + 1] == "openai/gpt-6.1-terra"


def test_models_list_tags_each_side(client, mx):
    models = client.get("/api/claude/status").json()["models"]
    assert all(m["provider"] == "anthropic" for m in models)              # only the built-ins
    client.post("/api/claude/auth", json={"kind": "gateway", "value": ZEN_KEY, "provider": "openrouter"})
    models = client.get("/api/claude/status").json()["models"]
    assert models and all(m["provider"] == "gateway" for m in models)     # it replaces them
    client.post("/api/claude/auth", json={"kind": "api_key", "value": "sk-ant-api-0123456789"})
    models = client.get("/api/claude/status").json()["models"]           # both: the picker lists both
    tags = {m["id"]: m["provider"] for m in models}
    assert tags["anthropic/claude-sonnet-5.5"] == "gateway"
    assert tags["claude-sonnet-5-5"] == "anthropic"
    client.delete("/api/claude/auth", params={"kind": "gateway"})
    models = client.get("/api/claude/status").json()["models"]
    assert models and all(m["provider"] == "anthropic" for m in models)


def test_delete_removes_one_side(client, mx):
    client.post("/api/claude/auth", json={"kind": "api_key", "value": "sk-ant-api-0123456789"})
    client.post("/api/claude/auth", json={"kind": "gateway", "value": ZEN_KEY, "provider": "zen"})
    assert client.delete("/api/claude/auth", params={"kind": "gateway"}).json() == {"ok": True}
    st = client.get("/api/claude/status").json()
    assert st["gateway"] is None and st["anthropic"]["connected"]
    assert st["auth"]["method"] == "api_key" and st["default_provider"] == "anthropic"
    assert config.get_secret("anthropic_api_key")                        # that side stays
    client.post("/api/claude/auth", json={"kind": "gateway", "value": ZEN_KEY, "provider": "zen"})
    assert client.delete("/api/claude/auth", params={"kind": "anthropic"}).json() == {"ok": True}
    st = client.get("/api/claude/status").json()
    assert st["gateway"] and not st["anthropic"]["connected"]
    assert st["auth"]["method"] == "gateway" and st["default_provider"] == "gateway"
    assert config.get_secret("anthropic_api_key") in (None, "")
    assert client.delete("/api/claude/auth", params={"kind": "nope"}).status_code == 400
    assert client.delete("/api/claude/auth").json() == {"ok": True}       # no kind: everything
    st = client.get("/api/claude/status").json()
    assert st["auth"]["method"] is None and st["gateway"] is None


def test_anthropic_turn_drops_an_environment_gateway(client, mx, monkeypatch):
    """A gateway configured in the service environment must not leak into an Anthropic turn - but
    it stays when it is the only credential there is (that install mode keeps working)."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gw.example.com/api")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "env-key")
    client.post("/api/claude/auth", json={"kind": "api_key", "value": "sk-ant-api-0123456789"})
    env = spawn.claude_env(None, "bypass", provider="anthropic")
    assert "ANTHROPIC_BASE_URL" not in env and "ANTHROPIC_AUTH_TOKEN" not in env
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-api-0123456789"
    assert spawn.claude_env(None, "bypass", provider="gateway")["ANTHROPIC_BASE_URL"] == "https://gw.example.com/api"
    config.set_secret("anthropic_api_key", None)                         # nothing Anthropic left:
    env = spawn.claude_env(None, "bypass", provider="anthropic")          # the env gateway is it
    assert env["ANTHROPIC_BASE_URL"] == "https://gw.example.com/api"


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
