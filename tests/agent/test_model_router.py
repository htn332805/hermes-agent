"""Tests for smart model routing (agent/model_router.py + wiring).

All tests are hermetic — the classifier and credential resolution are
stubbed, so nothing hits the network. The invariants under test:

* routing is a strict no-op when disabled (the default),
* a tier that maps to the current model never triggers a switch
  (the cache-safety guarantee),
* the min_tier floor is honored,
* classification fails open to the default tier,
* explicit delegation/model pins beat routing,
* the session-start helper fires at most once and skips resumed sessions.
"""

import types

import pytest

from agent import model_router
from agent.model_router import RoutingDecision


def _cfg(**routing):
    base = {
        "enabled": True,
        "apply_to_sessions": True,
        "apply_to_delegation": True,
        "tiers": {
            "light": {"provider": "openrouter", "model": "google/gemini-3-flash-preview"},
            "standard": {"provider": "", "model": ""},
            "heavy": {"provider": "anthropic", "model": "claude-opus-4.7"},
        },
        "default_tier": "standard",
        "min_tier": "",
        "announce": True,
    }
    base.update(routing)
    return {"smart_model_routing": base}


# ── pure helpers ────────────────────────────────────────────────────────


def test_parse_tier_exact_and_embedded():
    assert model_router._parse_tier("heavy", "standard") == "heavy"
    assert model_router._parse_tier("  Light\n", "standard") == "light"
    assert model_router._parse_tier("I think this is standard work", "heavy") == "standard"


def test_parse_tier_fails_open_to_default():
    assert model_router._parse_tier("", "standard") == "standard"
    assert model_router._parse_tier("banana", "heavy") == "heavy"


def test_min_tier_floor_bumps_up():
    cfg = _cfg(min_tier="standard")["smart_model_routing"]
    assert model_router._apply_min_tier_floor("light", cfg) == "standard"
    assert model_router._apply_min_tier_floor("heavy", cfg) == "heavy"


def test_min_tier_floor_ignores_invalid():
    cfg = _cfg(min_tier="bogus")["smart_model_routing"]
    assert model_router._apply_min_tier_floor("light", cfg) == "light"


def test_tier_target_reads_config():
    cfg = _cfg()["smart_model_routing"]
    assert model_router._tier_target("light", cfg) == (
        "openrouter",
        "google/gemini-3-flash-preview",
    )
    assert model_router._tier_target("standard", cfg) == ("", "")


# ── route() behavior ──────────────────────────────────────────────────────


def test_route_disabled_is_noop():
    decision = model_router.route(
        "anything",
        current_model="gpt-5.4",
        current_provider="openrouter",
        config=_cfg(enabled=False),
    )
    assert decision is None


def test_route_tier_with_no_target_stays(monkeypatch):
    # standard tier maps to empty → stay on current model.
    monkeypatch.setattr(model_router, "classify_complexity", lambda *a, **k: ("standard", "x"))
    decision = model_router.route(
        "normal task",
        current_model="gpt-5.4",
        current_provider="openrouter",
        config=_cfg(),
    )
    assert decision is None


def test_route_noop_when_tier_matches_current(monkeypatch):
    # heavy tier resolves to the model we're already on → no switch (cache-safe).
    monkeypatch.setattr(model_router, "classify_complexity", lambda *a, **k: ("heavy", "x"))
    monkeypatch.setattr(
        model_router,
        "_resolve_tier_credentials",
        lambda p, m: {"provider": "anthropic", "model": "claude-opus-4.7",
                      "base_url": None, "api_key": "sk", "api_mode": None},
    )
    decision = model_router.route(
        "hard refactor",
        current_model="claude-opus-4.7",
        current_provider="anthropic",
        config=_cfg(),
    )
    assert decision is None


def test_route_returns_decision_on_tier_change(monkeypatch):
    monkeypatch.setattr(model_router, "classify_complexity", lambda *a, **k: ("heavy", "x"))
    monkeypatch.setattr(
        model_router,
        "_resolve_tier_credentials",
        lambda p, m: {"provider": "anthropic", "model": "claude-opus-4.7",
                      "base_url": None, "api_key": "sk", "api_mode": None},
    )
    decision = model_router.route(
        "hard refactor",
        current_model="gpt-5.4",
        current_provider="openrouter",
        config=_cfg(),
    )
    assert isinstance(decision, RoutingDecision)
    assert decision.tier == "heavy"
    assert decision.model == "claude-opus-4.7"
    assert decision.provider == "anthropic"


def test_route_fails_open_when_credentials_unresolved(monkeypatch):
    monkeypatch.setattr(model_router, "classify_complexity", lambda *a, **k: ("light", "x"))
    monkeypatch.setattr(model_router, "_resolve_tier_credentials", lambda p, m: None)
    decision = model_router.route(
        "tiny edit",
        current_model="gpt-5.4",
        current_provider="openrouter",
        config=_cfg(),
    )
    assert decision is None


def test_route_honors_min_tier(monkeypatch):
    # classifier says light, but min_tier=heavy forces heavy.
    monkeypatch.setattr(model_router, "classify_complexity", lambda *a, **k: ("light", "x"))
    captured = {}

    def _fake_resolve(provider, model):
        captured["provider"] = provider
        captured["model"] = model
        return {"provider": provider, "model": model, "base_url": None,
                "api_key": "sk", "api_mode": None}

    monkeypatch.setattr(model_router, "_resolve_tier_credentials", _fake_resolve)
    decision = model_router.route(
        "tiny edit",
        current_model="gpt-5.4",
        current_provider="openrouter",
        config=_cfg(min_tier="heavy"),
    )
    assert decision is not None
    assert decision.tier == "heavy"
    assert captured["model"] == "claude-opus-4.7"


# ── classify_complexity fail-open ─────────────────────────────────────────


def test_classify_fails_open_without_aux_client(monkeypatch):
    import agent.auxiliary_client as aux

    monkeypatch.setattr(aux, "get_text_auxiliary_client", lambda task: (None, None))
    tier, reason = model_router.classify_complexity(
        "do something", routing_cfg=_cfg()["smart_model_routing"]
    )
    assert tier == "standard"
    assert "no auxiliary client" in reason


def test_classify_empty_message_returns_default():
    tier, reason = model_router.classify_complexity(
        "   ", routing_cfg=_cfg(default_tier="heavy")["smart_model_routing"]
    )
    assert tier == "heavy"


# ── session-start wiring (_maybe_apply_session_routing) ───────────────────


class _FakeAgent:
    def __init__(self):
        self.model = "gpt-5.4"
        self.provider = "openrouter"
        self.quiet_mode = True
        self.switched = None

    def switch_model(self, **kwargs):
        self.switched = kwargs
        self.model = kwargs["new_model"]
        self.provider = kwargs["new_provider"]


def test_session_routing_skips_resumed_session(monkeypatch):
    from agent import conversation_loop

    agent = _FakeAgent()
    # Non-empty history → must not classify or switch, but must mark applied.
    conversation_loop._maybe_apply_session_routing(agent, "hi", [{"role": "user", "content": "x"}])
    assert agent.switched is None
    assert agent._smart_routing_applied is True


def test_session_routing_applies_once_and_switches(monkeypatch):
    from agent import conversation_loop

    agent = _FakeAgent()
    monkeypatch.setattr(model_router, "get_routing_config", lambda config=None: _cfg()["smart_model_routing"])
    monkeypatch.setattr(
        model_router,
        "route",
        lambda *a, **k: RoutingDecision(
            tier="heavy", provider="anthropic", model="claude-opus-4.7",
            base_url=None, api_key="sk", api_mode=None, reason="classified",
        ),
    )
    conversation_loop._maybe_apply_session_routing(agent, "hard task", None)
    assert agent.switched is not None
    assert agent.model == "claude-opus-4.7"
    assert agent._smart_routing_applied is True

    # Second call must be a no-op (flag already set).
    agent.switched = None
    conversation_loop._maybe_apply_session_routing(agent, "another", None)
    assert agent.switched is None


# ── delegation wiring (_route_task_creds) ─────────────────────────────────


def test_delegation_routing_respects_explicit_model():
    from tools import delegate_tool

    base = {"model": "pinned/model", "provider": "openrouter", "base_url": None,
            "api_key": None, "api_mode": None}
    parent = types.SimpleNamespace(model="gpt-5.4", provider="openrouter")
    out = delegate_tool._route_task_creds(base, "anything", parent)
    assert out is base  # unchanged — explicit delegation.model wins


def test_delegation_routing_sets_model_when_unpinned(monkeypatch):
    from tools import delegate_tool

    monkeypatch.setattr(model_router, "get_routing_config", lambda config=None: _cfg()["smart_model_routing"])
    monkeypatch.setattr(
        model_router,
        "route",
        lambda *a, **k: RoutingDecision(
            tier="light", provider="openrouter", model="google/gemini-3-flash-preview",
            base_url=None, api_key="sk", api_mode=None, reason="classified",
        ),
    )
    base = {"model": None, "provider": None, "base_url": None, "api_key": None, "api_mode": None}
    parent = types.SimpleNamespace(model="claude-opus-4.7", provider="anthropic")
    out = delegate_tool._route_task_creds(base, "tiny task", parent)
    assert out["model"] == "google/gemini-3-flash-preview"
    assert out["provider"] == "openrouter"


def test_delegation_routing_noop_returns_base(monkeypatch):
    from tools import delegate_tool

    monkeypatch.setattr(model_router, "get_routing_config", lambda config=None: _cfg()["smart_model_routing"])
    monkeypatch.setattr(model_router, "route", lambda *a, **k: None)
    base = {"model": None, "provider": None, "base_url": None, "api_key": None, "api_mode": None}
    parent = types.SimpleNamespace(model="claude-opus-4.7", provider="anthropic")
    out = delegate_tool._route_task_creds(base, "task", parent)
    assert out is base
