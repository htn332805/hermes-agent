"""Smart model routing — the cheap "picker" behind ``smart_model_routing``.

A lightweight classifier labels an incoming request's complexity tier
(``light`` / ``standard`` / ``heavy``) and maps it to a tier-appropriate
model. This mirrors the Cursor "Auto" idea — right-size the model to the
task — while respecting Hermes' sacred per-conversation prompt cache.

The router is consulted ONLY at points where there is no cached prefix to
invalidate:

* at the start of a *fresh* session, before the first API call
  (:func:`run_conversation` gates on empty ``conversation_history``), and
* at each ``delegate_task`` boundary, where subagents get fresh context.

It never swaps the main model mid-conversation — that is ``/model``'s job
and it deliberately resets the cache.

Everything here fails open: a broken/slow/misconfigured classifier must
never wedge a turn. On any failure the caller stays on the current model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Ordered cheapest/smallest → most capable. Order is load-bearing: the
# ``min_tier`` floor and tier comparisons rely on it.
TIERS: Tuple[str, ...] = ("light", "standard", "heavy")

_CLASSIFIER_SYSTEM_PROMPT = (
    "You are a routing classifier for an autonomous AI coding agent. Read the "
    "user's request and label how much model capability it needs, as exactly "
    "one of these tiers:\n"
    "- light: trivial or quick — simple questions, tiny edits, lookups, "
    "formatting, one-line answers.\n"
    "- standard: ordinary coding and analysis — implement a function, explain "
    "code, write a normal test, routine debugging.\n"
    "- heavy: hard or sprawling — multi-file refactors, architecture/design, "
    "subtle debugging, deep multi-step reasoning, security-sensitive work.\n"
    "Bias toward the HIGHER tier when unsure; quality matters more than saving "
    "a little money. Respond with ONLY the single tier word, nothing else."
)

# Cap the message we send to the classifier — the opening request can be huge
# (pasted logs, files). The first ~4k chars carry the intent.
_MAX_CLASSIFY_CHARS = 4000


@dataclass
class RoutingDecision:
    """A resolved decision to run on a specific model.

    ``base_url`` / ``api_key`` / ``api_mode`` are resolved credentials ready
    to hand to ``AIAgent.switch_model`` (session routing) or to
    ``_build_child_agent`` overrides (delegation routing).
    """

    tier: str
    provider: str
    model: str
    base_url: Optional[str]
    api_key: Optional[str]
    api_mode: Optional[str]
    reason: str = ""


def get_routing_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the ``smart_model_routing`` config dict (never None)."""
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception as exc:  # noqa: BLE001
            logger.debug("model_router: load_config failed: %s", exc)
            return {}
    cfg = config.get("smart_model_routing") if isinstance(config, dict) else None
    return cfg if isinstance(cfg, dict) else {}


def is_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    return bool(get_routing_config(config).get("enabled"))


def _tier_index(tier: str) -> int:
    try:
        return TIERS.index(tier)
    except ValueError:
        return TIERS.index("standard")


def _apply_min_tier_floor(tier: str, routing_cfg: Dict[str, Any]) -> str:
    """Bump ``tier`` up to ``min_tier`` when a floor is configured."""
    floor = str(routing_cfg.get("min_tier") or "").strip().lower()
    if floor in TIERS and _tier_index(tier) < _tier_index(floor):
        return floor
    return tier


def _parse_tier(raw: str, default_tier: str) -> str:
    """Extract a tier word from a classifier response. Fail-open to default."""
    text = (raw or "").strip().lower()
    if not text:
        return default_tier
    # Exact single-word answer (the happy path) or first tier word mentioned.
    for tier in TIERS:
        if tier in text:
            return tier
    return default_tier


def classify_complexity(
    message: str,
    *,
    routing_cfg: Optional[Dict[str, Any]] = None,
    timeout: float = 20.0,
) -> Tuple[str, str]:
    """Classify ``message`` into a complexity tier.

    Returns ``(tier, reason)``. Always returns a valid tier — on any failure
    it returns the configured ``default_tier`` with a diagnostic reason.
    """
    routing_cfg = routing_cfg if routing_cfg is not None else get_routing_config()
    default_tier = str(routing_cfg.get("default_tier") or "standard").strip().lower()
    if default_tier not in TIERS:
        default_tier = "standard"

    if not (message or "").strip():
        return default_tier, "empty message"

    try:
        from agent.auxiliary_client import (
            get_auxiliary_extra_body,
            get_text_auxiliary_client,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("model_router: auxiliary client import failed: %s", exc)
        return default_tier, "auxiliary client unavailable"

    try:
        client, model = get_text_auxiliary_client("routing_classifier")
    except Exception as exc:  # noqa: BLE001
        logger.debug("model_router: get_text_auxiliary_client failed: %s", exc)
        return default_tier, "auxiliary client unavailable"

    if client is None or not model:
        return default_tier, "no auxiliary client configured"

    snippet = message.strip()[:_MAX_CLASSIFY_CHARS]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": snippet},
            ],
            temperature=0,
            max_tokens=16,
            timeout=timeout,
            extra_body=get_auxiliary_extra_body() or None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.info(
            "model_router: classifier call failed (%s) — using default tier %r",
            type(exc).__name__,
            default_tier,
        )
        return default_tier, f"classifier error: {type(exc).__name__}"

    try:
        raw = resp.choices[0].message.content or ""
    except Exception:  # noqa: BLE001
        raw = ""

    tier = _parse_tier(raw, default_tier)
    logger.info("model_router: classified tier=%s (raw=%r)", tier, (raw or "")[:40])
    return tier, "classified"


def _tier_target(tier: str, routing_cfg: Dict[str, Any]) -> Tuple[str, str]:
    """Return the configured ``(provider, model)`` for a tier ('' when unset)."""
    tiers = routing_cfg.get("tiers")
    if not isinstance(tiers, dict):
        return "", ""
    entry = tiers.get(tier)
    if not isinstance(entry, dict):
        return "", ""
    provider = str(entry.get("provider") or "").strip()
    model = str(entry.get("model") or "").strip()
    return provider, model


def _resolve_tier_credentials(provider: str, model: str) -> Optional[Dict[str, Any]]:
    """Resolve full credentials for a tier's provider:model pair.

    Reuses the same runtime-provider resolver delegation uses, so a routed
    tier behaves identically to ``delegation.provider``/``model``. Returns
    None (fail-open) when the provider can't be resolved.
    """
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested=provider, target_model=model)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "model_router: cannot resolve tier provider %r (model %r): %s — "
            "staying on current model",
            provider,
            model,
            exc,
        )
        return None

    api_key = runtime.get("api_key", "")
    if not api_key:
        logger.warning(
            "model_router: tier provider %r resolved but has no API key — "
            "staying on current model",
            provider,
        )
        return None

    return {
        "provider": runtime.get("provider") or provider,
        "model": model or runtime.get("model") or "",
        "base_url": runtime.get("base_url"),
        "api_key": api_key,
        "api_mode": runtime.get("api_mode"),
    }


def route(
    message: str,
    *,
    current_model: str,
    current_provider: str,
    config: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
) -> Optional[RoutingDecision]:
    """Decide which model ``message`` should run on.

    Returns a :class:`RoutingDecision` when the request should run on a
    *different* model than the current one, or ``None`` to stay put (routing
    disabled, tier unconfigured, no-op, or any resolution failure). ``None``
    is the cache-safe outcome — the caller makes no change.
    """
    routing_cfg = get_routing_config(config)
    if not routing_cfg.get("enabled"):
        return None

    if timeout is None:
        try:
            timeout = float(
                (config or {}).get("auxiliary", {})
                .get("routing_classifier", {})
                .get("timeout", 20)
            )
        except Exception:  # noqa: BLE001
            timeout = 20.0

    tier, reason = classify_complexity(message, routing_cfg=routing_cfg, timeout=timeout)
    tier = _apply_min_tier_floor(tier, routing_cfg)

    provider, model = _tier_target(tier, routing_cfg)
    if not provider or not model:
        # Tier intentionally maps to "stay on the current/parent model".
        logger.debug("model_router: tier %s has no target — staying", tier)
        return None

    cur_provider = (current_provider or "").strip().lower()
    cur_model = (current_model or "").strip()
    if provider.strip().lower() == cur_provider and model == cur_model:
        # Already on the right model — never break the cache for a no-op.
        logger.debug("model_router: tier %s already active (%s) — no-op", tier, model)
        return None

    creds = _resolve_tier_credentials(provider, model)
    if creds is None:
        return None

    return RoutingDecision(
        tier=tier,
        provider=creds["provider"],
        model=creds["model"],
        base_url=creds["base_url"],
        api_key=creds["api_key"],
        api_mode=creds["api_mode"],
        reason=reason,
    )
