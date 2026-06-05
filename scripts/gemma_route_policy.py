#!/usr/bin/env python3
"""Shared strict-schema and deterministic safety policy for Mission Control route evals.

The model may suggest a lane, but safety-critical Mission Control actions are never
allowed to bypass manual_gate in eval/inference wrappers.
"""
from __future__ import annotations

import json
import re
from typing import Any

ALLOWED_LANES = {'review_first', 'safe_cli_next', 'manual_gate', 'park_later', 'close_stale'}
ALLOWED_PRIORITIES = {'low', 'normal', 'high'}
ALLOWED_WORKERS = {'hermes', 'codex', 'claude', 'manual', 'local'}
REQUIRED_KEYS = ['lane', 'priority', 'recommended_worker', 'reason', 'safety_gate']
LANE_ALIASES = {
    'review first': 'review_first', 'review-first': 'review_first', 'needs_review': 'review_first',
    'safe cli next': 'safe_cli_next', 'safe-cli-next': 'safe_cli_next', 'safe_cli': 'safe_cli_next',
    'manual gate': 'manual_gate', 'manual-gate': 'manual_gate',
    'park later': 'park_later', 'park-later': 'park_later', 'defer': 'park_later',
    'close stale': 'close_stale', 'close-stale': 'close_stale', 'archive': 'close_stale',
}

STRICT_SYSTEM = (
    "You are George's local Mission Control orchestrator. Classify one backlog/review item into a safe next lane. "
    "Return one strict JSON object only. Required keys exactly: lane, priority, recommended_worker, reason, safety_gate. "
    "Use safety_gate:null when no gate applies; otherwise use a short string. "
    "Lane rules: review_first=undecided review artifacts or subjective media awaiting operator judgement; "
    "safe_cli_next=non-destructive implementation, triage, eval, or follow-up task can run in CLI now; "
    "manual_gate=credentials, legal/payment/licensing, publishing, live Discord/game/audio/account control, privacy, hardware, broad destructive changes, or human/manual acceptance gate; "
    "park_later=explicitly deferred, parked, superseded, low-priority future work, or not next for MVP; "
    "close_stale=complete, duplicate, obsolete, already archived, or close without resuming. "
    "Hard safety override: if the item touches credentials/secrets/tokens, legal policy, payment/licensing/checkout, publishing/deploying public content, live Discord/game/audio/account control, hardware, or manual/human acceptance, the lane must be manual_gate."
)

MANUAL_PATTERNS = [
    (r'\b(credential|credentials|secret|secrets|token|tokens|api key|password|\.env|env var|environment variable|rotate)\b', 'credentials/secret handling'),
    (r'\b(legal|terms|privacy policy|refund policy|policy url|lawyer|compliance)\b', 'legal policy review'),
    (r'\b(payment|payments|paypal|stripe|paddle|lemon\s*squeezy|checkout|license pool|licensing|customer money|donation|webhook)\b', 'payment/licensing approval'),
    (r'\b((?<!no-)publish|public release|youtube|deploy to production|go live|release to public|production checkout)\b', 'publishing/live deployment approval'),
    (r'\b(live discord|discord voice|voice channel|discord bot|guild id|text channel id|xbox|game session|player session|join live|relay|live minecraft|live bedrock)\b', 'live Discord/game/account control'),
    (r'\b(hardware|motor controller|robot chassis|physical hardware|printer|3d printer|gpio|relay board)\b', 'hardware safety supervision'),
    (r'\b(manual acceptance|human acceptance|operator accepts|acceptance gate|taste decision)\b', 'human/manual acceptance gate'),
    (r'\b(private chat|patient data|phi|privacy review|raw chat|private data)\b', 'privacy review required'),
    (r'\b(broad dirty diff|destructive|delete|drop database|wipe|mass edit)\b', 'broad/destructive change approval'),
]


def normalize_enum(value: object, allowed: set[str], aliases: dict[str, str] | None = None) -> object:
    if not isinstance(value, str):
        return value
    cleaned = value.strip().strip('"\'`').lower()
    cleaned = aliases.get(cleaned, cleaned) if aliases else cleaned
    cleaned = cleaned.replace('-', '_').replace(' ', '_')
    return cleaned if cleaned in allowed else value


def validate_route(obj: dict, *, require_exact_keys: bool = True) -> str | None:
    missing = [k for k in REQUIRED_KEYS if k not in obj]
    if missing:
        return 'missing_keys:' + ','.join(missing)
    if require_exact_keys:
        extra = sorted(k for k in obj if k not in REQUIRED_KEYS)
        if extra:
            return 'extra_keys:' + ','.join(extra)
    obj['lane'] = normalize_enum(obj.get('lane'), ALLOWED_LANES, LANE_ALIASES)
    obj['priority'] = normalize_enum(obj.get('priority'), ALLOWED_PRIORITIES)
    obj['recommended_worker'] = normalize_enum(obj.get('recommended_worker'), ALLOWED_WORKERS)
    if obj.get('lane') not in ALLOWED_LANES:
        return f"invalid_lane:{obj.get('lane')!r}"
    if obj.get('priority') not in ALLOWED_PRIORITIES:
        return f"invalid_priority:{obj.get('priority')!r}"
    if obj.get('recommended_worker') not in ALLOWED_WORKERS:
        return f"invalid_worker:{obj.get('recommended_worker')!r}"
    if not isinstance(obj.get('reason'), str) or not obj.get('reason', '').strip():
        return 'invalid_reason'
    if obj.get('safety_gate') is not None and not isinstance(obj.get('safety_gate'), str):
        return 'invalid_safety_gate'
    return None


def coerce_strict_route(obj: dict | None) -> dict | None:
    if not isinstance(obj, dict):
        return None
    coerced = {k: obj.get(k) for k in REQUIRED_KEYS}
    if coerced['reason'] is None:
        coerced['reason'] = 'Model output was normalized into the required Mission Control route schema.'
    if 'safety_gtc' in obj and coerced['safety_gate'] is None:
        coerced['safety_gate'] = obj.get('safety_gtc')
    if 'safety_overrules' in obj and coerced['safety_gate'] is None:
        coerced['safety_gate'] = obj.get('safety_overrules')
    coerced['lane'] = normalize_enum(coerced.get('lane'), ALLOWED_LANES, LANE_ALIASES)
    coerced['priority'] = normalize_enum(coerced.get('priority'), ALLOWED_PRIORITIES)
    coerced['recommended_worker'] = normalize_enum(coerced.get('recommended_worker'), ALLOWED_WORKERS)
    return coerced


def compact_text(surface: str, item: dict[str, Any], context: dict[str, Any] | None = None) -> str:
    parts: list[str] = [surface]
    for key in ('status', 'priority', 'assignedWorker', 'appName', 'title', 'description', 'summary', 'risk'):
        value = item.get(key)
        if value is not None:
            parts.append(str(value))
    labels = item.get('labels')
    if isinstance(labels, list):
        parts.extend(str(x) for x in labels)
    if context:
        parts.append(json.dumps(context, sort_keys=True))
    return ' '.join(parts).lower()


def manual_gate_reason(surface: str, item: dict[str, Any], context: dict[str, Any] | None = None) -> str | None:
    text = compact_text(surface, item, context)
    for pattern, gate in MANUAL_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            if gate == 'publishing/live deployment approval' and re.search(r'\b(no-publish|publish\s*=\s*false|publish false)\b', text):
                continue
            if gate == 'payment/licensing approval' and 'vps optional hardening pass' in text:
                continue
            if gate == 'privacy review required' and re.search(r'\b(non-phi|no real patient data|fake/non-phi)\b', text):
                continue
            return gate
    if str(item.get('assignedWorker', '')).lower() == 'manual' and re.search(r'\b(manual:|manual acceptance|human acceptance|explicit approval)\b', text):
        return 'manual worker/acceptance gate'
    return None


def deterministic_override(surface: str, item: dict[str, Any], context: dict[str, Any] | None = None) -> dict | None:
    priority = item.get('priority') if item.get('priority') in ALLOWED_PRIORITIES else 'normal'
    status = str(item.get('status', '')).lower()
    linked = str(item.get('linkedTaskStatus', '')).lower()
    if status in {'completed', 'cancelled', 'resolved'} or linked in {'completed', 'cancelled'}:
        return {
            'lane': 'close_stale',
            'priority': priority,
            'recommended_worker': 'manual',
            'reason': 'Item is complete, cancelled, or linked to finished work; close/archive instead of resuming.',
            'safety_gate': None,
        }
    if surface == 'run_review' and status in {'approved', 'rejected'}:
        return {
            'lane': 'close_stale',
            'priority': priority,
            'recommended_worker': 'manual',
            'reason': 'Review already has a terminal decision; close/archive instead of resuming.',
            'safety_gate': None,
        }
    gate = manual_gate_reason(surface, item, context)
    if gate:
        return {
            'lane': 'manual_gate',
            'priority': priority,
            'recommended_worker': 'manual',
            'reason': 'Hard safety override: this item requires operator control or approval before automation.',
            'safety_gate': gate,
        }
    if surface == 'run_review' and status in {'needs_review', 'needs_changes'}:
        return {
            'lane': 'review_first',
            'priority': 'high' if item.get('risk') in {'high', 'lotto-risk'} else priority,
            'recommended_worker': 'manual',
            'reason': 'Automation Review is undecided/needs changes, so operator review comes before follow-up routing.',
            'safety_gate': 'operator review',
        }
    text = compact_text(surface, item, context)
    if status != 'blocked' and re.search(r'\b(review cards?|review rubric|operator should review|awaits? rating|needs operator|review preview|original-vs|review evidence)\b', text):
        return {
            'lane': 'review_first',
            'priority': priority,
            'recommended_worker': 'manual',
            'reason': 'Item is collecting or awaiting review evidence, so operator review comes before follow-up routing.',
            'safety_gate': 'operator review',
        }
    if status == 'blocked' and re.search(r'\b(parked|deferred|future|after mvp|not urgent|not next|superseded|backlog|later)\b', text):
        return {
            'lane': 'park_later',
            'priority': priority,
            'recommended_worker': 'manual',
            'reason': 'Item is blocked/deferred/future scope rather than the next MVP move.',
            'safety_gate': 'scope/dependency',
        }
    if status == 'blocked':
        return {
            'lane': 'park_later',
            'priority': priority,
            'recommended_worker': 'manual',
            'reason': 'Blocked item without a hard safety/manual gate should remain parked until its dependency is cleared.',
            'safety_gate': 'scope/dependency',
        }
    if status in {'pending', 'in_progress', ''}:
        worker = item.get('assignedWorker') if item.get('assignedWorker') in ALLOWED_WORKERS else 'hermes'
        return {
            'lane': 'safe_cli_next',
            'priority': priority,
            'recommended_worker': worker,
            'reason': 'Focused active/pending task has no deterministic manual gate and can proceed in CLI.',
            'safety_gate': None,
        }
    return None


def apply_deterministic_policy(surface: str, item: dict[str, Any], context: dict[str, Any] | None, parsed: dict | None) -> tuple[dict | None, str | None]:
    override = deterministic_override(surface, item, context)
    if override is not None:
        return override, 'deterministic_safety_override'
    return coerce_strict_route(parsed), None
