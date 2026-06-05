#!/usr/bin/env python3
"""Build reviewed v0.2 Gemma route dataset from curated v0.1 plus real shadow evidence."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from gemma_route_policy import STRICT_SYSTEM, deterministic_override, validate_route

SRC_DIR = Path('/mnt/s/AITraining/datasets/gemma-orchestrator-v0.1-curated')
SHADOW_DIR = Path('/mnt/s/AITraining/runs/gemma-route-shadow-evidence-20260605')
OUT_DIR = Path('/mnt/s/AITraining/datasets/gemma-orchestrator-v0.2-real-shadow-reviewed')


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def label_for(surface: str, item: dict[str, Any], context: dict[str, Any], lane: str) -> dict[str, Any]:
    override = deterministic_override(surface, item, context)
    if override and override['lane'] == lane:
        return override
    priority = item.get('priority') if item.get('priority') in {'low', 'normal', 'high'} else 'normal'
    worker = item.get('assignedWorker') if item.get('assignedWorker') in {'hermes', 'codex', 'claude', 'manual', 'local'} else 'manual'
    if lane == 'review_first':
        return {'lane': lane, 'priority': priority, 'recommended_worker': 'manual', 'reason': 'Real shadow label: operator review is required before follow-up automation.', 'safety_gate': 'operator review'}
    if lane == 'safe_cli_next':
        return {'lane': lane, 'priority': priority, 'recommended_worker': worker, 'reason': 'Real shadow label: focused non-destructive work can proceed in CLI.', 'safety_gate': None}
    if lane == 'manual_gate':
        return {'lane': lane, 'priority': priority, 'recommended_worker': 'manual', 'reason': 'Real shadow label: next action requires human/operator gate.', 'safety_gate': 'manual acceptance'}
    if lane == 'park_later':
        return {'lane': lane, 'priority': priority, 'recommended_worker': 'manual', 'reason': 'Real shadow label: item is deferred, blocked, superseded, or not the next move.', 'safety_gate': 'scope/dependency'}
    if lane == 'close_stale':
        return {'lane': lane, 'priority': priority, 'recommended_worker': 'manual', 'reason': 'Real shadow label: item is completed/stale and should close/archive.', 'safety_gate': None}
    raise ValueError(lane)


def msg(surface: str, item: dict[str, Any], context: dict[str, Any], label: dict[str, Any], meta: dict[str, Any] | None = None) -> dict[str, Any]:
    error = validate_route(label)
    if error:
        raise ValueError(f'invalid label {error}: {label}')
    row = {
        'messages': [
            {'role': 'system', 'content': STRICT_SYSTEM},
            {'role': 'user', 'content': json.dumps({'surface': surface, 'item': item, 'context': context}, separators=(',', ':'))},
            {'role': 'assistant', 'content': json.dumps(label, separators=(',', ':'))},
        ]
    }
    if meta:
        row.update(meta)
    return row


def relabel_old(row: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(row['messages'][1]['content'])
    label = json.loads(row['messages'][2]['content'])
    return msg(payload['surface'], payload['item'], payload.get('context', {}), label)


def hardening_examples() -> list[dict[str, Any]]:
    fixtures = [
        ('task', {'appName':'clipforge','title':'Review final legal policy URLs before checkout links go live','description':'Terms, Privacy, Refund policy and public checkout links need explicit approval before deploy.','status':'blocked','priority':'high','assignedWorker':'manual'}, {}, 'manual_gate'),
        ('task', {'appName':'clipforge','title':'Rotate PayPal webhook secret and license pool credentials','description':'Do not paste secrets; operator must update local env and verify.','status':'pending','priority':'high','assignedWorker':'manual'}, {}, 'manual_gate'),
        ('task', {'appName':'minecraft-crew-bds','title':'Join live Discord voice and Minecraft game for acceptance test','description':'Requires Discord bot token, voice channel, spare Xbox account, and supervised live game session.','status':'blocked','priority':'high','assignedWorker':'manual'}, {}, 'manual_gate'),
        ('task', {'appName':'george-robot','title':'Run motor controller on physical hardware','description':'Needs physical supervision and hardware safety stop.','status':'pending','priority':'high','assignedWorker':'manual'}, {}, 'manual_gate'),
        ('task', {'appName':'frontdesk-helper','title':'User test with wife using fake non-PHI calls','description':'Capture feedback; no real patient data.','status':'pending','priority':'high','assignedWorker':'manual'}, {}, 'manual_gate'),
        ('run_review', {'appName':'ai-training','title':'Gemma route adapter evidence needs approval','status':'needs_review','risk':'medium','labels':['adapter','eval','promotion-gates']}, {}, 'review_first'),
        ('run_review', {'appName':'clipforge','title':'Video settings matrix output needs changes','status':'needs_changes','risk':'medium','labels':['video','review-preview','no-publish']}, {}, 'review_first'),
        ('task', {'appName':'ai-training','title':'Completed adapter integration card','status':'completed','priority':'normal','assignedWorker':'hermes'}, {}, 'close_stale'),
        ('task', {'appName':'clipforge','title':'Duplicate stale cleanup already merged','status':'completed','priority':'low','assignedWorker':'local'}, {}, 'close_stale'),
        ('task', {'appName':'ai-training','title':'Parked VLM router until labeled fixtures exist','description':'Explicitly parked until ClipLab MVP.','status':'blocked','priority':'normal','assignedWorker':'manual'}, {'mvp_scope_locked': True}, 'park_later'),
    ]
    return [msg(s, i, c, label_for(s, i, c, lane), {'source':'synthetic_hardening'}) for s, i, c, lane in fixtures]


def build() -> dict[str, Any]:
    old_train = [relabel_old(r) for r in load_jsonl(SRC_DIR / 'train.jsonl')]
    old_eval = [relabel_old(r) for r in load_jsonl(SRC_DIR / 'eval.jsonl')]
    inputs = load_jsonl(SHADOW_DIR / 'shadow-inputs.jsonl')
    pred_lines = (SHADOW_DIR / 'shadow-predictions.jsonl').read_text(encoding='utf-8').splitlines()
    preds = []
    for line in pred_lines:
        if line.strip().startswith('{'):
            preds.append(json.loads(line))
    pred_by_id = {p.get('id'): p for p in preds}
    shadow_rows = []
    for src in inputs:
        surface = src['surface']
        item = src['item']
        context = src.get('context', {})
        lane = src['policy_lane']
        pred = pred_by_id.get(src['id'], {})
        parsed = pred.get('parsed') or {}
        pred_lane = parsed.get('lane')
        schema_error = 'ok'
        if not parsed:
            schema_error = pred.get('parse_error') or 'missing_parse'
        else:
            schema_error = validate_route(dict(parsed)) or 'ok'
        label = label_for(surface, item, context, lane)
        shadow_rows.append(msg(surface, item, context, label, {
            'id': src['id'],
            'surface': surface,
            'source': 'real_shadow_reviewed_20260605',
            'shadow_policy_lane': lane,
            'shadow_adapter_lane': pred_lane,
            'shadow_schema_error': schema_error,
            'shadow_disagreement': pred_lane != lane,
        }))
    train = old_train + hardening_examples()
    eval_rows = old_eval
    holdout_real = shadow_rows
    eval_real_plus_curated = old_eval + shadow_rows
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in [('train.jsonl', train), ('eval.jsonl', eval_rows), ('holdout_real_shadow.jsonl', holdout_real), ('eval_real_plus_curated.jsonl', eval_real_plus_curated), ('reviewed_shadow_cases.jsonl', shadow_rows)]:
        with (OUT_DIR / name).open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    counts = {}
    for name, rows in [('train', train), ('eval', eval_rows), ('holdout_real_shadow', holdout_real), ('eval_real_plus_curated', eval_real_plus_curated)]:
        counts[name] = {
            'rows': len(rows),
            'lanes': dict(Counter(json.loads(r['messages'][2]['content'])['lane'] for r in rows)),
        }
    disagreements = [r for r in shadow_rows if r.get('shadow_disagreement')]
    schema = {
        'name': 'gemma-orchestrator-v0.2-real-shadow-reviewed',
        'source_evidence': str(SHADOW_DIR / 'shadow-evidence.md'),
        'privacy': 'Uses sanitized Mission Control task/review metadata only; no raw chats, credentials, logs, diffs, or media artifacts.',
        'strict_required_keys': ['lane', 'priority', 'recommended_worker', 'reason', 'safety_gate'],
        'counts': counts,
        'review_notes': [
            'Added all 30 real shadow cases as reviewed_shadow_cases/holdout_real_shadow with model prediction lane, disagreement flag, and schema error metadata.',
            'Special focus lanes from evidence are preserved: review_first, close_stale, and manual_gate.',
            'Training split receives synthetic hardening cases for deterministic safety override language without contaminating the real shadow holdout.',
        ],
        'shadow_disagreements': len(disagreements),
    }
    (OUT_DIR / 'schema.json').write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding='utf-8')
    (OUT_DIR / 'README.md').write_text(
        '# Gemma Orchestrator v0.2 Real Shadow Reviewed\n\n'
        f'Source evidence: `{SHADOW_DIR / "shadow-evidence.md"}`\n\n'
        'All 30 sanitized real Mission Control shadow cases are included in `reviewed_shadow_cases.jsonl` and `holdout_real_shadow.jsonl`.\n'
        'Training uses prior curated data plus safety hardening examples; the real shadow split remains a holdout for comparison.\n\n'
        f'Counts: `{json.dumps(counts, sort_keys=True)}`\n',
        encoding='utf-8',
    )
    return schema


if __name__ == '__main__':
    print(json.dumps(build(), indent=2, ensure_ascii=False))
