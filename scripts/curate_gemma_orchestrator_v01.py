#!/usr/bin/env python3
"""Curate a privacy-safe Gemma/Mission Control route-classification dataset.

This intentionally uses synthetic/summarized backlog metadata only. It does not read
raw chats, logs, credentials, source files, or media artifacts.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

OUT_DIR = Path('/mnt/s/AITraining/datasets/gemma-orchestrator-v0.1-curated')
SYSTEM = (
    "You are George's local Mission Control orchestrator. Classify one backlog/review item into a safe next lane. "
    "Return strict JSON only with lane, priority, recommended_worker, reason, safety_gate. "
    "Lane rules: review_first=undecided review artifacts or subjective media awaiting operator judgement; "
    "safe_cli_next=non-destructive implementation, triage, eval, or follow-up task can run in CLI now; "
    "manual_gate=credentials, live account/game/audio control, publishing, privacy review, hardware, or human acceptance gate; "
    "park_later=explicitly deferred, parked, superseded, low-priority future work, or not next for MVP; "
    "close_stale=complete, duplicate, obsolete, already archived, or close without resuming."
)

LANE_RULES = {
    'review_first': 'Undecided review artifacts or subjective media awaiting operator judgement.',
    'safe_cli_next': 'A non-destructive implementation, triage, eval, or follow-up task can safely run in CLI now.',
    'manual_gate': 'The next action requires credentials, live account/game/audio control, publishing, or a human acceptance gate.',
    'park_later': 'Useful but explicitly deferred, archived, superseded, low-priority future work, or not next for MVP.',
    'close_stale': 'The lane/session/review is complete, duplicate, obsolete, or should be archived without resuming work.',
}


def msg(surface: str, item: dict, context: dict, lane: str, priority: str, worker: str, reason: str, gate=None):
    return {
        'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': json.dumps({'surface': surface, 'item': item, 'context': context}, separators=(',', ':'))},
            {'role': 'assistant', 'content': json.dumps({'lane': lane, 'priority': priority, 'recommended_worker': worker, 'reason': reason, 'safety_gate': gate}, separators=(',', ':'))},
        ]
    }

examples = []
# review_first: active automation outputs needing operator decision.
for app, title, risk, labels, prio in [
    ('clipforge', 'Generated highlight clip candidate awaits rating', 'medium', ['video','clip','needs-rating'], 'high'),
    ('beat-lab', 'New song render needs diamond/nugget decision', 'low', ['music','subjective'], 'normal'),
    ('george-desktop', 'Voice UX demo recording awaits acceptance review', 'medium', ['demo','voice','needs-review'], 'high'),
    ('ai-training', 'Adapter eval report needs approve/reject decision', 'medium', ['automation','eval-report'], 'normal'),
    ('clipforge', 'Settings-matrix route output waiting for operator review', 'medium', ['automation','route-training','real-fixture'], 'high'),
    ('beat-lab', 'Recovered audio render needs human quality rating', 'medium', ['audio','manual-review'], 'normal'),
    ('app-manager-mcp', 'Kanban route demo still waiting for approve or changes decision', 'medium', ['automation','ui','decision-needed'], 'high'),
    ('clipforge', 'Subjective export preview is undecided in Automation Review', 'medium', ['video','subjective','undecided'], 'high'),
    ('george-desktop', 'Operator has not accepted or rejected the new voice sample yet', 'low', ['voice','acceptance','pending-human'], 'normal'),
    ('ai-training', 'Comparison artifact is marked needs_changes and needs human triage first', 'medium', ['eval','needs_changes','automation-review'], 'normal'),
    ('beat-lab', 'Manual listen pass needed before creating follow-up mixing task', 'low', ['music','listen-pass','review'], 'normal'),
]:
    examples.append(msg('run_review', {'app': app, 'title': title, 'status': 'needs_review', 'risk': risk, 'labels': labels}, {'active_workers': 1}, 'review_first', prio, 'manual', 'Undecided subjective or automation review should be judged before follow-up automation.', 'operator review'))

# safe_cli_next: focused implementation/eval/triage with no broad destructive side effects.
for app, title, worker, prio in [
    ('app-manager-mcp', 'Add deterministic route classifier unit tests', 'claude', 'high'),
    ('ai-training', 'Run baseline vs adapter eval on sanitized examples', 'local', 'high'),
    ('clipforge', 'Implement transcript sidecar filename normalization', 'codex', 'normal'),
    ('beat-lab', 'Regenerate preset manifest from checked-in presets', 'hermes', 'normal'),
    ('george-desktop', 'Fix microphone diagnostic panel copy', 'claude', 'normal'),
    ('ai-training', 'Curate synthetic Mission Control labels', 'local', 'high'),
    ('clipforge', 'Review route-training note and create follow-up task', 'hermes', 'normal'),
    ('app-manager-mcp', 'Tighten automation review archive filter', 'claude', 'high'),
]:
    examples.append(msg('task', {'app': app, 'title': title, 'status': 'pending', 'priority': prio, 'assignedWorker': worker}, {'repo_active_workers': 0, 'total_write_workers': 0}, 'safe_cli_next', prio, worker, 'Focused non-destructive task has no manual gate and worker budget is available.', None))

# manual_gate: credentials, live services, publishing, hardware, or subjective acceptance.
for app, title, prio, gate in [
    ('minecraft-pizzeria', 'Live Bedrock bot test with player session', 'high', 'live game session and supervision'),
    ('george-desktop', 'Enable Discord voice token and join live channel', 'high', 'credentials and live voice channel'),
    ('clipforge', 'Publish generated shorts to YouTube account', 'high', 'publishing account approval'),
    ('beat-lab', 'Choose diamond track for public release', 'normal', 'human taste decision'),
    ('license-server', 'Rotate payment provider credentials', 'high', 'secret handling'),
    ('app-manager-mcp', 'Approve broad dirty diff touching multiple apps', 'high', 'human approval for broad changes'),
    ('ai-training', 'Train on private chat export', 'high', 'privacy review required'),
    ('george-robot', 'Run motor controller on physical hardware', 'high', 'hardware safety supervision'),
]:
    examples.append(msg('task', {'app': app, 'title': title, 'status': 'pending', 'priority': prio, 'assignedWorker': 'manual'}, {'repo_active_workers': 0}, 'manual_gate', prio, 'manual', 'Next action involves credentials, live systems, publishing, privacy, hardware, or taste judgement.', gate))

# park_later: explicitly deferred/superseded/not MVP.
for app, title, prio in [
    ('ai-training', 'Experimental giant model tune after MVP', 'normal'),
    ('clipforge', 'ComfyUI exploratory route after core export ships', 'normal'),
    ('beat-lab', 'Archive rejected scratch render idea', 'low'),
    ('george-robot', 'Robot body dream backlog concept', 'normal'),
    ('trading', 'Long-term research note not actionable intraday', 'low'),
    ('clipforge', 'Superseded settings matrix rerun evidence', 'normal'),
    ('ai-training', 'Parked VLM router until labeled fixtures exist', 'normal'),
    ('marketing-site', 'Nice-to-have animation polish after paid beta', 'normal'),
]:
    examples.append(msg('task', {'app': app, 'title': title, 'status': 'blocked', 'priority': prio, 'assignedWorker': 'manual'}, {'mvp_scope_locked': True}, 'park_later', prio, 'manual', 'The item is deferred, superseded, blocked on future scope, or not the next MVP move.', 'scope/dependency'))

# close_stale: worker sessions/reviews that should not be resumed.
for surface, item in [
    ('worker_session', {'label': 'Old Codex lane for completed task', 'status': 'paused', 'linkedTaskStatus': 'completed'}),
    ('worker_session', {'label': 'Duplicate Hermes eval run', 'status': 'paused', 'linkedTaskStatus': 'cancelled'}),
    ('run_review', {'app': 'clipforge', 'title': 'Rejected duplicate smoke run', 'status': 'rejected', 'risk': 'low', 'labels': ['duplicate','smoke']}),
    ('run_review', {'app': 'app-manager-mcp', 'title': 'Approved review already converted to task', 'status': 'approved', 'risk': 'medium', 'labels': ['linked-task-created']}),
    ('task', {'app': 'beat-lab', 'title': 'Obsolete one-off temp cleanup', 'status': 'completed', 'priority': 'normal'}),
    ('worker_session', {'label': 'Worker whose branch was merged', 'status': 'paused', 'linkedTaskStatus': 'completed'}),
    ('worker_session', {'label': 'Old local terminal for cancelled experiment', 'status': 'paused', 'linkedTaskStatus': 'cancelled'}),
    ('run_review', {'app': 'ai-training', 'title': 'Superseded adapter smoke report from yesterday', 'status': 'rejected', 'risk': 'low', 'labels': ['superseded','adapter']}),
    ('run_review', {'app': 'beat-lab', 'title': 'Approved nugget review already archived', 'status': 'approved', 'risk': 'low', 'labels': ['approved','archived']}),
    ('task', {'app': 'app-manager-mcp', 'title': 'Duplicate route badge fix already merged', 'status': 'completed', 'priority': 'normal'}),
    ('task', {'app': 'clipforge', 'title': 'Cancelled stale branch cleanup card', 'status': 'cancelled', 'priority': 'low'}),
]:
    examples.append(msg(surface, item, {}, 'close_stale', 'normal', 'manual', 'The item is complete, duplicate, obsolete, or already converted; close/archive instead of resuming.', None))

# Deterministic holdout eval with phrasing not duplicated exactly from train.
eval_examples = [
    msg('run_review', {'app':'clipforge','title':'Auto Edit candidate clip needs operator rating','status':'needs_review','risk':'medium','labels':['video','needs-review']}, {}, 'review_first','high','manual','Undecided subjective or automation review should be judged before follow-up automation.','operator review'),
    msg('run_review', {'app':'beat-lab','title':'Fresh instrumental render awaits nugget decision','status':'needs_review','risk':'low','labels':['music','manual-review']}, {}, 'review_first','normal','manual','Undecided subjective or automation review should be judged before follow-up automation.','operator review'),
    msg('run_review', {'app':'ai-training','title':'needs_changes comparison report should be reviewed before another promotion attempt','status':'needs_changes','risk':'medium','labels':['adapter','eval','needs-decision']}, {}, 'review_first','normal','manual','Undecided subjective or automation review should be judged before follow-up automation.','operator review'),
    msg('run_review', {'app':'george-desktop','title':'Voice Lab sample is still awaiting accept/reject from operator','status':'needs_review','risk':'low','labels':['voice','acceptance-review']}, {}, 'review_first','normal','manual','Undecided subjective or automation review should be judged before follow-up automation.','operator review'),
    msg('run_review', {'app':'clipforge','title':'Subjective highlight export has no diamond/nugget verdict yet','status':'needs_review','risk':'medium','labels':['video','subjective','manual-rating']}, {}, 'review_first','high','manual','Undecided subjective or automation review should be judged before follow-up automation.','operator review'),
    msg('task', {'app':'ai-training','title':'Compare Gemma baseline and adapter on route eval','status':'pending','priority':'high','assignedWorker':'local'}, {'repo_active_workers':0}, 'safe_cli_next','high','local','Focused non-destructive task has no manual gate and worker budget is available.', None),
    msg('task', {'app':'app-manager-mcp','title':'Add route badge snapshot test','status':'pending','priority':'normal','assignedWorker':'claude'}, {'repo_active_workers':0}, 'safe_cli_next','normal','claude','Focused non-destructive task has no manual gate and worker budget is available.', None),
    msg('task', {'app':'george-desktop','title':'Join live Discord voice room for acceptance test','status':'pending','priority':'high','assignedWorker':'manual'}, {}, 'manual_gate','high','manual','Next action involves credentials, live systems, publishing, privacy, hardware, or taste judgement.','credentials and live voice channel'),
    msg('task', {'app':'clipforge','title':'Publish selected highlight to public channel','status':'pending','priority':'high','assignedWorker':'manual'}, {}, 'manual_gate','high','manual','Next action involves credentials, live systems, publishing, privacy, hardware, or taste judgement.','publishing account approval'),
    msg('task', {'app':'ai-training','title':'Park multimodal router until labeled fixtures exist','status':'blocked','priority':'normal','assignedWorker':'manual'}, {'mvp_scope_locked': True}, 'park_later','normal','manual','The item is deferred, superseded, blocked on future scope, or not the next MVP move.','scope/dependency'),
    msg('task', {'app':'george-robot','title':'Dream robot chassis brainstorming after paid beta','status':'blocked','priority':'normal','assignedWorker':'manual'}, {'mvp_scope_locked': True}, 'park_later','normal','manual','The item is deferred, superseded, blocked on future scope, or not the next MVP move.','scope/dependency'),
    msg('worker_session', {'label':'Paused Claude lane for completed task','status':'paused','linkedTaskStatus':'completed'}, {}, 'close_stale','normal','manual','The item is complete, duplicate, obsolete, or already converted; close/archive instead of resuming.', None),
    msg('run_review', {'app':'clipforge','title':'Approved review already spawned follow-up task','status':'approved','risk':'medium','labels':['linked-task-created']}, {}, 'close_stale','normal','manual','The item is complete, duplicate, obsolete, or already converted; close/archive instead of resuming.', None),
    msg('worker_session', {'label':'Cancelled Codex experiment window from an obsolete branch','status':'paused','linkedTaskStatus':'cancelled'}, {}, 'close_stale','normal','manual','The item is complete, duplicate, obsolete, or already converted; close/archive instead of resuming.', None),
    msg('task', {'app':'beat-lab','title':'Duplicate cleanup card already finished by another worker','status':'completed','priority':'low'}, {}, 'close_stale','normal','manual','The item is complete, duplicate, obsolete, or already converted; close/archive instead of resuming.', None),
    msg('run_review', {'app':'ai-training','title':'Rejected prior adapter smoke run should stay archived','status':'rejected','risk':'low','labels':['rejected','superseded']}, {}, 'close_stale','normal','manual','The item is complete, duplicate, obsolete, or already converted; close/archive instead of resuming.', None),
]

OUT_DIR.mkdir(parents=True, exist_ok=True)
train = examples
holdout = eval_examples
for name, rows in [('train.jsonl', train), ('eval.jsonl', holdout)]:
    with (OUT_DIR / name).open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

schema = {
    'name': 'gemma-orchestrator-v0.1-curated',
    'purpose': 'Route Mission Control backlog/review/session items into safe next lanes from sanitized summarized metadata.',
    'labels': LANE_RULES,
    'output_schema': {
        'lane': 'review_first|safe_cli_next|manual_gate|park_later|close_stale',
        'priority': 'low|normal|high',
        'recommended_worker': 'hermes|codex|claude|manual|local',
        'reason': 'string',
        'safety_gate': 'string|null',
    },
    'privacy': 'Synthetic/summarized examples only; no raw private chats, credentials, full logs, source diffs, or media artifacts.',
    'revision_notes': [
        'Initial v0.1 labels were useful as a smoke test but too imbalanced for a useful adapter: train lanes were park_later=13, manual_gate=12, and only one each of review_first/safe_cli_next/close_stale.',
        'Several initial run_review labels collapsed approved/rejected/needs_changes cases into broad park_later/manual_gate without deterministic route rules.',
        'This curated set balances all five lanes and encodes explicit Mission Control policy: undecided reviews -> review_first; safe focused tasks -> safe_cli_next; credentials/live/publishing/privacy/hardware/taste -> manual_gate; deferred/superseded/future -> park_later; completed/duplicate/converted -> close_stale.',
    ],
    'counts': {'train': len(train), 'eval': len(holdout), 'train_lanes': Counter(json.loads(r['messages'][2]['content'])['lane'] for r in train), 'eval_lanes': Counter(json.loads(r['messages'][2]['content'])['lane'] for r in holdout)},
}
# Convert Counters for JSON.
schema['counts']['train_lanes'] = dict(schema['counts']['train_lanes'])
schema['counts']['eval_lanes'] = dict(schema['counts']['eval_lanes'])
(OUT_DIR / 'schema.json').write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding='utf-8')
(OUT_DIR / 'README.md').write_text(
    '# Gemma Orchestrator v0.1 Curated\n\n'
    'Privacy-safe synthetic/summarized Mission Control routing dataset. No raw private chats, secrets, credentials, logs, diffs, or media artifacts.\n\n'
    f"Counts: train={len(train)}, eval={len(holdout)}.\n\n"
    'Revision: the original v0.1 smoke labels were revised because the class distribution was heavily imbalanced and did not provide deterministic route-classification coverage.\n',
    encoding='utf-8',
)
print(json.dumps({'out_dir': str(OUT_DIR), 'train': len(train), 'eval': len(holdout), 'train_lanes': schema['counts']['train_lanes'], 'eval_lanes': schema['counts']['eval_lanes']}, indent=2))
