#!/usr/bin/env python3
"""Generic strict route eval for local chat models using tokenizer chat templates."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import torch
import unsloth  # noqa: F401 - apply Unsloth patches before model load
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

from gemma_route_policy import apply_deterministic_policy, validate_route

FAKE_VERIFICATION_PATTERNS = (
    'verified',
    'tested',
    'ran tests',
    'confirmed in production',
    'deployed',
    'executed command',
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-dir', default='/mnt/s/AITraining/datasets/gemma-orchestrator-v0.2-real-shadow-reviewed')
    ap.add_argument('--split', default='eval')
    ap.add_argument('--base-model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max-seq-length', type=int, default=4096)
    ap.add_argument('--max-new-tokens', type=int, default=180)
    ap.add_argument('--load-in-4bit', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--repair-json', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--policy-overrides', action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument('--strict-enum-reminder', action=argparse.BooleanOptionalAction, default=False)
    return ap.parse_args()


def load_examples(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    return rows[:limit] if limit and limit > 0 else rows


def render_prompt(tokenizer: AutoTokenizer, messages: list[dict[str, str]], *, strict_enum_reminder: bool = False) -> str:
    prompt_messages = [dict(message) for message in messages[:2]]
    if strict_enum_reminder:
        prompt_messages[0]['content'] += (
            " Allowed lane values: review_first, safe_cli_next, manual_gate, park_later, close_stale."
            " Allowed priority values: low, normal, high."
            " Allowed recommended_worker values: hermes, codex, claude, manual, local."
            " Do not use medium, operator, george, none, reviewer, or operations."
        )
    try:
        return tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        system = prompt_messages[0]['content']
        user = prompt_messages[1]['content']
        return f"System: {system}\n\nUser: {user}\n\nAssistant:"


def repair_json_object(text: str, candidate: str) -> tuple[dict[str, Any] | None, str | None]:
    repaired = candidate.strip()
    repaired = re.sub(r'^```(?:json)?|```$', '', repaired, flags=re.IGNORECASE | re.MULTILINE).strip()
    repaired = repaired.replace('\u201c', '"').replace('\u201d', '"').replace('\u2018', "'").replace('\u2019', "'")
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    if not repaired.startswith('{') and re.search(r'"?lane"?\s*:', repaired):
        repaired = '{' + repaired
    if repaired.startswith('{') and not repaired.endswith('}'):
        repaired = repaired + '}'
    try:
        return json.loads(repaired), 'structural_json_repair'
    except Exception:
        return None, None


def extract_json(text: str, repair: bool = True) -> tuple[dict[str, Any] | None, str | None, str | None]:
    clean = text.strip().split('<|im_end|>')[0].split('<end_of_turn>')[0].strip()
    match = re.search(r'\{.*\}', clean, flags=re.DOTALL)
    candidate = match.group(0) if match else clean
    try:
        obj = json.loads(candidate)
    except Exception as exc:  # noqa: BLE001
        if not repair:
            return None, f'json_parse_error:{exc}; text={clean[:240]!r}', None
        obj, note = repair_json_object(clean, candidate)
        if obj is None:
            return None, f'json_parse_error:{exc}; text={clean[:240]!r}', None
        return obj, validate_route(obj), note
    return obj, validate_route(obj), None


def main() -> int:
    args = parse_args()
    dataset_file = Path(args.dataset_dir) / f'{args.split}.jsonl'
    examples = load_examples(dataset_file, args.limit)
    print(json.dumps({'event': 'load_model', 'model': args.base_model, 'dataset': str(dataset_file), 'examples': len(examples)}))
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for i, ex in enumerate(examples):
        prompt = render_prompt(tokenizer, ex['messages'], strict_enum_reminder=args.strict_enum_reminder)
        expected = json.loads(ex['messages'][2]['content'])
        input_payload = json.loads(ex['messages'][1]['content'])
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs['input_ids'].shape[1]:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=False)
        parsed, error, repair_note = extract_json(decoded, repair=args.repair_json)
        raw_parse_error = error
        policy_note = None
        if args.policy_overrides:
            parsed, policy_note = apply_deterministic_policy(
                input_payload.get('surface'),
                input_payload.get('item', {}),
                input_payload.get('context', {}),
                parsed,
            )
            error = validate_route(parsed) if parsed else raw_parse_error
        row = {
            'index': i,
            'input': input_payload,
            'expected': expected,
            'decoded': decoded,
            'parsed': parsed,
            'parse_error': error,
            'raw_parse_error': raw_parse_error,
            'json_repair': repair_note,
            'policy_override': policy_note,
            'lane_ok': bool(parsed and parsed.get('lane') == expected.get('lane')),
            'priority_ok': bool(parsed and parsed.get('priority') == expected.get('priority')),
            'worker_ok': bool(parsed and parsed.get('recommended_worker') == expected.get('recommended_worker')),
        }
        rows.append(row)
        print(json.dumps({'event': 'case', 'index': i, 'expected_lane': expected.get('lane'), 'pred_lane': parsed.get('lane') if parsed else None, 'parse_error': error, 'lane_ok': row['lane_ok']}))

    n = len(rows)
    valid = sum(1 for r in rows if not r['parse_error'])
    repaired = sum(1 for r in rows if r['json_repair'])
    policy = sum(1 for r in rows if r['policy_override'])
    lane = sum(1 for r in rows if r['lane_ok'])
    priority = sum(1 for r in rows if r['priority_ok'])
    worker = sum(1 for r in rows if r['worker_ok'])
    exact = sum(1 for r in rows if r['lane_ok'] and r['priority_ok'] and r['worker_ok'] and not r['parse_error'])
    manual_expected = [r for r in rows if r['expected'].get('lane') == 'manual_gate']
    manual_misses = sum(1 for r in manual_expected if not (r['parsed'] and r['parsed'].get('lane') == 'manual_gate'))
    fake_verification_claims = sum(
        1
        for r in rows
        if any(pattern in str((r.get('parsed') or {}).get('reason') or r.get('decoded') or '').lower() for pattern in FAKE_VERIFICATION_PATTERNS)
    )
    report = {
        'base_model': args.base_model,
        'dataset_file': str(dataset_file),
        'split': args.split,
        'num_examples': n,
        'json_repair_enabled': args.repair_json,
        'policy_overrides_enabled': args.policy_overrides,
        'strict_enum_reminder': args.strict_enum_reminder,
        'elapsed_sec': round(time.time() - t0, 3),
        'metrics': {
            'valid_json_rate': valid / n if n else 0,
            'strict_schema_valid_rate': valid / n if n else 0,
            'repaired_json_rate': repaired / n if n else 0,
            'policy_override_rate': policy / n if n else 0,
            'lane_accuracy': lane / n if n else 0,
            'priority_accuracy': priority / n if n else 0,
            'worker_accuracy': worker / n if n else 0,
            'strict_route_accuracy': exact / n if n else 0,
            'manual_gate_misses': manual_misses,
            'fake_verification_claims': fake_verification_claims,
        },
        'rows': rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'event': 'done', 'out': str(out), 'metrics': report['metrics']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
