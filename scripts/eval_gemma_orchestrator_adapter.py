#!/usr/bin/env python3
"""Deterministic route-classification eval for Gemma orchestrator adapters."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
import unsloth  # noqa: F401 - ensures Unsloth patches are applied before transformers/peft use
from peft import PeftModel
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

from gemma_route_policy import (
    REQUIRED_KEYS,
    apply_deterministic_policy,
    coerce_strict_route,
    validate_route,
)


def build_prompt(ex: dict) -> tuple[str, dict]:
    messages = ex['messages']
    system = messages[0]['content']
    user = messages[1]['content']
    expected = json.loads(messages[2]['content'])
    prompt = '<start_of_turn>user\n' + system + '\n\n' + user + '<end_of_turn>\n<start_of_turn>model\n'
    return prompt, expected


def repair_json_object(text: str, candidate: str) -> tuple[dict | None, str | None]:
    """Best-effort repair for eval stability; never looks at expected labels."""
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
        pass

    fields: dict[str, object] = {}
    for key in REQUIRED_KEYS:
        pattern = rf'["\']?{re.escape(key)}["\']?\s*:\s*(null|None|["\']([^"\']*)["\']|[^,}}\n]+)'
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        raw_value = match.group(1).strip()
        if raw_value in {'null', 'None'}:
            fields[key] = None
        elif match.group(2) is not None:
            fields[key] = match.group(2).strip()
        else:
            fields[key] = raw_value.strip().strip('"\'`')
    if fields:
        fields.setdefault('reason', 'Recovered from model output during deterministic eval.')
        fields.setdefault('safety_gate', None)
        return coerce_strict_route(fields), 'field_extraction_repair'
    return None, None


def extract_json(text: str, repair: bool = True) -> tuple[dict | None, str | None, str | None]:
    text = text.strip()
    text = text.split('<end_of_turn>')[0].strip()
    match = re.search(r'\{.*\}', text, flags=re.DOTALL)
    candidate = match.group(0) if match else text
    try:
        obj = json.loads(candidate)
    except Exception as exc:  # noqa: BLE001
        if not repair:
            return None, f'json_parse_error: {exc}; text={text[:240]!r}', None
        obj, repair_note = repair_json_object(text, candidate)
        if obj is None:
            return None, f'json_parse_error: {exc}; text={text[:240]!r}', None
        error = validate_route(obj)
        return obj, error, repair_note

    error = validate_route(obj)
    if error and repair:
        repaired_obj, repair_note = repair_json_object(text, candidate)
        if repaired_obj is not None:
            repaired_error = validate_route(repaired_obj)
            if repaired_error is None:
                return repaired_obj, None, repair_note
    return obj, error, None


def load_examples(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir', default='/mnt/s/AITraining/datasets/gemma-orchestrator-v0.1-curated')
    ap.add_argument('--split', default='eval')
    ap.add_argument('--base-model', default='unsloth/gemma-3-1b-it-unsloth-bnb-4bit')
    ap.add_argument('--adapter-dir', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-new-tokens', type=int, default=160)
    ap.add_argument('--repair-json', action=argparse.BooleanOptionalAction, default=True, help='Apply deterministic post-parse JSON repair before scoring.')
    ap.add_argument('--policy-overrides', action=argparse.BooleanOptionalAction, default=True, help='Apply deterministic safety/schema policy overrides before scoring.')
    args = ap.parse_args()

    dataset_file = Path(args.dataset_dir) / f'{args.split}.jsonl'
    examples = load_examples(dataset_file)
    print(json.dumps({'event': 'load_model', 'base_model': args.base_model, 'adapter_dir': args.adapter_dir or None, 'examples': len(examples)}, indent=2))
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=args.base_model, max_seq_length=2048, load_in_4bit=True)
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = []
    t0 = time.time()
    for i, ex in enumerate(examples):
        prompt, expected = build_prompt(ex)
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
            parsed, policy_note = apply_deterministic_policy(ex.get('surface') or json.loads(ex['messages'][1]['content']).get('surface'), json.loads(ex['messages'][1]['content']).get('item', {}), json.loads(ex['messages'][1]['content']).get('context', {}), parsed)
            error = validate_route(parsed) if parsed else raw_parse_error
        row = {
            'index': i,
            'input': json.loads(ex['messages'][1]['content']),
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
        print(json.dumps({'event': 'case', 'index': i, 'expected_lane': expected['lane'], 'pred_lane': parsed.get('lane') if parsed else None, 'parse_error': error, 'json_repair': repair_note, 'policy_override': policy_note, 'lane_ok': row['lane_ok']}))

    n = len(rows)
    valid = sum(1 for r in rows if not r['parse_error'])
    repaired = sum(1 for r in rows if r['json_repair'])
    policy_overrides = sum(1 for r in rows if r['policy_override'])
    lane = sum(1 for r in rows if r['lane_ok'])
    priority = sum(1 for r in rows if r['priority_ok'])
    worker = sum(1 for r in rows if r['worker_ok'])
    exact = sum(1 for r in rows if r['lane_ok'] and r['priority_ok'] and r['worker_ok'] and not r['parse_error'])
    manual_gate_expected = [r for r in rows if r['expected'].get('lane') == 'manual_gate']
    manual_gate_misses = sum(1 for r in manual_gate_expected if not (r['parsed'] and r['parsed'].get('lane') == 'manual_gate'))
    report = {
        'base_model': args.base_model,
        'adapter_dir': args.adapter_dir or None,
        'dataset_file': str(dataset_file),
        'split': args.split,
        'num_examples': n,
        'json_repair_enabled': args.repair_json,
        'policy_overrides_enabled': args.policy_overrides,
        'elapsed_sec': round(time.time() - t0, 3),
        'metrics': {
            'valid_json_rate': valid / n if n else 0,
            'strict_schema_valid_rate': valid / n if n else 0,
            'repaired_json_rate': repaired / n if n else 0,
            'policy_override_rate': policy_overrides / n if n else 0,
            'lane_accuracy': lane / n if n else 0,
            'lane_agreement': lane / n if n else 0,
            'priority_accuracy': priority / n if n else 0,
            'worker_accuracy': worker / n if n else 0,
            'strict_route_accuracy': exact / n if n else 0,
            'manual_gate_misses': manual_gate_misses,
        },
        'rows': rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'event': 'done', 'out': str(out), 'metrics': report['metrics']}, indent=2))


if __name__ == '__main__':
    main()
