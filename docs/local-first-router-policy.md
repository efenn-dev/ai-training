# AI Training local-first / frontier-last router policy

Generated: 2026-06-19

## Inventory result

A focused repo search did not find a dedicated Claude local-routing app inside `ai-training`. The reusable pieces currently present are:

- `scripts/gemma_route_policy.py` — strict Mission Control route schema plus deterministic safety overrides for route evals.
- `scripts/eval_route_model_generic.py` — Unsloth-based route-model eval harness against the Gemma orchestrator dataset.
- `PHASE2_LOCAL_FALLBACK_HANDOFF.md` — handoff from Phase 1/2 local fallback work.
- `/mnt/c/projects/fable-trace-lora/data/reports/PHASE2_LOCAL_FALLBACK_ROUTER_POLICY.md` — companion repo policy and benchmark evidence.

Decision: reuse the existing safety-policy/eval concepts, but do not treat them as a production router. This task adds an explicit AI Training policy/config and a shadow-mode decision logger. Production routing remains disabled.

## Current evidence

From the Phase 2 local fallback policy:

| Candidate | Evidence | Decision |
| --- | --- | --- |
| Captured `local_qwen3_coder` | 49/64, 82.29% required-tool recall, 0 hallucinated tools, 79.17% verification honesty | promising but not fresh; not promoted by default |
| Fresh Qwen2.5-Coder-14B Q4 | normalized 16/64, 26.04% recall, 2 hallucinated `list_directory`, 0% verification honesty | triage/summaries only |
| Fresh Qwen3-4B | 24/64, 41.67% recall, 0 hallucinated tools, 0% verification honesty, very fast | summaries/classification/read-only triage only |

Tiny local coding remains disabled until a fresh endpoint earns `local_coding_with_review` with the thresholds in `config/local_frontier_router_policy.json`.

## Routing tiers

1. No model / deterministic scripts
   - Use for schema validation, manifest counts, dataset counts, eval metric aggregation, freshness checks, and packet generation.
   - This tier is preferred whenever it can answer the question without language-model judgment.

2. Local model draft only
   - Use for summarization, draft labels, route drafts, status packets, low-risk inspection, missing-field detection, and classification.
   - Output is advisory and must include model/provenance/confidence metadata.
   - Local output may create `draft` fields only. It must not mark examples reviewed or approve anything.

3. Frontier required
   - Required for risky edits, training promotion decisions, ambiguous product calls, hard debugging, config/secrets/routing changes, final reviews, and multi-file production code changes.

4. Human/operator required
   - Required for subjective labels, adapter approvals, production enablement, holdout split changes, and manual safety gates.

## Local model hard limits

Local models must never:

- set `status=reviewed`;
- approve adapters or promote training outputs;
- launch training by themselves;
- change production routing/config/secrets;
- overwrite holdout splits;
- claim tests/builds passed without real tool execution.

## Shadow-mode logging

The script `scripts/local_frontier_router.py` can classify a task and optionally append a shadow decision to `/mnt/s/AITraining/router-shadow/decisions.jsonl`.

Example:

```bash
python3 scripts/local_frontier_router.py \
  --task-type draft_label \
  --risk low \
  --text "Draft labels for gaming curation candidates" \
  --log-shadow
```

Shadow logs are for training/eval evidence only. They do not enable production routing.

## Promotion gate

Local tiny-code routing can only be considered after a fresh live endpoint benchmark meets all thresholds:

- verdict: `local_coding_with_review`;
- required-tool recall >= 75%;
- verification honesty >= 70%;
- hallucinated tool calls = 0;
- WSL path correctness >= 90%;
- no critical regressions on tool validity, inspect-before-edit, or verification honesty.

Adapter promotion and production routing still require a separate Automation Review/human approval task.
