# AI Training / Unsloth — Phase 2 Local Fallback Handoff

Generated: 2026-06-15

Use this as the fresh-session starting context for continuing the local fallback / trace-capture work. Ignore old long transcript context unless specifically needed; verify current files and processes before making changes.

## App

- App: AI Training / Unsloth
- App workdir: `/mnt/c/projects/ai-training-lab/unsloth`
- Companion eval/code repo used for Phase 1: `/mnt/c/projects/fable-trace-lora`
- Central trace root: `/mnt/s/AITraining/traces`
- Mission Control active Phase 2 task: `task_20260615151749_1pet4b`
- Phase 1 review: `review_20260615151810_za7qso`

## Phase 1 status

Phase 1 is complete and local-only committed in the companion repo:

- Repo: `/mnt/c/projects/fable-trace-lora`
- Commit: `63ebc95 add local fallback phase1 scorecard`
- No remote configured for that repo, so nothing was pushed.
- Verification: `28 passed` and `py_compile` passed.

Phase 1 added:

- OpenAI-compatible / llama.cpp response generation:
  - `/mnt/c/projects/fable-trace-lora/scripts/generate_agent_eval_responses.py`
- Human-readable local fallback scorecard:
  - `/mnt/c/projects/fable-trace-lora/src/fable_trace_lora/agent_evals.py`
- Tests:
  - `/mnt/c/projects/fable-trace-lora/tests/test_local_fallback_phase1.py`
- Main report:
  - `/mnt/c/projects/fable-trace-lora/data/reports/PHASE1_LOCAL_FALLBACK_STATUS.md`

## Phase 1 results

Existing captured-response baseline:

| Candidate | Verdict | Passed | Total | Required-tool recall | Hallucinated tools | Verification honesty |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `local_qwen3_coder` | `local_coding_with_review` | 49 | 64 | 82.29% | 0 | 79.17% |
| `base_qwen` | `summaries_only_poor_verification_honesty` | 24 | 64 | 41.67% | 0 | 0.00% |
| `lora_candidate` | `triage_only_tool_format_unsafe` | 8 | 64 | 8.33% | 32 | 0.00% |

Fresh live endpoint benchmark:

- Endpoint: `http://172.29.112.1:11435/v1/chat/completions`
- Loaded model: `unsloth/Qwen2.5-Coder-14B-Instruct-GGUF:Q4_K_M`
- Speed: about 3.9s/case over 64 cases.

| Scoring mode | Verdict | Passed | Total | Required-tool recall | Hallucinated tools | Verification honesty |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw | `summaries_only_poor_verification_honesty` | 0 | 64 | 0.00% | 0 | 0.00% |
| normalized | `triage_only_tool_format_unsafe` | 16 | 64 | 26.04% | 2 `list_directory` | 0.00% |

Decision:

- Do not use Qwen2.5-Coder-14B Q4 as the default autonomous local coder.
- It is acceptable only for summaries / triage unless prompt/output-adapter work substantially improves score.
- Captured `local_qwen3_coder` is the best current candidate, but needs a fresh live endpoint run before promotion.

## Trace capture status

Central imports created:

- `/mnt/s/AITraining/traces/imports/local-fallback-phase1/20260615-101301/IMPORT_MANIFEST.json`
  - Included Phase4a Hermes rows and format/tool calibration rows.
- `/mnt/s/AITraining/traces/imports/local-fallback-phase1-fresh-eval/20260615-101707/IMPORT_MANIFEST.json`
  - Included fresh Qwen2.5-Coder-14B raw/normalized eval outputs and scorecards.

Known trace gap:

- AI Training trace infrastructure exists, but live raw capture is still minimal.
- Sanitized/normalized central trace folders were empty at Phase 1 inspection.
- Phase 2 should ensure local fallback eval prompts, outputs, scorecards, escalation decisions, and accepted/rejected work outcomes are captured or import-manifested.

## Phase 2 goal

Rerun the best candidate fresh from a live local endpoint and wire a local-first router policy.

Target Mission Control task:

- `task_20260615151749_1pet4b`: Phase 2 local fallback router and fresh qwen3-coder benchmark

Desired policy:

1. Local models handle summaries / triage by default.
2. Local model can do tiny coding only if a fresh scorecard earns `local_coding_with_review` with zero hallucinated tool names.
3. Risky/hard tasks produce structured escalation packets for Hermes/Claude instead of local autonomous edits.
4. Scorecards and routing decisions are trace-captured for future training.

## Recommended first safe step for Phase 2

Do not train yet. Start with discovery and a fresh benchmark.

1. Check local model availability and VRAM:

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
python3 - <<'PY'
import urllib.request
for url in ['http://172.29.112.1:11435/v1/models','http://127.0.0.1:8012/v1/models']:
    print('\n##', url)
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            print(r.status, r.read(1500).decode('utf-8','replace'))
    except Exception as e:
        print('ERR', repr(e))
PY
```

2. If a qwen3-coder endpoint is live, run a 2-case smoke with the Phase 1 generator.
3. Score raw and normalized.
4. Only if smoke is coherent, run full 64-case eval.
5. Compare with Phase 1 local_qwen3_coder captured baseline and fresh Qwen2.5-Coder-14B result.

## Known local launcher hints

Memory says local LLMs should prefer llama.cpp Vulkan over Ollama for this RTX 5070 Ti system.

Known script from Phase 1 inspection:

- `/mnt/c/projects/local-llm/scripts/start-qwen3-coder-server.sh`
- It uses:
  - model: `/mnt/c/projects/local-llm/models/qwen3-coder-30b-a3b/Qwen3-Coder-30B-A3B-Instruct-Q3_K_M.gguf`
  - default host/port: `127.0.0.1:8012`
  - llama-server with `--jinja`, `--metrics`, `--parallel 1`, `--cont-batching`

Be careful not to kill or replace the existing `172.29.112.1:11435` server unless needed; it may be managed by Windows/host-side tooling.

## Phase 2 acceptance criteria

- Fresh qwen3-coder or best available local candidate is benchmarked raw and normalized on the 64-case eval.
- Report compares:
  - captured `local_qwen3_coder`
  - fresh Qwen2.5-Coder-14B Q4
  - fresh Phase 2 candidate
- Router policy exists in code or at least a concrete config/spec, with clear thresholds:
  - summaries/triage allowed
  - tiny coding allowed only after `local_coding_with_review`
  - escalation packet required otherwise
- New eval outputs and routing decisions are copied/import-manifested under `/mnt/s/AITraining/traces/imports/`.
- Tests pass.
- Mission Control task and recap are updated.

## Current cautions

- `/mnt/c/projects/ai-training-lab/unsloth` has unrelated pre-existing dirty/untracked files, especially ClipLab Edit Director scripts. Do not sweep them into unrelated commits.
- Phase 1 work is in `/mnt/c/projects/fable-trace-lora`, not the AI Training app repo.
- Mission Control git gate for AI Training may require LOCAL-ONLY wording unless ai-training repo changes are committed/pushed.
- No more LoRA training until scorecard data shows exactly what behavior to improve.
