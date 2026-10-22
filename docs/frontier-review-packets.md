# Compact frontier review packets

`scripts/generate_frontier_review_packet.py` builds small Markdown and JSON artifacts for deciding whether a frontier reviewer is actually needed. The generator is deterministic and pointer-first: by default it summarizes counts, short examples, digests, metrics, and artifact paths without embedding raw transcripts, logs, prompts, completions, or full datasets.

## Supported lanes

- `gaming-pipeline` — gaming/video curation and ClipLab-style candidate review packets.
- `general-ops-router` — Mission Control / ops routing eval and queue packets.

## Basic usage

```bash
python3 scripts/generate_frontier_review_packet.py \
  --lane gaming-pipeline \
  --queue /mnt/s/AITraining/curation/queue.jsonl \
  --dataset-dir /mnt/s/AITraining/datasets/cliplab-edit-director-v0.1 \
  --eval-report /mnt/s/AITraining/runs/<run>/eval-adapter.json \
  --artifact /mnt/s/AITraining/runs/<run> \
  --out-dir /mnt/s/AITraining/review-packets
```

For router work:

```bash
python3 scripts/generate_frontier_review_packet.py \
  --lane general-ops-router \
  --queue /mnt/s/AITraining/curation/queue.jsonl \
  --dataset-dir /mnt/s/AITraining/datasets/gemma-orchestrator-v0.2-real-shadow-reviewed \
  --eval-report /mnt/s/AITraining/runs/<run>/route-eval.json \
  --out-dir /mnt/s/AITraining/review-packets
```

Outputs:

- `<packet-id>.md` — cheap human/frontier-readable summary.
- `<packet-id>.json` — structured packet with compact changed examples, low-confidence drafts, disagreements, errors, eval metrics, and raw artifact pointers.

## Optional inputs

- `--baseline-queue <path>` compares a prior queue JSONL to the current queue and summarizes added/removed/modified examples.
- `--eval-report <path>` can be repeated for baseline/adapter/frontier reports.
- `--artifact <path>` can be repeated for run directories, adapters, logs, screenshots, or source reports. Contents are not embedded; only pointers and sizes are included.
- `--include-raw-snippets` permits short snippets. Leave this off for normal cheap frontier review.
- `--max-examples` controls how many compact samples appear per section.
- `--max-markdown-bytes` warns if the Markdown packet grows past the intended budget.

## Frontier review checklist

The packet marks `frontier_review_required=true` when any checklist item trips:

- Training/adapters, production routing, config/secrets, publishing, or irreversible state changes are in scope.
- Local draft confidence is below threshold or missing for examples being promoted.
- Local/frontier/human labels disagree or eval rows disagree with expected outputs.
- Schema, parse, sanitization, or secret-like-content errors are present.
- Eval metrics miss strict schema/route/safety gates.
- The decision is subjective/taste-based or requires operator acceptance.

If none of those trip and the task is only counts/status/packet generation, keep it local/no-model.
