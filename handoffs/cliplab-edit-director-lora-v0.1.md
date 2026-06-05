# Handoff: ClipLab Edit Director LoRA v0.1

Created: 2026-05-29 13:03 CDT
Status: parked until ClipLab MVP is complete
App/session lane: AI Training / Unsloth (`ai-training`)

## User intent

Emmitt wants to use the Unsloth environment for practical product training after ClipLab MVP is finished. Candidate domains discussed:

1. Video game mechanics
2. Video editing techniques / highlights
3. Stock/trading data

Decision for first real training project: **ClipLab Edit Director LoRA v0.1**.

## Recommendation

Start with a text/JSON edit-director model, not raw video training.

Architecture:

```text
video/audio/pipeline features -> structured text/JSON summary -> fine-tuned LLM -> strict JSON edit/highlight decision
```

Do not start with raw MP4 -> LLM. Unsloth fine-tunes text LLMs unless/until we deliberately do a vision-language model project.

## First model

Initial fast iteration model:

- Qwen2.5-3B-Instruct, 4-bit QLoRA

If dataset quality is good and evals justify it, rerun with:

- Qwen2.5-7B-Instruct or Qwen3-8B, 4-bit QLoRA

Avoid starting with 7B+ before proving the dataset/eval loop.

## Dataset target

Dataset folder:

```text
S:\AITraining\datasets\cliplab-edit-director-v0.1
/mnt/s/AITraining/datasets/cliplab-edit-director-v0.1
```

Suggested files:

- `README.md`
- `schema.json`
- `train.jsonl`
- `eval.jsonl`
- `examples_raw/`
- `exported_candidates/`
- `labels/`

Training format: chat JSONL.

Example shape:

```json
{"messages":[{"role":"system","content":"You are ClipLab's edit director. Given gameplay event summaries, transcript cues, and candidate clips, choose the best highlights and return strict JSON."},{"role":"user","content":"{... pipeline candidate summary ...}"},{"role":"assistant","content":"{\"selected\":[{\"start\":123.4,\"end\":151.2,\"score\":0.91,\"reason\":\"funny clutch reaction plus clear audio spike\"}],\"reject\":[],\"edit_notes\":[]}"}]}
```

## Data sources

Use existing ClipLab/gaming-pipeline outputs when MVP is stable:

- gaming-pipeline stage 03 highlight candidates
- stage 03b edit-director outputs
- transcripts
- scene/cut/event summaries
- export manifests
- manually corrected keep/reject labels from Emmitt

Important: do not only train on current pipeline outputs or the model will mimic existing mistakes. Need human labels/corrections.

## Labels to collect

For each candidate clip or segment:

- keep / reject
- corrected start/end time
- highlight reason
- rejection reason when rejected
- platform fit: Shorts, YouTube, TikTok, longform
- edit notes: trim, caption emphasis, meme timing, zoom, intro/outro
- family-safe/child-safe notes when relevant

## Output target

Strict JSON edit/highlight decisions compatible with ClipLab's edit plan / timeline direction.

Should include:

- selected clips
- rejected clips
- scores
- reasons
- corrected trim points
- platform fit
- caption/emphasis notes
- edit pacing notes

## Evaluation set

Keep held-out examples separate from training.

Evaluate on whether the model:

- selects good highlights
- rejects dead air / false positives
- adjusts trim boundaries usefully
- returns valid schema JSON
- preserves family-safe constraints
- explains decisions cleanly
- beats current heuristic/prompt baseline

## Trading-data caution

Stock/trading training should be a later separate lane, not combined with ClipLab.

If pursued, train only setup classification / risk labels / journaling style. Do not train an autonomous prediction or buy/sell model.

## Environment already installed

Unsloth workspace:

```text
S:\AITraining\unsloth
/mnt/s/AITraining/unsloth
```

Activation:

```bash
source /mnt/s/AITraining/unsloth/activate.sh
```

Verification:

```bash
/mnt/c/projects/ai-training/scripts/verify_unsloth.sh
```

Known caveat: Flash Attention 2 reports broken; Unsloth falls back to Xformers. Acceptable for first LoRA experiments.

## Resume prompt

When ClipLab MVP is complete, resume with:

```text
Read C:\projects\ai-training\handoffs\cliplab-edit-director-lora-v0.1.md and C:\projects\ai-training\README.md. Build the ClipLab Edit Director LoRA v0.1 dataset scaffold under S:\AITraining\datasets\cliplab-edit-director-v0.1. Do not train yet. Create schema/train/eval placeholders and an exporter plan from gaming-pipeline outputs, then ask for/prepare human labeling workflow.
```
