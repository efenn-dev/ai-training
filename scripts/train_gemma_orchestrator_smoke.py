import unsloth
import json, os, subprocess, sys, time
from pathlib import Path
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel, is_bfloat16_supported
from gemma_route_policy import validate_route

DATASET_DIR = Path(os.environ.get('DATASET_DIR', '/mnt/s/AITraining/datasets/gemma-orchestrator-v0.1'))
MODEL_NAME = os.environ.get('BASE_MODEL', 'unsloth/gemma-3-1b-it-unsloth-bnb-4bit')
RUN_ID = os.environ.get('RUN_ID') or 'gemma-orchestrator-smoke-' + time.strftime('%Y%m%d-%H%M%S')
OUT_DIR = Path(os.environ.get('OUT_DIR', f'/mnt/s/AITraining/adapters/{RUN_ID}'))
MAX_STEPS = int(os.environ.get('MAX_STEPS', '8'))
RUN_EVAL_GATE = os.environ.get('RUN_EVAL_GATE', '1').lower() not in {'0', 'false', 'no'}
OUT_DIR.mkdir(parents=True, exist_ok=True)

def validate_dataset(path: Path):
    bad = []
    for i, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        assistant = json.loads(row['messages'][2]['content'])
        error = validate_route(assistant)
        if error:
            bad.append({'line': i, 'error': error, 'assistant': assistant})
    if bad:
        raise SystemExit(f'{path} failed strict route schema validation: {json.dumps(bad[:5], ensure_ascii=False)}')

def format_example(ex):
    messages = ex['messages']
    system = messages[0]['content']
    user = messages[1]['content']
    assistant = messages[2]['content']
    text = "<start_of_turn>user\n" + system + "\n\n" + user + "<end_of_turn>\n<start_of_turn>model\n" + assistant + "<end_of_turn>"
    return {'text': text}

print(json.dumps({'event':'load_model','model':MODEL_NAME,'dataset':str(DATASET_DIR),'out':str(OUT_DIR),'max_steps':MAX_STEPS}, indent=2))
validate_dataset(DATASET_DIR/'train.jsonl')
validate_dataset(DATASET_DIR/'eval.jsonl')
model, tokenizer = FastLanguageModel.from_pretrained(model_name=MODEL_NAME, max_seq_length=2048, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=8, target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'], lora_alpha=16, lora_dropout=0, bias='none', use_gradient_checkpointing='unsloth', random_state=3407)
raw = load_dataset('json', data_files={'train': str(DATASET_DIR/'train.jsonl'), 'eval': str(DATASET_DIR/'eval.jsonl')})
train_ds = raw['train'].map(format_example, remove_columns=raw['train'].column_names)
eval_ds = raw['eval'].map(format_example, remove_columns=raw['eval'].column_names)
args = SFTConfig(output_dir=str(OUT_DIR), per_device_train_batch_size=1, gradient_accumulation_steps=2, warmup_steps=1, max_steps=MAX_STEPS, learning_rate=2e-4, logging_steps=1, optim='adamw_8bit', weight_decay=0.01, lr_scheduler_type='linear', seed=3407, fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(), report_to='none', dataset_text_field='text', save_steps=MAX_STEPS)
trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=train_ds, eval_dataset=eval_ds, args=args)
trainer.train()
model.save_pretrained(str(OUT_DIR/'adapter'))
tokenizer.save_pretrained(str(OUT_DIR/'adapter'))
summary = {'model':MODEL_NAME,'dataset':str(DATASET_DIR),'max_steps':MAX_STEPS,'adapter':str(OUT_DIR/'adapter')}
if RUN_EVAL_GATE:
    gate_run_dir = OUT_DIR / 'eval-gate'
    cmd = [
        sys.executable,
        str(Path(__file__).with_name('compare_gemma_orchestrator_adapter.py')),
        '--dataset-dir',
        str(DATASET_DIR),
        '--base-model',
        MODEL_NAME,
        '--adapter-dir',
        str(OUT_DIR/'adapter'),
        '--run-dir',
        str(gate_run_dir),
    ]
    subprocess.run(cmd, check=True)
    summary['eval_gate_report'] = str(gate_run_dir / 'comparison-report.md')
(OUT_DIR/'RUN_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps({'event':'done','adapter':str(OUT_DIR/'adapter')}, indent=2))
