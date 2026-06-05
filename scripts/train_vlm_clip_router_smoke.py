import unsloth
import json, os, time
from pathlib import Path
from PIL import Image
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastVisionModel, is_bfloat16_supported
from unsloth.trainer import UnslothVisionDataCollator

DATASET_DIR = Path(os.environ.get('DATASET_DIR', '/mnt/s/AITraining/datasets/vlm-clip-router-v0.1'))
MODEL_NAME = os.environ.get('VLM_BASE_MODEL', 'unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit')
RUN_ID = os.environ.get('RUN_ID') or 'vlm-clip-router-smoke-' + time.strftime('%Y%m%d-%H%M%S')
OUT_DIR = Path(os.environ.get('OUT_DIR', f'/mnt/s/AITraining/adapters/{RUN_ID}'))
MAX_STEPS = int(os.environ.get('MAX_STEPS', '4'))
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_rows(path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        if not line.strip(): continue
        item=json.loads(line)
        img=Image.open(item['image']).convert('RGB')
        user=item['messages'][0]['content']
        assistant=item['messages'][1]['content']
        rows.append({'messages':[{'role':'user','content':[{'type':'image','image':img},{'type':'text','text':user}]},{'role':'assistant','content':[{'type':'text','text':assistant}]}]})
    return rows

print(json.dumps({'event':'load_vlm','model':MODEL_NAME,'dataset':str(DATASET_DIR),'out':str(OUT_DIR),'max_steps':MAX_STEPS}, indent=2))
model, tokenizer = FastVisionModel.from_pretrained(model_name=MODEL_NAME, load_in_4bit=True, use_gradient_checkpointing='unsloth')
model = FastVisionModel.get_peft_model(model, finetune_vision_layers=True, finetune_language_layers=True, finetune_attention_modules=True, finetune_mlp_modules=True, r=8, lora_alpha=16, lora_dropout=0, bias='none', random_state=3407)
train_ds=Dataset.from_list(load_rows(DATASET_DIR/'train.jsonl'))
args=SFTConfig(output_dir=str(OUT_DIR), per_device_train_batch_size=1, gradient_accumulation_steps=2, warmup_steps=1, max_steps=MAX_STEPS, learning_rate=2e-4, logging_steps=1, optim='adamw_8bit', weight_decay=0.01, lr_scheduler_type='linear', seed=3407, fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(), report_to='none', remove_unused_columns=False, dataset_text_field='', dataset_kwargs={'skip_prepare_dataset': True}, save_steps=MAX_STEPS)
trainer=SFTTrainer(model=model, processing_class=tokenizer, data_collator=UnslothVisionDataCollator(model, tokenizer), train_dataset=train_ds, args=args)
trainer.train()
model.save_pretrained(str(OUT_DIR/'adapter'))
tokenizer.save_pretrained(str(OUT_DIR/'adapter'))
(OUT_DIR/'RUN_SUMMARY.json').write_text(json.dumps({'model':MODEL_NAME,'dataset':str(DATASET_DIR),'max_steps':MAX_STEPS,'adapter':str(OUT_DIR/'adapter')}, indent=2))
print(json.dumps({'event':'done','adapter':str(OUT_DIR/'adapter')}, indent=2))
