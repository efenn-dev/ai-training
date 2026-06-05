#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/mnt/s/AITraining/unsloth"
RUN_DIR="/mnt/s/AITraining/runs"
RESULT_JSON="$RUN_DIR/verify-latest.json"
TIMEOUT_SECONDS="${VERIFY_TIMEOUT_SECONDS:-120}"

mkdir -p "$RUN_DIR"
cd "$WORKDIR"
source .venv/bin/activate

python - <<'PY'
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

RUN_DIR = Path('/mnt/s/AITraining/runs')
RESULT_JSON = RUN_DIR / 'verify-latest.json'
TIMEOUT_SECONDS = int(os.environ.get('VERIFY_TIMEOUT_SECONDS', '120'))
RUN_DIR.mkdir(parents=True, exist_ok=True)

result: dict[str, Any] = {
    'event': 'verify_unsloth',
    'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'python': sys.version,
    'platform': platform.platform(),
    'workdir': str(Path.cwd()),
    'timeout_seconds_per_check': TIMEOUT_SECONDS,
    'checks': {},
}


def flush() -> None:
    RESULT_JSON.write_text(json.dumps(result, indent=2), encoding='utf-8')


def run_check(name: str, code: str, timeout: int | None = None) -> dict[str, Any]:
    timeout = timeout or TIMEOUT_SECONDS
    print(f'[verify_unsloth] starting {name} (timeout={timeout}s)', flush=True)
    start = time.time()
    proc = subprocess.run(
        [sys.executable, '-c', code],
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    elapsed = round(time.time() - start, 2)
    payload: dict[str, Any] = {
        'status': 'ok' if proc.returncode == 0 else 'error',
        'returncode': proc.returncode,
        'elapsed_seconds': elapsed,
        'stdout': proc.stdout.strip(),
        'stderr': proc.stderr.strip(),
    }
    print(f"[verify_unsloth] {name}: {payload['status']} in {elapsed}s", flush=True)
    result['checks'][name] = payload
    flush()
    return payload


def run_check_safe(name: str, code: str, timeout: int | None = None) -> None:
    try:
        run_check(name, code, timeout)
    except subprocess.TimeoutExpired as exc:
        payload = {
            'status': 'timeout',
            'timeout_seconds': timeout or TIMEOUT_SECONDS,
            'stdout': (exc.stdout or '').strip() if isinstance(exc.stdout, str) else '',
            'stderr': (exc.stderr or '').strip() if isinstance(exc.stderr, str) else '',
        }
        print(f"[verify_unsloth] {name}: timeout after {payload['timeout_seconds']}s", flush=True)
        result['checks'][name] = payload
        flush()
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        payload = {'status': 'exception', 'error': repr(exc)}
        print(f'[verify_unsloth] {name}: exception {exc!r}', flush=True)
        result['checks'][name] = payload
        flush()


flush()
run_check_safe('nvidia_smi', """
import json, subprocess
proc = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True, capture_output=True)
print(json.dumps({'returncode': proc.returncode, 'stdout': proc.stdout.strip(), 'stderr': proc.stderr.strip()}))
raise SystemExit(proc.returncode)
""", timeout=15)
run_check_safe('torch_cuda', """
import json, torch
payload = {'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'cuda_version': torch.version.cuda}
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    payload.update({'gpu': torch.cuda.get_device_name(0), 'vram_gb': round(props.total_memory / 1024**3, 2), 'capability': f'{props.major}.{props.minor}'})
print(json.dumps(payload))
raise SystemExit(0 if payload['cuda_available'] else 2)
""")
run_check_safe('unsloth_import', """
import json, unsloth
print(json.dumps({'unsloth': getattr(unsloth, '__version__', 'imported')}))
""")
run_check_safe('hf_stack_imports', """
import json, transformers, datasets, trl, peft
print(json.dumps({'transformers': transformers.__version__, 'datasets': datasets.__version__, 'trl': trl.__version__, 'peft': peft.__version__}))
""")

checks = result['checks']
required = ['torch_cuda', 'unsloth_import', 'hf_stack_imports']
result['completed_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
result['ok'] = all(checks.get(name, {}).get('status') == 'ok' for name in required)
flush()
print(json.dumps(result, indent=2), flush=True)
if not result['ok']:
    raise SystemExit('AI training verification failed; see ' + str(RESULT_JSON))
PY
