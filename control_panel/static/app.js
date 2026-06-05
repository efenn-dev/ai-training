const state = {
  selectedJob: null,
  logOffset: -1,
  artifactRoot: 'runs',
  artifactPath: '',
  logTimer: null,
};

const $ = (id) => document.getElementById(id);

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function fmtBytes(bytes) {
  const n = Number(bytes || 0);
  if (n > 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
  if (n > 1024 ** 2) return `${(n / 1024 ** 2).toFixed(2)} MB`;
  if (n > 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${n} B`;
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : null; } catch { data = { text }; }
  if (!res.ok) throw new Error(data?.error || `${res.status} ${res.statusText}`);
  return data;
}

function latestVerifyBadge(latest) {
  if (!latest) return '<span class="status-warn">No verify run yet</span>';
  if (latest.ok) return '<span class="status-ok">Verify OK</span>';
  return '<span class="status-bad">Verify needs attention</span>';
}

function renderTrainingState(training) {
  const payload = training || {};
  const stateName = payload.state || 'unknown';
  const titleMap = {
    idle: 'No training running',
    verifying: 'Verify running',
    training: 'Training running',
    setup: 'Setup / repair running',
    failed: 'Last training failed',
    unknown: 'State unknown',
  };
  $('training-state-title').textContent = titleMap[stateName] || 'State unknown';
  $('training-state-chip').textContent = stateName;
  $('training-state-chip').className = `state-chip state-${stateName}`;
  $('training-state-summary').textContent = payload.summary || 'No training summary available.';

  const latestEval = payload.latest_eval || {};
  const gates = latestEval.gates || {};
  const gateText = Object.entries(gates).length
    ? Object.entries(gates).map(([name, gate]) => `${name}: ${gate.passed ? 'pass' : 'fail'}${gate.actual !== undefined ? ` (${gate.actual})` : ''}`).join(' · ')
    : 'No eval gates found';

  const detailRows = [
    ['Active job', payload.active_job_id || 'none'],
    ['Last adapter', payload.last_training_adapter || 'none'],
    ['Last train log', payload.last_training_log || 'none'],
    ['Latest eval', latestEval.path || 'none'],
    ['Eval gates', gateText],
    ['Next action', payload.next_recommended_action || 'Review curated data before training.'],
  ];
  $('training-state-details').innerHTML = detailRows.map(([label, value]) => `
    <div class="training-detail">
      <span>${esc(label)}</span>
      <b>${esc(value)}</b>
    </div>`).join('');
}

async function refreshStatus() {
  try {
    const data = await api('/api/status');
    const gpu = data.gpu?.gpus?.[0] || {};
    $('overall-status').innerHTML = latestVerifyBadge(data.latest_verify);
    renderTrainingState(data.training_state);
    const verify = data.latest_verify || {};
    const checks = verify.checks || {};
    $('metrics').innerHTML = [
      ['GPU', gpu.name || 'not detected'],
      ['VRAM used/free', gpu.memory_used_mb ? `${gpu.memory_used_mb}/${gpu.memory_free_mb} MB` : 'n/a'],
      ['GPU util/temp', gpu.utilization_percent ? `${gpu.utilization_percent}% / ${gpu.temperature_c}°C` : 'n/a'],
      ['Venv', data.venv_exists ? 'present' : 'missing'],
      ['S: free', data.disk?.ok ? `${data.disk.free_gb} GB` : 'unknown'],
      ['Datasets', data.roots?.datasets?.exists ? 'ready' : 'missing'],
      ['Runs', data.roots?.runs?.exists ? 'ready' : 'missing'],
      ['Adapters', data.roots?.adapters?.exists ? 'ready' : 'missing'],
    ].map(([label, value]) => `<div class="metric"><span>${esc(label)}</span><b>${esc(value)}</b></div>`).join('');

    const summary = {
      latestVerifyOk: verify.ok ?? null,
      startedAt: verify.started_at,
      completedAt: verify.completed_at,
      checks: Object.fromEntries(Object.entries(checks).map(([k, v]) => [k, { status: v.status, elapsed_seconds: v.elapsed_seconds, returncode: v.returncode }])),
    };
    $('verify-summary').textContent = JSON.stringify(summary, null, 2);
    await renderJobs(data.jobs || []);
  } catch (err) {
    $('overall-status').innerHTML = `<span class="status-bad">${esc(err.message)}</span>`;
  }
}

async function runCommand(command) {
  const body = {};
  if (command === 'setup') {
    const ok = confirm('Setup/repair can download and install packages into the Unsloth env. Continue?');
    if (!ok) return;
    body.confirm = true;
  }
  if (command === 'verify') {
    body.verifyTimeoutSeconds = Number($('verify-timeout').value || 180);
  }
  if (command === 'smoke-train') {
    const runId = $('run-id').value.trim();
    if (runId) body.runId = runId;
    const ok = confirm('Launch a GPU smoke training run? This may take time and consume VRAM.');
    if (!ok) return;
  }
  try {
    const job = await api(`/api/commands/${command}/run`, { method: 'POST', body: JSON.stringify(body) });
    state.selectedJob = job.id;
    state.logOffset = -1;
    await refreshStatus();
    await refreshLog();
  } catch (err) {
    alert(err.message);
  }
}

async function renderJobs(jobs) {
  if (!jobs.length) {
    $('jobs').innerHTML = '<p class="subtle">No jobs yet.</p>';
    return;
  }
  $('jobs').innerHTML = `<table class="table"><thead><tr><th>Job</th><th>Status</th><th>Started</th><th>Log</th><th></th></tr></thead><tbody>${jobs.map(job => `
    <tr>
      <td><span class="linkish" data-job="${esc(job.id)}">${esc(job.label || job.command)}</span><br><small>${esc(job.id)}</small></td>
      <td><span class="badge">${esc(job.status)}</span>${job.returncode === null ? '' : `<br><small>rc ${esc(job.returncode)}</small>`}</td>
      <td>${esc(job.started_at || '')}</td>
      <td><small>${esc(job.log_path || '')}</small></td>
      <td>${job.status === 'running' ? `<button class="secondary" data-cancel="${esc(job.id)}">Cancel</button>` : ''}</td>
    </tr>`).join('')}</tbody></table>`;

  document.querySelectorAll('[data-job]').forEach(el => el.addEventListener('click', () => {
    state.selectedJob = el.dataset.job;
    state.logOffset = -1;
    refreshLog();
  }));
  document.querySelectorAll('[data-cancel]').forEach(el => el.addEventListener('click', async () => {
    if (!confirm('Cancel this job?')) return;
    await api(`/api/jobs/${el.dataset.cancel}/cancel`, { method: 'POST', body: '{}' });
    await refreshStatus();
  }));
}

async function refreshJobs() {
  const jobs = await api('/api/jobs');
  await renderJobs(jobs);
}

async function refreshLog() {
  if (!state.selectedJob) return;
  try {
    const data = await api(`/api/jobs/${state.selectedJob}/log?offset=${state.logOffset}`);
    $('job-log').textContent = data.text || '(empty log)';
    $('job-log').scrollTop = $('job-log').scrollHeight;
    state.logOffset = data.next_offset ?? data.offset;
  } catch (err) {
    $('job-log').textContent = err.message;
  }
}

function renderArtifactTabs() {
  const roots = ['datasets', 'runs', 'adapters', 'merged-models'];
  $('artifact-tabs').innerHTML = roots.map(root => `<button class="secondary ${root === state.artifactRoot ? 'active' : ''}" data-root="${root}">${root}</button>`).join('');
  document.querySelectorAll('[data-root]').forEach(el => el.addEventListener('click', () => {
    state.artifactRoot = el.dataset.root;
    state.artifactPath = '';
    refreshArtifacts();
  }));
}

async function refreshArtifacts() {
  renderArtifactTabs();
  $('artifact-preview').textContent = 'Select a previewable text/log/json file.';
  try {
    const data = await api(`/api/artifacts?root=${encodeURIComponent(state.artifactRoot)}&path=${encodeURIComponent(state.artifactPath)}`);
    $('artifact-path').textContent = `${data.root}: /${data.path || ''}`;
    if (!data.items.length) {
      $('artifacts').innerHTML = '<p class="subtle">Empty.</p>';
      return;
    }
    $('artifacts').innerHTML = data.items.map(item => `
      <div class="file-row">
        <span class="${item.type === 'dir' || item.previewable ? 'linkish' : ''}" data-artifact-path="${esc(item.path)}" data-artifact-type="${esc(item.type)}" data-previewable="${item.previewable ? '1' : '0'}">${item.type === 'dir' ? '📁' : '📄'} ${esc(item.name)}</span>
        <span class="badge">${esc(item.type)}</span>
        <small>${fmtBytes(item.size)}</small>
      </div>`).join('');
    document.querySelectorAll('[data-artifact-path]').forEach(el => el.addEventListener('click', async () => {
      if (el.dataset.artifactType === 'dir') {
        state.artifactPath = el.dataset.artifactPath;
        await refreshArtifacts();
      } else if (el.dataset.previewable === '1') {
        await previewArtifact(el.dataset.artifactPath);
      }
    }));
  } catch (err) {
    $('artifacts').innerHTML = `<p class="status-bad">${esc(err.message)}</p>`;
  }
}

async function previewArtifact(path) {
  try {
    const data = await api(`/api/artifacts/preview?root=${encodeURIComponent(state.artifactRoot)}&path=${encodeURIComponent(path)}`);
    $('artifact-preview').textContent = data.text;
  } catch (err) {
    $('artifact-preview').textContent = err.message;
  }
}

function upArtifact() {
  if (!state.artifactPath) return;
  const parts = state.artifactPath.split('/').filter(Boolean);
  parts.pop();
  state.artifactPath = parts.join('/');
  refreshArtifacts();
}

document.querySelectorAll('[data-run]').forEach(btn => btn.addEventListener('click', () => runCommand(btn.dataset.run)));
$('refresh-status').addEventListener('click', refreshStatus);
$('refresh-jobs').addEventListener('click', refreshJobs);
$('artifact-up').addEventListener('click', upArtifact);

refreshStatus();
refreshArtifacts();
setInterval(refreshStatus, 10000);
setInterval(() => { if (state.selectedJob) refreshLog(); }, 3000);
