const state = {
  selectedJob: null,
  logOffset: -1,
  artifactRoot: 'agent-arena',
  artifactPath: '',
  selectedTask: null,
  traceLab: null,
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

function setView(viewId) {
  document.querySelectorAll('.view').forEach(view => view.classList.toggle('active', view.id === viewId));
  document.querySelectorAll('[data-view]').forEach(btn => btn.classList.toggle('active', btn.dataset.view === viewId));
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
    <div class="training-detail"><span>${esc(label)}</span><b>${esc(value)}</b></div>`).join('');
}

async function refreshStatus() {
  try {
    const data = await api('/api/status');
    const gpu = data.gpu?.gpus?.[0] || {};
    $('overall-status').innerHTML = latestVerifyBadge(data.latest_verify);
    renderTrainingState(data.training_state);
    renderTraceLab(data.trace_lab || {});
    const verify = data.latest_verify || {};
    const checks = verify.checks || {};
    $('metrics').innerHTML = [
      ['GPU', gpu.name || 'not detected'],
      ['VRAM used/free', gpu.memory_used_mb ? `${gpu.memory_used_mb}/${gpu.memory_free_mb} MB` : 'n/a'],
      ['GPU util/temp', gpu.utilization_percent ? `${gpu.utilization_percent}% / ${gpu.temperature_c}°C` : 'n/a'],
      ['Venv', data.venv_exists ? 'present' : 'missing'],
      ['S: free', data.disk?.ok ? `${data.disk.free_gb} GB` : 'unknown'],
      ['Arena', data.roots?.['agent-arena']?.exists ? 'ready' : 'missing'],
      ['Datasets', data.roots?.datasets?.exists ? 'ready' : 'missing'],
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


function renderTraceLab(traceLab) {
  state.traceLab = traceLab || {};
  const imports = traceLab.imports || [];
  const sources = traceLab.recommended_sources || [];
  const latestImport = imports[0] || {};
  $('trace-metrics').innerHTML = [
    ['Raw events', traceLab.raw_events ?? 0],
    ['Raw files', traceLab.raw_event_files ?? 0],
    ['Imports', traceLab.import_count ?? 0],
    ['Normalized rows', traceLab.normalized_rows ?? 0],
    ['Sanitized rows', traceLab.sanitized_rows ?? 0],
    ['Sources', sources.length],
    ['Latest import', latestImport.imported_at || 'none'],
    ['Trace root', traceLab.root || 'unknown'],
  ].map(([label, value]) => `<div class="metric"><span>${esc(label)}</span><b>${esc(value)}</b></div>`).join('');

  $('trace-summary-json').textContent = JSON.stringify({
    root: traceLab.root,
    raw_events: traceLab.raw_events,
    imports: imports.slice(0, 3).map(item => ({ source: item.source, imported_at: item.imported_at, row_counts: item.row_counts, destination: item.destination })),
    updated_at: traceLab.updated_at,
  }, null, 2);

  if (!sources.length) {
    $('trace-sources').innerHTML = '<p class="subtle">No source registry yet. Click “Write Source Registry”.</p>';
  } else {
    $('trace-sources').innerHTML = sources.slice(0, 8).map(source => `
      <article class="source-card">
        <div class="card-title"><h4>${esc(source.id)}</h4><span class="badge">${esc(source.priority || 'review')}</span></div>
        <p>${esc(source.why || '')}</p>
        <small>${esc(source.license || 'license unknown')} · ${esc(source.url || '')}</small>
      </article>`).join('');
  }
}

async function refreshTraces() {
  try {
    renderTraceLab(await api('/api/traces/summary'));
  } catch (err) {
    $('trace-summary-json').textContent = err.message;
  }
}

async function runTraceAction(path) {
  try {
    const job = await api(path, { method: 'POST', body: '{}' });
    state.selectedJob = job.id;
    state.logOffset = -1;
    setView('jobs-view');
    await refreshStatus();
    await refreshLog();
  } catch (err) { alert(err.message); }
}

async function runCommand(command) {
  const body = {};
  if (command === 'setup') {
    if (!confirm('Setup/repair can download and install packages into the Unsloth env. Continue?')) return;
    body.confirm = true;
  }
  if (command === 'verify') body.verifyTimeoutSeconds = Number($('verify-timeout').value || 180);
  if (command === 'smoke-train') {
    const runId = $('run-id').value.trim();
    if (runId) body.runId = runId;
    if (!confirm('Launch a GPU smoke training run? This may take time and consume VRAM.')) return;
  }
  try {
    const job = await api(`/api/commands/${command}/run`, { method: 'POST', body: JSON.stringify(body) });
    state.selectedJob = job.id;
    state.logOffset = -1;
    setView('jobs-view');
    await refreshStatus();
    await refreshLog();
  } catch (err) { alert(err.message); }
}

async function refreshArena() {
  try {
    const tasks = await api('/api/arena/tasks');
    if (!tasks.length) {
      $('arena-tasks').innerHTML = '<p class="subtle">No arena tasks yet. Create one on the left.</p>';
      return;
    }
    $('arena-tasks').innerHTML = tasks.map(task => `
      <article class="task-card ${state.selectedTask === task.id ? 'selected' : ''}" data-task="${esc(task.id)}">
        <div><b>${esc(task.title)}</b><small>${esc(task.id)}</small></div>
        <div class="task-meta"><span class="badge">${esc(task.status)}</span><span>${esc(task.candidate_count)} candidates</span>${task.has_winner ? '<span class="status-ok">winner</span>' : ''}</div>
      </article>`).join('');
    document.querySelectorAll('[data-task]').forEach(el => el.addEventListener('click', () => loadTask(el.dataset.task)));
  } catch (err) { $('arena-tasks').innerHTML = `<p class="status-bad">${esc(err.message)}</p>`; }
}

async function createTask() {
  const body = {
    title: $('arena-title').value.trim(),
    repoPath: $('arena-repo').value.trim(),
    prompt: $('arena-prompt').value.trim(),
    acceptanceCriteria: $('arena-acceptance').value.trim(),
  };
  try {
    const task = await api('/api/arena/tasks', { method: 'POST', body: JSON.stringify(body) });
    $('arena-title').value = '';
    $('arena-prompt').value = '';
    $('arena-acceptance').value = '';
    await refreshArena();
    await loadTask(task.id);
  } catch (err) { alert(err.message); }
}

async function loadTask(taskId) {
  state.selectedTask = taskId;
  const payload = await api(`/api/arena/tasks/${encodeURIComponent(taskId)}`);
  const task = payload.task;
  $('selected-task-title').textContent = task.title;
  $('selected-task-detail').innerHTML = `
    <div class="task-detail-grid">
      <div><span>Task ID</span><b>${esc(task.id)}</b></div>
      <div><span>Status</span><b>${esc(task.status)}</b></div>
      <div><span>Repo</span><b>${esc(task.repo_path)}</b></div>
      <div><span>Gate</span><b>${esc(task.promotion_gate)}</b></div>
    </div>
    <h3>Prompt</h3><pre class="prompt-box">${esc(task.prompt)}</pre>
    <h3>Acceptance criteria</h3><pre class="prompt-box">${esc(task.acceptance_criteria || 'No criteria supplied.')}</pre>`;
  $('score-task').disabled = false;
  $('add-candidate').disabled = false;
  $('run-candidate-batch').disabled = false;
  renderCandidates(payload);
  await refreshArena();
}

function renderCandidates(payload) {
  const candidates = payload.candidates || [];
  if (!candidates.length) {
    $('candidate-list').innerHTML = '<p class="subtle">No candidates yet. Paste candidate diffs/test logs from agent runs.</p>';
  } else {
    const winnerId = payload.winner?.candidate_id;
    $('candidate-list').innerHTML = candidates.map(c => {
      const score = c.scorecard?.score ?? 'n/a';
      const reasons = (c.scorecard?.reasons || []).map(esc).join(' · ');
      return `<article class="candidate-card">
        <div class="card-title"><h4>${esc(c.label)}</h4><span class="score">${esc(score)}</span></div>
        <p>${esc(c.summary || 'No summary')}</p>
        <div class="task-meta"><span class="badge">${esc(c.worker || 'manual')}</span>${c.worktree_path ? `<span>${esc(c.worktree_path)}</span>` : ''}${c.agent_log_path ? `<span>${esc(c.agent_log_path)}</span>` : ''}</div>
        <small>${esc(reasons)}</small>
        <div class="actions"><button class="secondary" data-winner="${esc(c.id)}" ${winnerId === c.id ? 'disabled' : ''}>${winnerId === c.id ? 'Winner selected' : 'Select winner'}</button></div>
      </article>`;
    }).join('');
    document.querySelectorAll('[data-winner]').forEach(btn => btn.addEventListener('click', () => selectWinner(btn.dataset.winner)));
  }
  $('arena-review').textContent = JSON.stringify(payload.review || payload.winner || { note: 'Score candidates to write a review artifact.' }, null, 2);
}

async function addCandidate() {
  if (!state.selectedTask) return;
  const body = {
    label: $('candidate-label').value.trim(),
    summary: $('candidate-summary').value.trim(),
    diff: $('candidate-diff').value,
    testLog: $('candidate-tests').value,
    reviewNotes: $('candidate-review').value,
  };
  try {
    await api(`/api/arena/tasks/${encodeURIComponent(state.selectedTask)}/candidates`, { method: 'POST', body: JSON.stringify(body) });
    ['candidate-label', 'candidate-summary', 'candidate-diff', 'candidate-tests', 'candidate-review'].forEach(id => $(id).value = '');
    await loadTask(state.selectedTask);
  } catch (err) { alert(err.message); }
}

async function scoreTask() {
  if (!state.selectedTask) return;
  try {
    await api(`/api/arena/tasks/${encodeURIComponent(state.selectedTask)}/score`, { method: 'POST', body: '{}' });
    await loadTask(state.selectedTask);
  } catch (err) { alert(err.message); }
}

async function runCandidateBatch() {
  if (!state.selectedTask) return;
  const worker = $('batch-worker').value;
  const count = Number($('batch-count').value || 3);
  const testCommand = $('batch-test-command').value.trim();
  const warning = worker === 'scripted-smoke'
    ? 'Run a safe scripted plumbing batch? This will create isolated git worktrees and import candidate diffs.'
    : `Run ${count} real ${worker} candidate agent(s)? This can spend tokens and edit isolated worktrees. No commits or merges will happen.`;
  if (!confirm(warning)) return;
  try {
    const detail = await api(`/api/arena/tasks/${encodeURIComponent(state.selectedTask)}`);
    const body = { worker, count, testCommand, repoPath: detail.task.repo_path };
    const job = await api(`/api/arena/tasks/${encodeURIComponent(state.selectedTask)}/batch`, { method: 'POST', body: JSON.stringify(body) });
    state.selectedJob = job.id;
    state.logOffset = -1;
    setView('jobs-view');
    await refreshStatus();
    await refreshLog();
  } catch (err) { alert(err.message); }
}

async function selectWinner(candidateId) {
  if (!state.selectedTask) return;
  const note = prompt('Manual note for why this candidate won. It will be written into the preference dataset.', 'Best verified candidate; keep behind manual promotion gate.');
  if (note === null) return;
  try {
    const result = await api(`/api/arena/tasks/${encodeURIComponent(state.selectedTask)}/winner`, { method: 'POST', body: JSON.stringify({ candidateId, note }) });
    alert(`Winner selected. Dataset updated: ${result.dataset_path}`);
    await loadTask(state.selectedTask);
    state.artifactRoot = 'agent-arena';
    state.artifactPath = 'datasets';
    await refreshArtifacts();
  } catch (err) { alert(err.message); }
}

async function renderJobs(jobs) {
  if (!jobs.length) { $('jobs').innerHTML = '<p class="subtle">No jobs yet.</p>'; return; }
  $('jobs').innerHTML = `<table class="table"><thead><tr><th>Job</th><th>Status</th><th>Started</th><th>Log</th><th></th></tr></thead><tbody>${jobs.map(job => `
    <tr><td><span class="linkish" data-job="${esc(job.id)}">${esc(job.label || job.command)}</span><br><small>${esc(job.id)}</small></td>
    <td><span class="badge">${esc(job.status)}</span>${job.returncode === null ? '' : `<br><small>rc ${esc(job.returncode)}</small>`}</td>
    <td>${esc(job.started_at || '')}</td><td><small>${esc(job.log_path || '')}</small></td>
    <td>${job.status === 'running' ? `<button class="secondary" data-cancel="${esc(job.id)}">Cancel</button>` : ''}</td></tr>`).join('')}</tbody></table>`;
  document.querySelectorAll('[data-job]').forEach(el => el.addEventListener('click', () => { state.selectedJob = el.dataset.job; state.logOffset = -1; refreshLog(); }));
  document.querySelectorAll('[data-cancel]').forEach(el => el.addEventListener('click', async () => {
    if (!confirm('Cancel this job?')) return;
    await api(`/api/jobs/${el.dataset.cancel}/cancel`, { method: 'POST', body: '{}' });
    await refreshStatus();
  }));
}

async function refreshJobs() { await renderJobs(await api('/api/jobs')); }

async function refreshLog() {
  if (!state.selectedJob) return;
  try {
    const data = await api(`/api/jobs/${state.selectedJob}/log?offset=${state.logOffset}`);
    $('job-log').textContent = data.text || '(empty log)';
    $('job-log').scrollTop = $('job-log').scrollHeight;
    state.logOffset = data.next_offset ?? data.offset;
  } catch (err) { $('job-log').textContent = err.message; }
}

function renderArtifactTabs() {
  const roots = ['agent-arena', 'traces', 'datasets', 'runs', 'adapters', 'merged-models'];
  $('artifact-tabs').innerHTML = roots.map(root => `<button class="secondary ${root === state.artifactRoot ? 'active' : ''}" data-root="${root}">${root}</button>`).join('');
  document.querySelectorAll('[data-root]').forEach(el => el.addEventListener('click', () => { state.artifactRoot = el.dataset.root; state.artifactPath = ''; refreshArtifacts(); }));
}

async function refreshArtifacts() {
  renderArtifactTabs();
  $('artifact-preview').textContent = 'Select a previewable text/log/json file.';
  try {
    const data = await api(`/api/artifacts?root=${encodeURIComponent(state.artifactRoot)}&path=${encodeURIComponent(state.artifactPath)}`);
    $('artifact-path').textContent = `${data.root}: /${data.path || ''}`;
    if (!data.items.length) { $('artifacts').innerHTML = '<p class="subtle">Empty.</p>'; return; }
    $('artifacts').innerHTML = data.items.map(item => `
      <div class="file-row"><span class="${item.type === 'dir' || item.previewable ? 'linkish' : ''}" data-artifact-path="${esc(item.path)}" data-artifact-type="${esc(item.type)}" data-previewable="${item.previewable ? '1' : '0'}">${item.type === 'dir' ? '📁' : '📄'} ${esc(item.name)}</span>
      <span class="badge">${esc(item.type)}</span><small>${fmtBytes(item.size)}</small></div>`).join('');
    document.querySelectorAll('[data-artifact-path]').forEach(el => el.addEventListener('click', async () => {
      if (el.dataset.artifactType === 'dir') { state.artifactPath = el.dataset.artifactPath; await refreshArtifacts(); }
      else if (el.dataset.previewable === '1') await previewArtifact(el.dataset.artifactPath);
    }));
  } catch (err) { $('artifacts').innerHTML = `<p class="status-bad">${esc(err.message)}</p>`; }
}

async function previewArtifact(path) {
  try { $('artifact-preview').textContent = (await api(`/api/artifacts/preview?root=${encodeURIComponent(state.artifactRoot)}&path=${encodeURIComponent(path)}`)).text; }
  catch (err) { $('artifact-preview').textContent = err.message; }
}

function upArtifact() {
  if (!state.artifactPath) return;
  const parts = state.artifactPath.split('/').filter(Boolean);
  parts.pop();
  state.artifactPath = parts.join('/');
  refreshArtifacts();
}

document.querySelectorAll('[data-view]').forEach(btn => btn.addEventListener('click', () => setView(btn.dataset.view)));
document.querySelectorAll('[data-run]').forEach(btn => btn.addEventListener('click', () => runCommand(btn.dataset.run)));
$('refresh-status').addEventListener('click', refreshStatus);
$('refresh-jobs').addEventListener('click', refreshJobs);
$('artifact-up').addEventListener('click', upArtifact);
$('refresh-arena').addEventListener('click', refreshArena);
$('refresh-traces').addEventListener('click', refreshTraces);
$('import-fable-traces').addEventListener('click', () => runTraceAction('/api/traces/import-fable'));
$('write-trace-sources').addEventListener('click', () => runTraceAction('/api/traces/write-sources'));
$('create-task').addEventListener('click', createTask);
$('add-candidate').addEventListener('click', addCandidate);
$('score-task').addEventListener('click', scoreTask);
$('run-candidate-batch').addEventListener('click', runCandidateBatch);

refreshStatus();
refreshArena();
refreshArtifacts();
refreshTraces();
setInterval(refreshStatus, 10000);
setInterval(() => { if (state.selectedJob) refreshLog(); }, 3000);
