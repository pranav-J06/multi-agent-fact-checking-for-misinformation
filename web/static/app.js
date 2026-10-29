/**
 * VeritasAI - Frontend Logic & Live Pipeline Orchestrator
 */

function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  const div = document.createElement('div');
  div.textContent = String(str);
  return div.innerHTML;
}

// Tab Switching
function switchTab(tabId) {
  document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));

  const targetPane = document.getElementById(`tab-${tabId}`);
  const targetBtn = document.getElementById(`tab-btn-${tabId}`);

  if (targetPane) targetPane.classList.add('active');
  if (targetBtn) targetBtn.classList.add('active');
}

// Quick Preset Loader
function loadPreset(claimText) {
  const textarea = document.getElementById('claim-input');
  textarea.value = claimText;
  textarea.focus();
}

// Live Claim Verification
async function handleVerify() {
  const inputEl = document.getElementById('claim-input');
  const btnVerify = document.getElementById('btn-verify');
  const stepper = document.getElementById('pipeline-stepper');
  const resultsContainer = document.getElementById('results-container');
  const claimText = inputEl.value.trim();

  if (!claimText) {
    alert('Please enter a claim to verify.');
    return;
  }

  // UI state: loading
  btnVerify.disabled = true;
  stepper.classList.remove('hidden');
  resultsContainer.classList.add('hidden');
  resetStepper();

  // Progress simulation across stages
  const stepInterval = runStepperAnimation();

  try {
    const response = await fetch('/api/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ claim: claimText })
    });

    clearInterval(stepInterval);
    finishStepper();

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || 'Verification failed');
    }

    const data = await response.json();
    renderResults(data);

  } catch (error) {
    clearInterval(stepInterval);
    document.getElementById('stepper-status').textContent = `Error: ${error.message}`;
    alert(`Error verifying claim: ${error.message}`);
  } finally {
    btnVerify.disabled = false;
  }
}

// Stepper Animation
function resetStepper() {
  for (let i = 1; i <= 6; i++) {
    const el = document.getElementById(`step-${i}`);
    el.classList.remove('active', 'completed');
  }
  document.getElementById('step-1').classList.add('active');
  document.getElementById('stepper-status').textContent = '1. Preprocessing input claim string...';
}

function runStepperAnimation() {
  let step = 1;
  const messages = [
    '1. Preprocessing input claim string [1, L]...',
    '2. Claim Extractor Agent: Decomposing into 2-6 atomic sub-claims...',
    '3. Evidence Retriever: Embedding & FAISS CPU dense vector search...',
    '4. Cross-Verifier Agent: Running dual-pass Support & Refute over Llama-3.1-8B...',
    '5. Feature Pooling: Computing 10-D statistical stance tensor...',
    '6. Verdict Aggregator Head: Executing trained MLP classification (<50ms)...'
  ];

  return setInterval(() => {
    if (step < 6) {
      document.getElementById(`step-${step}`).classList.remove('active');
      document.getElementById(`step-${step}`).classList.add('completed');
      step++;
      document.getElementById(`step-${step}`).classList.add('active');
      document.getElementById('stepper-status').textContent = messages[step - 1];
    }
  }, 1400);
}

function finishStepper() {
  for (let i = 1; i <= 6; i++) {
    const el = document.getElementById(`step-${i}`);
    el.classList.remove('active');
    el.classList.add('completed');
  }
  document.getElementById('stepper-status').textContent = 'Verification Pipeline Complete!';
}

// Render Results
function renderResults(data) {
  const container = document.getElementById('results-container');
  container.classList.remove('hidden');

  // 1. Verdict Banner
  const banner = document.getElementById('verdict-banner');
  banner.className = 'verdict-banner';
  const label = data.verdict;
  const confPercent = (data.confidence * 100).toFixed(1);

  if (label === 'SUPPORTED') {
    banner.classList.add('verdict-supported');
  } else if (label === 'REFUTED') {
    banner.classList.add('verdict-refuted');
  } else {
    banner.classList.add('verdict-nei');
  }

  document.getElementById('verdict-title').textContent = label;
  document.getElementById('verdict-conf').textContent = `${confPercent}%`;
  document.getElementById('verdict-explanation').textContent = data.explanation;

  // Real Fact Display
  const realFactBox = document.getElementById('real-fact-box');
  const realFactEl = document.getElementById('verdict-real-fact');
  if (realFactBox && realFactEl) {
    if (data.real_fact) {
      realFactEl.textContent = data.real_fact;
      realFactBox.className = 'real-fact-box';
      if (label === 'SUPPORTED') {
        realFactBox.classList.add('fact-supported');
      } else if (label === 'REFUTED') {
        realFactBox.classList.add('fact-refuted');
      } else {
        realFactBox.classList.add('fact-nei');
      }
      realFactBox.classList.remove('hidden');
    } else {
      realFactBox.classList.add('hidden');
    }
  }

  const aggMs = data.latencies.verdict_aggregator_ms || 0.6;
  const totalS = data.latencies.total_pipeline_s || 8.0;
  document.getElementById('time-aggregator').textContent = `${aggMs.toFixed(2)} ms`;
  document.getElementById('time-total').textContent = `${totalS.toFixed(2)} s`;

  // 2. Probabilities
  const probs = data.probabilities || {};
  const pSup = ((probs['SUPPORTED'] || 0) * 100).toFixed(1);
  const pRef = ((probs['REFUTED'] || 0) * 100).toFixed(1);
  const pNei = ((probs['NOT ENOUGH INFO'] || 0) * 100).toFixed(1);

  document.getElementById('bar-sup').style.width = `${pSup}%`;
  document.getElementById('val-sup').textContent = `${pSup}%`;
  document.getElementById('bar-ref').style.width = `${pRef}%`;
  document.getElementById('val-ref').textContent = `${pRef}%`;
  document.getElementById('bar-nei').style.width = `${pNei}%`;
  document.getElementById('val-nei').textContent = `${pNei}%`;

  // 3. Latencies
  const lats = data.latencies || {};
  document.getElementById('lat-ext').textContent = `${(lats.claim_extractor_s || 0).toFixed(2)}s`;
  document.getElementById('lat-ret').textContent = `${(lats.evidence_retriever_s || 0).toFixed(3)}s`;
  document.getElementById('lat-ver').textContent = `${(lats.cross_verifier_s || 0).toFixed(2)}s`;
  document.getElementById('lat-pool').textContent = `${((lats.feature_pooling_s || 0) * 1000).toFixed(1)}ms`;
  document.getElementById('lat-agg').textContent = `${aggMs.toFixed(2)}ms`;

  // 4. 10-D Feature Vector Grid
  const featGrid = document.getElementById('feature-tensor-grid');
  featGrid.innerHTML = '';
  const featDict = data.feature_vector || {};

  for (const [key, val] of Object.entries(featDict)) {
    const item = document.createElement('div');
    item.className = 'feat-item';
    const num = typeof val === 'number' ? val.toFixed(3) : val;
    item.innerHTML = `
      <span class="feat-name">${escapeHtml(key)}</span>
      <span class="feat-val">${escapeHtml(num)}</span>
    `;
    featGrid.appendChild(item);
  }

  // 5. Atomic Sub-Claims & Stances
  const subclaimsList = document.getElementById('subclaims-container');
  subclaimsList.innerHTML = '';
  const citations = data.evidence_citations || [];

  citations.forEach((item, idx) => {
    const card = document.createElement('div');
    card.className = 'subclaim-card';

    const supPercent = ((item.support_score || 0) * 100).toFixed(0);
    const refPercent = ((item.refute_score || 0) * 100).toFixed(0);

    const citationsHtml = (item.citations || []).map(c => `
      <div class="cited-evidence-block">
        <div class="cited-evidence-title">Documentary Evidence:</div>
        <div>${escapeHtml(c)}</div>
      </div>
    `).join('');

    card.innerHTML = `
      <div class="subclaim-title-row">
        <span class="subclaim-idx">Sub-claim ${idx + 1}</span>
        <div class="subclaim-text">${escapeHtml(item.sub_claim)}</div>
      </div>
      <div class="stances-comparison-grid">
        <div class="stance-box stance-box-sup">
          <div class="stance-box-header">
            <span class="stance-box-title" style="color: var(--color-sup);">Pass A: Support Stance</span>
            <span class="stance-score-val" style="color: var(--color-sup);">${supPercent}%</span>
          </div>
          <p class="stance-rationale">${escapeHtml(item.support_rationale || 'Supported by evidence.')}</p>
        </div>
        <div class="stance-box stance-box-ref">
          <div class="stance-box-header">
            <span class="stance-box-title" style="color: var(--color-ref);">Pass B: Refute Stance</span>
            <span class="stance-score-val" style="color: var(--color-ref);">${refPercent}%</span>
          </div>
          <p class="stance-rationale">${escapeHtml(item.refute_rationale || 'No contradiction found.')}</p>
        </div>
      </div>
      ${citationsHtml}
    `;

    subclaimsList.appendChild(card);
  });

  // Smooth scroll to results
  container.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Load Benchmarks Data on Initial Load
async function loadBenchmarksData() {
  try {
    const res = await fetch('/api/benchmarks');
    if (!res.ok) return;
    const data = await res.json();

    // Summary numbers
    if (data.fever_benchmark && data.fever_benchmark.metrics) {
      const acc = (data.fever_benchmark.metrics.accuracy * 100).toFixed(1);
      const f1 = data.fever_benchmark.metrics.macro_f1.toFixed(3);
      document.getElementById('bench-fever-acc').textContent = `${acc}%`;
      document.getElementById('bench-fever-f1').textContent = f1;
    }
    if (data.fever_benchmark && data.fever_benchmark.latency_stats && data.fever_benchmark.latency_stats.verdict_aggregator_ms) {
      const meanMs = data.fever_benchmark.latency_stats.verdict_aggregator_ms.mean.toFixed(2);
      document.getElementById('bench-agg-latency').textContent = `${meanMs} ms`;
    }

    // Dynamic Political Stress Test Metrics
    const liarData = data.liar_benchmark || data.political_benchmark;
    if (liarData && liarData.metrics) {
      const polAcc = (liarData.metrics.accuracy * 100).toFixed(1);
      const polF1 = liarData.metrics.macro_f1.toFixed(3);
      const polAccEl = document.getElementById('bench-political-acc');
      const polF1El = document.getElementById('bench-political-f1');
      if (polAccEl) polAccEl.textContent = `${polAcc}%`;
      if (polF1El) polF1El.textContent = polF1;
    }

    // Dynamic Confusion Matrix
    if (data.fever_benchmark && data.fever_benchmark.metrics && data.fever_benchmark.metrics.confusion_matrix) {
      const cm = data.fever_benchmark.metrics.confusion_matrix;
      const cmTbody = document.querySelector('.cm-table tbody');
      if (cmTbody && cm.length === 3) {
        cmTbody.innerHTML = `
          <tr>
            <td><strong>SUPPORTED</strong></td>
            <td class="cm-cell cm-diag">${cm[0][0]}</td>
            <td class="cm-cell">${cm[0][1]}</td>
            <td class="cm-cell">${cm[0][2]}</td>
          </tr>
          <tr>
            <td><strong>REFUTED</strong></td>
            <td class="cm-cell">${cm[1][0]}</td>
            <td class="cm-cell cm-diag">${cm[1][1]}</td>
            <td class="cm-cell">${cm[1][2]}</td>
          </tr>
          <tr>
            <td><strong>NOT ENOUGH INFO</strong></td>
            <td class="cm-cell">${cm[2][0]}</td>
            <td class="cm-cell">${cm[2][1]}</td>
            <td class="cm-cell cm-diag">${cm[2][2]}</td>
          </tr>
        `;
      }
    }

    // Baselines table
    const tableBody = document.getElementById('baseline-table-body');
    if (tableBody && data.comparative_baselines) {
      tableBody.innerHTML = '';
      data.comparative_baselines.forEach((b, idx) => {
        const tr = document.createElement('tr');
        const isProposed = idx === 0;
        if (isProposed) tr.style.background = 'rgba(56, 189, 248, 0.05)';

        const accPct = (b.fever_accuracy * 100).toFixed(1);
        const lat = b.avg_aggregator_latency_ms < 1000
          ? `${b.avg_aggregator_latency_ms.toFixed(2)} ms`
          : `${(b.avg_aggregator_latency_ms / 1000).toFixed(1)} s`;

        tr.innerHTML = `
          <td><strong>${escapeHtml(b.model_architecture)}</strong> ${isProposed ? '<span class="badge-tag">Proposed</span>' : ''}</td>
          <td>${escapeHtml(b.novelty)}</td>
          <td><strong>${accPct}%</strong></td>
          <td>${b.fever_macro_f1.toFixed(3)}</td>
          <td><span style="color: ${isProposed ? 'var(--color-sup)' : 'inherit'}; font-weight: 700;">${lat}</span></td>
          <td>${escapeHtml(b.hallucination_cascade_risk)}</td>
          <td><code>${b.trainable_parameters}</code></td>
        `;
        tableBody.appendChild(tr);
      });
    }

  } catch (e) {
    console.error('Failed to load benchmarks:', e);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  loadBenchmarksData();
});
