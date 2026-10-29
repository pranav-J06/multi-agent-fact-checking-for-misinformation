/**
 * VeritasAI - Clean UI Application Logic
 * Only displays:
 * 1. Type claim textbox
 * 2. Verify claim button
 * 3. Atomic sub-claims and overall verdict (Zero raw numbers, scores, or latency values)
 */

function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  const div = document.createElement('div');
  div.textContent = String(str);
  return div.innerHTML;
}

// Preset loading helper
function loadPreset(claimText) {
  const textarea = document.getElementById('claim-input');
  if (textarea) {
    textarea.value = claimText;
    textarea.focus();
    textarea.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

// Clear claim input and results
function handleClear() {
  const textarea = document.getElementById('claim-input');
  if (textarea) {
    textarea.value = '';
    textarea.focus();
  }
  const resultsContainer = document.getElementById('results-container');
  if (resultsContainer) {
    resultsContainer.classList.add('hidden');
  }
  const loader = document.getElementById('loading-indicator');
  if (loader) {
    loader.classList.add('hidden');
  }
}

// Live Claim Verification
async function handleVerify() {
  const inputEl = document.getElementById('claim-input');
  const btnVerify = document.getElementById('btn-verify');
  const btnText = document.getElementById('btn-verify-text');
  const loader = document.getElementById('loading-indicator');
  const resultsContainer = document.getElementById('results-container');
  const claimText = inputEl ? inputEl.value.trim() : '';

  if (!claimText) {
    alert('Please enter a claim statement to verify.');
    if (inputEl) inputEl.focus();
    return;
  }

  // Set loading state
  if (btnVerify) btnVerify.disabled = true;
  if (btnText) btnText.textContent = 'Verifying...';
  if (loader) loader.classList.remove('hidden');
  if (resultsContainer) resultsContainer.classList.add('hidden');

  try {
    const response = await fetch('/api/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ claim: claimText })
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error || err.detail || `Server error (${response.status})`);
    }

    const data = await response.json();
    renderResults(data);

  } catch (error) {
    alert(`Verification failed: ${error.message}`);
  } finally {
    if (loader) loader.classList.add('hidden');
    if (btnVerify) btnVerify.disabled = false;
    if (btnText) btnText.textContent = 'Verify Claim';
  }
}

// Render Results Data (Strictly Qualitative - No Raw Values/Scores/Latencies)
function renderResults(data) {
  const container = document.getElementById('results-container');
  if (!container) return;
  container.classList.remove('hidden');

  // 1. Verdict Banner
  const banner = document.getElementById('verdict-banner');
  const label = data.verdict || 'NOT ENOUGH INFO';

  if (banner) {
    banner.className = 'verdict-banner';
    if (label === 'SUPPORTED') {
      banner.classList.add('verdict-supported');
    } else if (label === 'REFUTED') {
      banner.classList.add('verdict-refuted');
    } else {
      banner.classList.add('verdict-nei');
    }
  }

  const titleEl = document.getElementById('verdict-title');
  if (titleEl) titleEl.textContent = label;

  const expEl = document.getElementById('verdict-explanation');
  if (expEl) {
    // Clean explanation of any remaining numeric references like 99.8% or scores
    let cleanExp = data.explanation || 'Verification completed successfully.';
    cleanExp = cleanExp.replace(/\s+with\s+\d+(\.\d+)?%\s+confidence/gi, '');
    cleanExp = cleanExp.replace(/\(average\s+[a-z]+\s+stance:\s+\d+(\.\d+)?\)/gi, '');
    cleanExp = cleanExp.replace(/\(minimum\s+similarity:\s+\d+(\.\d+)?\)/gi, '');
    cleanExp = cleanExp.replace(/agreement\s+ratio\s+\d+(\.\d+)?/gi, 'strong consensus');
    expEl.textContent = cleanExp;
  }

  // Real Fact Box
  const realFactBox = document.getElementById('real-fact-box');
  const realFactEl = document.getElementById('verdict-real-fact');
  if (realFactBox && realFactEl) {
    if (data.real_fact) {
      realFactEl.textContent = data.real_fact;
      realFactBox.className = 'real-fact-card';
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

  // 2. Atomic Sub-Claims & Citations (Zero Numeric Values)
  const subclaimsList = document.getElementById('subclaims-container');
  if (subclaimsList) {
    subclaimsList.innerHTML = '';
    const citations = data.evidence_citations || [];

    citations.forEach((item, idx) => {
      const card = document.createElement('div');
      card.className = 'subclaim-card';

      const supScore = item.support_score || 0;
      const refScore = item.refute_score || 0;

      // Qualitative stance status (No percentages or numbers)
      let statusBadgeClass = 'badge-nei';
      let statusBadgeText = 'Insufficient Evidence';
      let stanceRationale = item.support_rationale || item.refute_rationale || 'No conclusive evidence found.';

      if (supScore >= 0.5 && supScore > refScore) {
        statusBadgeClass = 'badge-sup';
        statusBadgeText = 'Supported by Evidence';
        stanceRationale = item.support_rationale || 'Verified as accurate by evidence.';
      } else if (refScore >= 0.5 && refScore > supScore) {
        statusBadgeClass = 'badge-ref';
        statusBadgeText = 'Contradicted by Evidence';
        stanceRationale = item.refute_rationale || 'Contradicted by evidence.';
      }

      // Filter and clean citations (Strip out "(sim: 0.XX)" numbers)
      const rawCitations = item.citations || [];
      let citationsHtml = '';
      if (rawCitations.length > 0) {
        const cleanCitations = rawCitations.map(c => {
          // Remove "(sim: 0.85)" or similar score tags
          return c.replace(/\s*\(sim:\s*\d+(\.\d+)?\)/gi, '').trim();
        });

        citationsHtml = `
          <div class="citations-wrapper">
            <div class="citations-label">
              <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
              </svg>
              <span>Retrieved Evidence</span>
            </div>
            ${cleanCitations.map(c => `<div class="citation-sentence">${escapeHtml(c)}</div>`).join('')}
          </div>
        `;
      }

      card.innerHTML = `
        <div class="subclaim-header">
          <div class="subclaim-header-left">
            <span class="subclaim-badge">Sub-Claim ${idx + 1}</span>
            <span class="subclaim-status ${statusBadgeClass}">${statusBadgeText}</span>
          </div>
          <div class="subclaim-text">${escapeHtml(item.sub_claim)}</div>
        </div>

        <div class="subclaim-analysis-box">
          <div class="analysis-label">Analysis</div>
          <p class="analysis-text">${escapeHtml(stanceRationale)}</p>
        </div>

        ${citationsHtml}
      `;

      subclaimsList.appendChild(card);
    });
  }

  // Smooth scroll down to the results
  setTimeout(() => {
    container.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 100);
}

// Keyboard Shortcut: Ctrl + Enter / Cmd + Enter to verify
document.addEventListener('DOMContentLoaded', () => {
  const inputEl = document.getElementById('claim-input');
  if (inputEl) {
    inputEl.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        handleVerify();
      }
    });
  }
});
