/**
 * TrustLayer AI — app.js
 * Hallucination-aware RAG with trust scoring
 *
 * Architecture:
 *   User Query → Retriever → Top-K Context → Answer Generator
 *   → Verifier LLM → Trust Score → UI Output
 *
 * In this demo, responses are simulated locally.
 * Replace DEMO_RESPONSES with real API calls to your
 * backend (LangChain / LlamaIndex / OpenAI pipeline).
 */

'use strict';

// ── Constants ──────────────────────────────────────────────
const CIRCUMFERENCE = 2 * Math.PI * 44; // r=44 SVG circle

// Trust score thresholds (from PRD)
const CONFIDENCE = {
  SUPPORTED:           0.92,
  PARTIALLY_SUPPORTED: 0.61,
  NOT_SUPPORTED:       0.08,
};

// ── State ──────────────────────────────────────────────────
let mode        = 'trust'; // 'trust' | 'fast'
let isLoading   = false;
let queryCount  = 0;
let flaggedCount = 0;
let trustScores = [];
let animFrame   = null;
let typingEl    = null;

// ── Demo response database ─────────────────────────────────
// Replace these with live API calls in production.
const DEMO_RESPONSES = {
  specs: {
    answer: 'The product features a modular architecture with 4 core components: an API gateway layer, distributed processing engine, real-time analytics dashboard, and a fault-tolerant data store. It supports up to 10,000 concurrent connections and provides a 99.9% uptime SLA.',
    verdict: 'SUPPORTED',
    confidence: CONFIDENCE.SUPPORTED,
    reason: 'The answer directly maps to sections 2.1, 2.3, and 4.7 of product_specs.pdf. All claims about architecture, connection limits, and SLA are explicitly stated in the retrieved context.',
    chunks: [
      { id: 'S1', text: 'modular architecture with 4 core components: API gateway, processing engine, analytics dashboard, fault-tolerant data store' },
      { id: 'S2', text: 'supports up to 10,000 concurrent connections with 99.9% SLA guarantee' },
    ],
  },

  revenue: {
    answer: null, // withheld
    verdict: 'NOT_SUPPORTED',
    confidence: CONFIDENCE.NOT_SUPPORTED,
    reason: 'No document in the indexed corpus contains financial or revenue data for 2025. The query cannot be answered from available context. Answer withheld to prevent hallucination.',
    chunks: [],
  },

  compliance: {
    answer: 'The compliance guidelines outline three mandatory requirements: (1) all data must be encrypted at rest using AES-256, (2) access logs must be retained for a minimum of 90 days, and (3) third-party integrations require a security review before deployment.',
    verdict: 'PARTIALLY_SUPPORTED',
    confidence: CONFIDENCE.PARTIALLY_SUPPORTED,
    reason: 'Items 1 and 2 are directly supported by compliance_2024.txt. Item 3 regarding third-party review is implied but not explicitly stated — partial confidence assigned.',
    chunks: [
      { id: 'C1', text: 'all data encrypted at rest using AES-256 standard, mandatory for all environments' },
      { id: 'C2', text: 'access logs retention minimum 90 days per regulatory requirement §4.2' },
    ],
  },
};

// ── Simulated document uploads ─────────────────────────────
const FAKE_DOCS = [
  { name: 'quarterly_report.pdf',   size: '31 KB · 14 chunks' },
  { name: 'technical_spec_v2.txt',  size: '12 KB · 6 chunks'  },
  { name: 'legal_contracts.pdf',    size: '88 KB · 41 chunks' },
  { name: 'audit_log.txt',          size: '7 KB · 3 chunks'   },
  { name: 'policy_manual.pdf',      size: '55 KB · 27 chunks' },
];

// ── Mode toggle ────────────────────────────────────────────
function setMode(m) {
  mode = m;
  document.getElementById('fastBtn').classList.toggle('active', m === 'fast');
  document.getElementById('trustBtn').classList.toggle('active', m === 'trust');
}

// ── Upload simulation ──────────────────────────────────────
function simulateUpload() {
  const doc  = FAKE_DOCS[Math.floor(Math.random() * FAKE_DOCS.length)];
  const list = document.getElementById('docList');
  const div  = document.createElement('div');
  div.className = 'tl-doc-item';
  div.innerHTML = `
    <div class="tl-doc-icon">📄</div>
    <div style="flex:1;overflow:hidden;">
      <div class="tl-doc-name">${escHtml(doc.name)}</div>
      <div class="tl-doc-size">${escHtml(doc.size)}</div>
    </div>`;
  list.appendChild(div);

  // Brief flash animation
  div.style.opacity = '0';
  requestAnimationFrame(() => {
    div.style.transition = 'opacity 0.4s';
    div.style.opacity    = '1';
  });
}

// ── Input handlers ─────────────────────────────────────────
function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    submitQuery();
  }
}

function sendSample(q) {
  const input = document.getElementById('queryInput');
  input.value = q;
  submitQuery();
}

// ── Main query flow ────────────────────────────────────────
async function submitQuery() {
  if (isLoading) return;

  const input = document.getElementById('queryInput');
  const query = input.value.trim();
  if (!query) return;

  input.value = '';
  isLoading   = true;
  document.getElementById('sendBtn').disabled = true;

  // 1. Render user message
  addUserMsg(query);

  // 2. Reset analytics panel
  clearAnalytics();

  // 3. Brief delay then show typing
  await sleep(280);
  showTyping();

  // 4. Animate pipeline steps
  animatePipeline();

  // 5. Choose demo response by keyword match
  const resp     = classifyQuery(query);
  const waitTime = mode === 'fast' ? 1600 : 3000;
  await sleep(waitTime);

  // 6. Remove typing indicator, render answer
  hideTyping();
  renderAIMsg(resp);

  // 7. Update analytics
  updateAnalytics(resp);

  // 8. Update session stats
  queryCount++;
  if (resp.verdict === 'NOT_SUPPORTED') flaggedCount++;
  trustScores.push(resp.confidence);
  updateStats();

  isLoading = false;
  document.getElementById('sendBtn').disabled = false;
}

// Simple keyword classifier — replace with real RAG in production
function classifyQuery(q) {
  const ql = q.toLowerCase();
  if (ql.includes('revenue') || ql.includes('financial') || ql.includes('2025') || ql.includes('not in') || ql.includes('sales')) {
    return DEMO_RESPONSES.revenue;
  }
  if (ql.includes('compliance') || ql.includes('guidelines') || ql.includes('summarize') || ql.includes('regulatory')) {
    return DEMO_RESPONSES.compliance;
  }
  return DEMO_RESPONSES.specs;
}

// ── Chat rendering ─────────────────────────────────────────
function addUserMsg(text) {
  const chat = document.getElementById('chatArea');

  // Remove empty state on first message
  const empty = chat.querySelector('.tl-chat-empty');
  if (empty) empty.remove();

  const el = document.createElement('div');
  el.className = 'tl-msg user';
  el.innerHTML = `
    <div class="tl-msg-bubble">${escHtml(text)}</div>
    <div class="tl-msg-meta">You · Just now</div>`;
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
}

function showTyping() {
  const chat = document.getElementById('chatArea');
  typingEl = document.createElement('div');
  typingEl.className = 'tl-msg ai';
  typingEl.innerHTML = `
    <div class="tl-typing"><span></span><span></span><span></span></div>
    <div class="tl-msg-meta">TrustLayer is analyzing…</div>`;
  chat.appendChild(typingEl);
  chat.scrollTop = chat.scrollHeight;
}

function hideTyping() {
  if (typingEl) { typingEl.remove(); typingEl = null; }
}

function renderAIMsg(resp) {
  const chat = document.getElementById('chatArea');
  const el   = document.createElement('div');
  el.className = 'tl-msg ai';

  // Verdict tag HTML
  const verdictHtml = buildVerdictTag(resp.verdict);

  // Confidence badge
  const confBadge = `<span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#475569;display:inline-block;margin-top:4px;">Confidence: ${Math.round(resp.confidence * 100)}%</span>`;

  let bodyHtml;
  if (resp.answer) {
    // Normal grounded answer
    bodyHtml = `
      <div class="tl-msg-bubble">
        ${escHtml(resp.answer)}
        <br>
        ${verdictHtml}
        <br>
        ${confBadge}
      </div>`;
  } else {
    // Withheld — no data found
    bodyHtml = `
      <div class="tl-withheld">
        <div class="tl-withheld-title">⚠ Answer Withheld</div>
        <div class="tl-withheld-text">
          No reliable data found in indexed documents.<br>
          Answering from external knowledge is disabled.<br><br>
          <em>This response is not grounded in retrieved data.</em>
        </div>
        ${verdictHtml}
      </div>`;
  }

  // Warning banner for hallucination detection
  let warnHtml = '';
  if (resp.verdict === 'NOT_SUPPORTED') {
    warnHtml = `
      <div class="tl-warning-banner">
        <div class="tl-warn-icon">!</div>
        <span>Hallucination risk detected — answer withheld by TrustLayer</span>
      </div>`;
  }

  el.innerHTML = bodyHtml + warnHtml + `<div class="tl-msg-meta">TrustLayer AI · ${mode === 'trust' ? 'Trust' : 'Fast'} Mode</div>`;
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
}

function buildVerdictTag(verdict) {
  if (verdict === 'SUPPORTED') {
    return `<span class="tl-verdict-tag verdict-supported">● SUPPORTED</span>`;
  } else if (verdict === 'PARTIALLY_SUPPORTED') {
    return `<span class="tl-verdict-tag verdict-partial">● PARTIAL</span>`;
  } else {
    return `<span class="tl-verdict-tag verdict-not">● NOT SUPPORTED</span>`;
  }
}

// ── Analytics panel ────────────────────────────────────────
function clearAnalytics() {
  setTrustMeter(0, '#1E2A40', '—', 'AWAITING QUERY', '#475569');
  document.getElementById('verifyReason').textContent = 'Running verification pipeline…';
  document.getElementById('chunkArea').innerHTML = `
    <div class="tl-pipeline-step active" style="padding:4px 0;">
      <div class="tl-step-dot" style="background:#3B82F6;box-shadow:0 0 6px #3B82F6;animation:pulse 1s infinite;"></div>
      Retrieving chunks…
    </div>`;
}

function updateAnalytics(resp) {
  const pct = Math.round(resp.confidence * 100);
  let color, statusText;

  if (resp.verdict === 'SUPPORTED') {
    color = '#22C55E'; statusText = 'HIGH CONFIDENCE';
  } else if (resp.verdict === 'PARTIALLY_SUPPORTED') {
    color = '#FACC15'; statusText = 'MEDIUM TRUST';
  } else {
    color = '#EF4444'; statusText = 'LOW CONFIDENCE';
  }

  animateTrustTo(pct, color, statusText, color);
  document.getElementById('verifyReason').textContent = resp.reason;

  const ca = document.getElementById('chunkArea');
  if (resp.chunks.length === 0) {
    ca.innerHTML = `
      <div class="tl-source-chip">
        <div class="tl-source-text" style="color:#EF4444;">
          No relevant chunks retrieved. Query is outside indexed knowledge.
        </div>
      </div>`;
  } else {
    ca.innerHTML = resp.chunks.map(c => `
      <div class="tl-source-chip">
        <div class="tl-source-num">${escHtml(c.id)}</div>
        <div class="tl-source-text">
          "…<span class="tl-chunk-highlight">${escHtml(c.text)}</span>…"
        </div>
      </div>`).join('');
  }
}

// ── Trust meter animation ──────────────────────────────────
function animateTrustTo(target, color, statusText, statusColor) {
  let current = 0;
  const arc   = document.getElementById('trustArc');
  arc.style.stroke = color;
  if (animFrame) cancelAnimationFrame(animFrame);

  function step() {
    current = Math.min(current + 2, target);
    const offset = CIRCUMFERENCE - (current / 100) * CIRCUMFERENCE;
    arc.style.strokeDashoffset = offset;
    document.getElementById('trustPct').textContent = current + '%';
    document.getElementById('trustPct').style.color = color;
    document.getElementById('trustStatus').textContent = statusText;
    document.getElementById('trustStatus').style.color  = statusColor;
    if (current < target) animFrame = requestAnimationFrame(step);
  }
  animFrame = requestAnimationFrame(step);
}

function setTrustMeter(pct, color, pctText, statusText, statusColor) {
  const arc = document.getElementById('trustArc');
  arc.style.stroke = color;
  arc.style.strokeDashoffset = CIRCUMFERENCE;
  document.getElementById('trustPct').textContent     = pctText;
  document.getElementById('trustPct').style.color     = statusColor;
  document.getElementById('trustStatus').textContent  = statusText;
  document.getElementById('trustStatus').style.color  = statusColor;
}

// ── Pipeline step animator ─────────────────────────────────
async function animatePipeline() {
  const stepIds = ['step1', 'step2', 'step3', 'step4', 'step5'];
  const delay   = mode === 'fast' ? 280 : 540;

  // Reset all steps
  stepIds.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.className = 'tl-pipeline-step';
    el.querySelector('.tl-step-dot').style.cssText = '';
  });

  for (let i = 0; i < stepIds.length; i++) {
    await sleep(delay);
    const el = document.getElementById(stepIds[i]);
    if (!el) continue;
    el.className = 'tl-pipeline-step active';
    el.querySelector('.tl-step-dot').style.cssText =
      'background:#3B82F6;box-shadow:0 0 6px #3B82F6;';

    if (i > 0) {
      const prev = document.getElementById(stepIds[i - 1]);
      if (prev) {
        prev.className = 'tl-pipeline-step done';
        prev.querySelector('.tl-step-dot').style.cssText = 'background:#22C55E;';
      }
    }
  }

  // Mark last as done after final delay
  await sleep(delay);
  const last = document.getElementById(stepIds[stepIds.length - 1]);
  if (last) {
    last.className = 'tl-pipeline-step done';
    last.querySelector('.tl-step-dot').style.cssText = 'background:#22C55E;';
  }
}

// ── Session stats ──────────────────────────────────────────
function updateStats() {
  document.getElementById('statQueries').textContent  = queryCount;
  document.getElementById('statHalluc').textContent   = flaggedCount;
  if (trustScores.length > 0) {
    const avg = trustScores.reduce((a, b) => a + b, 0) / trustScores.length;
    document.getElementById('statTrust').textContent  = Math.round(avg * 100) + '%';
  }
}

// ── Utilities ──────────────────────────────────────────────
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}