/* ============================================================
   GridSearchCV Dashboard — script.js
   ============================================================ */

'use strict';

// ─── DATA (from Python GridSearchCV results) ────────────────
const DATA = {
  "default_results": {
    "SVM": {
      "accuracy": 0.9825, "precision": 0.9861, "recall": 0.9861,
      "f1_score": 0.9861, "auc_roc": 0.9950, "train_time": 0.011
    },
    "Random Forest": {
      "accuracy": 0.9561, "precision": 0.9589, "recall": 0.9722,
      "f1_score": 0.9655, "auc_roc": 0.9939, "train_time": 0.137
    },
    "Logistic Regression": {
      "accuracy": 0.9825, "precision": 0.9861, "recall": 0.9861,
      "f1_score": 0.9861, "auc_roc": 0.9954, "train_time": 0.022
    }
  },
  "tuned_results": {
    "SVM": {
      "accuracy": 0.9825, "precision": 0.9861, "recall": 0.9861,
      "f1_score": 0.9861, "auc_roc": 0.9937, "train_time": 5.363,
      "best_params": { "C": "0.1", "gamma": "scale", "kernel": "linear" },
      "best_cv_score": 0.9845, "cv_results_count": 32
    },
    "Random Forest": {
      "accuracy": 0.9474, "precision": 0.9583, "recall": 0.9583,
      "f1_score": 0.9583, "auc_roc": 0.9940, "train_time": 22.289,
      "best_params": { "max_depth": "None", "min_samples_leaf": "2", "min_samples_split": "5", "n_estimators": "50" },
      "best_cv_score": 0.9685, "cv_results_count": 108
    },
    "Logistic Regression": {
      "accuracy": 0.9737, "precision": 0.9726, "recall": 0.9861,
      "f1_score": 0.9793, "auc_roc": 0.9957, "train_time": 2.571,
      "best_params": { "C": "0.1", "penalty": "l2", "solver": "saga" },
      "best_cv_score": 0.9845, "cv_results_count": 20
    }
  },
  "improvements": {
    "SVM": 0.00,
    "Random Forest": -0.74,
    "Logistic Regression": -0.69
  },
  "param_grids": {
    "SVM": 32,
    "Random Forest": 108,
    "Logistic Regression": 20
  }
};

const COLORS = {
  green:  '#1eff00',
  cyan:   '#00e5ff',
  yellow: '#ffed4e',
  red:    '#ff3366',
  purple: '#bd00ff'
};

// ─── STATE ──────────────────────────────────────────────────
let currentModel = 'SVM';
let charts = {};

// ─── INIT ───────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  updateHeaderStats();
  updateGridStatus();
  initModelSelector();
  updateComparison(currentModel);
  renderAllCharts();
  buildParamsTable();
  buildPerfTable();
  initCopyButton();
});

// ─── HEADER STATS ───────────────────────────────────────────
function updateHeaderStats() {
  const bestModel = Object.keys(DATA.tuned_results).reduce((a, b) =>
    DATA.tuned_results[a].f1_score > DATA.tuned_results[b].f1_score ? a : b
  );
  const bestF1 = DATA.tuned_results[bestModel].f1_score;

  document.getElementById('bestModelHeader').textContent = bestModel;
  document.getElementById('bestF1Header').textContent = bestF1.toFixed(4);
}

// ─── GRID STATUS ────────────────────────────────────────────
function updateGridStatus() {
  const totalCombos = Object.values(DATA.param_grids).reduce((a, b) => a + b, 0);
  const totalTime = Object.values(DATA.tuned_results).reduce((sum, r) => sum + r.train_time, 0);
  const bestImprove = Math.max(...Object.values(DATA.improvements));

  document.getElementById('totalCombinations').textContent = totalCombos;
  document.getElementById('totalTime').textContent = totalTime.toFixed(1) + 's';
  document.getElementById('bestImprovement').textContent = (bestImprove >= 0 ? '+' : '') + bestImprove.toFixed(2) + '%';
}

// ─── MODEL SELECTOR ─────────────────────────────────────────
function initModelSelector() {
  const btns = document.querySelectorAll('.model-btn');
  btns.forEach(btn => {
    btn.addEventListener('click', () => {
      btns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentModel = btn.dataset.model;
      updateComparison(currentModel);
    });
  });
}

// ─── COMPARISON CARDS ───────────────────────────────────────
function updateComparison(model) {
  const def = DATA.default_results[model];
  const tuned = DATA.tuned_results[model];

  // Default metrics
  document.getElementById('defaultAcc').textContent = (def.accuracy * 100).toFixed(2) + '%';
  document.getElementById('defaultPrec').textContent = (def.precision * 100).toFixed(2) + '%';
  document.getElementById('defaultRec').textContent = (def.recall * 100).toFixed(2) + '%';
  document.getElementById('defaultF1').textContent = (def.f1_score * 100).toFixed(2) + '%';
  document.getElementById('defaultAUC').textContent = (def.auc_roc * 100).toFixed(2) + '%';
  document.getElementById('defaultTime').textContent = def.train_time.toFixed(3) + 's';

  // Tuned metrics
  document.getElementById('tunedAcc').textContent = (tuned.accuracy * 100).toFixed(2) + '%';
  document.getElementById('tunedPrec').textContent = (tuned.precision * 100).toFixed(2) + '%';
  document.getElementById('tunedRec').textContent = (tuned.recall * 100).toFixed(2) + '%';
  document.getElementById('tunedF1').textContent = (tuned.f1_score * 100).toFixed(2) + '%';
  document.getElementById('tunedAUC').textContent = (tuned.auc_roc * 100).toFixed(2) + '%';
  document.getElementById('cvScore').textContent = (tuned.best_cv_score * 100).toFixed(2) + '%';

  // Deltas
  updateDelta('deltaAcc', def.accuracy, tuned.accuracy);
  updateDelta('deltaPrec', def.precision, tuned.precision);
  updateDelta('deltaRec', def.recall, tuned.recall);
  updateDelta('deltaF1', def.f1_score, tuned.f1_score);
  updateDelta('deltaAUC', def.auc_roc, tuned.auc_roc);

  // Best params
  const paramsDiv = document.getElementById('bestParams');
  paramsDiv.innerHTML = Object.entries(tuned.best_params)
    .map(([k, v]) => `<div class="param-tag best">${k}: ${v}</div>`)
    .join('');
}

function updateDelta(id, oldVal, newVal) {
  const delta = ((newVal - oldVal) / oldVal) * 100;
  const el = document.getElementById(id);
  el.textContent = (delta >= 0 ? '+' : '') + delta.toFixed(2) + '%';
  el.className = 'ml-delta';
  if (delta > 0.1) el.classList.add('positive');
  else if (delta < -0.1) el.classList.add('negative');
  else el.classList.add('neutral');
}

// ─── CHARTS ─────────────────────────────────────────────────
function renderAllCharts() {
  renderMetricsChart();
  renderDeltaChart();
  renderAllModelsChart();
}

function renderMetricsChart() {
  const ctx = document.getElementById('metricsChart').getContext('2d');
  const model = currentModel;
  const def = DATA.default_results[model];
  const tuned = DATA.tuned_results[model];

  if (charts.metrics) charts.metrics.destroy();
  charts.metrics = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
      datasets: [
        {
          label: 'Default',
          data: [def.accuracy, def.precision, def.recall, def.f1_score, def.auc_roc].map(v => v * 100),
          borderColor: COLORS.cyan,
          backgroundColor: COLORS.cyan + '22',
          pointBackgroundColor: COLORS.cyan,
          borderWidth: 2
        },
        {
          label: 'Tuned',
          data: [tuned.accuracy, tuned.precision, tuned.recall, tuned.f1_score, tuned.auc_roc].map(v => v * 100),
          borderColor: COLORS.green,
          backgroundColor: COLORS.green + '22',
          pointBackgroundColor: COLORS.green,
          borderWidth: 2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#e0ffe0', font: { family: 'JetBrains Mono', size: 10 } }
        }
      },
      scales: {
        r: {
          min: 93, max: 100,
          ticks: {
            color: '#5eff5e88',
            font: { family: 'JetBrains Mono', size: 9 },
            backdropColor: 'transparent',
            callback: v => v + '%'
          },
          grid: { color: 'rgba(30,255,0,.15)' },
          pointLabels: { color: '#5eff5e88', font: { family: 'JetBrains Mono', size: 9 } }
        }
      }
    }
  });
}

function renderDeltaChart() {
  const ctx = document.getElementById('deltaChart').getContext('2d');
  const models = Object.keys(DATA.improvements);
  const improvements = models.map(m => DATA.improvements[m]);

  if (charts.delta) charts.delta.destroy();
  charts.delta = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: models,
      datasets: [{
        label: 'F1-Score Improvement (%)',
        data: improvements,
        backgroundColor: improvements.map(v => v >= 0 ? COLORS.green + 'aa' : COLORS.red + 'aa'),
        borderColor: improvements.map(v => v >= 0 ? COLORS.green : COLORS.red),
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          ticks: { color: '#5eff5e88', font: { family: 'JetBrains Mono', size: 9 } },
          grid: { display: false }
        },
        y: {
          ticks: {
            color: '#5eff5e88',
            font: { family: 'JetBrains Mono', size: 9 },
            callback: v => v + '%'
          },
          grid: { color: 'rgba(30,255,0,.08)' }
        }
      }
    }
  });
}

function renderAllModelsChart() {
  const ctx = document.getElementById('allModelsChart').getContext('2d');
  const models = Object.keys(DATA.default_results);

  if (charts.allModels) charts.allModels.destroy();
  charts.allModels = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: models,
      datasets: [
        {
          label: 'Default F1',
          data: models.map(m => DATA.default_results[m].f1_score * 100),
          backgroundColor: COLORS.cyan + 'aa',
          borderColor: COLORS.cyan,
          borderWidth: 1
        },
        {
          label: 'Tuned F1',
          data: models.map(m => DATA.tuned_results[m].f1_score * 100),
          backgroundColor: COLORS.green + 'aa',
          borderColor: COLORS.green,
          borderWidth: 1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#e0ffe0', font: { family: 'JetBrains Mono', size: 10 } }
        }
      },
      scales: {
        x: {
          ticks: { color: '#5eff5e88', font: { family: 'JetBrains Mono', size: 10 } },
          grid: { display: false }
        },
        y: {
          min: 94,
          ticks: {
            color: '#5eff5e88',
            font: { family: 'JetBrains Mono', size: 9 },
            callback: v => v + '%'
          },
          grid: { color: 'rgba(30,255,0,.08)' }
        }
      }
    }
  });
}

// ─── PARAMS TABLE ───────────────────────────────────────────
function buildParamsTable() {
  const tbody = document.getElementById('paramsTableBody');
  const models = Object.keys(DATA.tuned_results);

  models.forEach(model => {
    const params = DATA.tuned_results[model].best_params;
    Object.entries(params).forEach(([param, value]) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${model}</td>
        <td style="color:${COLORS.cyan}">${param}</td>
        <td style="color:#6b7280">—</td>
        <td style="color:${COLORS.green};font-weight:600">${value}</td>
        <td class="impact-high">HIGH</td>
      `;
      tbody.appendChild(tr);
    });
  });
}

// ─── PERFORMANCE TABLE ──────────────────────────────────────
function buildPerfTable() {
  const tbody = document.getElementById('perfTableBody');
  const models = Object.keys(DATA.default_results);

  models.forEach(model => {
    const def = DATA.default_results[model];
    const tuned = DATA.tuned_results[model];

    // Default row
    const tr1 = document.createElement('tr');
    tr1.innerHTML = `
      <td style="font-weight:600">${model}</td>
      <td class="status-default">DEFAULT</td>
      <td>${(def.accuracy * 100).toFixed(2)}%</td>
      <td>${(def.precision * 100).toFixed(2)}%</td>
      <td>${(def.recall * 100).toFixed(2)}%</td>
      <td>${(def.f1_score * 100).toFixed(2)}%</td>
      <td>${(def.auc_roc * 100).toFixed(2)}%</td>
      <td>${def.train_time.toFixed(3)}</td>
      <td>—</td>
    `;
    tbody.appendChild(tr1);

    // Tuned row
    const tr2 = document.createElement('tr');
    tr2.innerHTML = `
      <td style="font-weight:600">${model}</td>
      <td class="status-tuned">TUNED</td>
      <td>${(tuned.accuracy * 100).toFixed(2)}%</td>
      <td>${(tuned.precision * 100).toFixed(2)}%</td>
      <td>${(tuned.recall * 100).toFixed(2)}%</td>
      <td style="color:${COLORS.green};font-weight:600">${(tuned.f1_score * 100).toFixed(2)}%</td>
      <td>${(tuned.auc_roc * 100).toFixed(2)}%</td>
      <td>${tuned.train_time.toFixed(2)}</td>
      <td style="color:${COLORS.green};font-size:.7rem">CV: ${(tuned.best_cv_score * 100).toFixed(2)}%</td>
    `;
    tbody.appendChild(tr2);
  });
}

// ─── COPY BUTTON ────────────────────────────────────────────
function initCopyButton() {
  const btn = document.getElementById('copyBtn');
  const code = document.getElementById('codeBlock');

  btn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(code.textContent);
      btn.textContent = 'COPIED';
      setTimeout(() => { btn.textContent = 'COPY'; }, 2000);
    } catch {
      btn.textContent = 'FAILED';
      setTimeout(() => { btn.textContent = 'COPY'; }, 1500);
    }
  });
}