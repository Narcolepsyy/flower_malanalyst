"""
Interactive FL Dashboard - Run experiments with configuration from the UI.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, Response, jsonify, request

app = Flask(__name__)

# State
RESULTS_PATH = Path("state/experiment_results.json")
EXPLANATION_PATH = Path("state/explanations.json")
EXPERIMENT_STATUS = {"running": False, "current": None, "log": []}


def load_explanations() -> Dict[str, Any]:
    default = {"round": None, "model_type": None, "top_features": []}
    if not EXPLANATION_PATH.exists():
        return default
    try:
        data = json.loads(EXPLANATION_PATH.read_text())
        for k, v in default.items():
            data.setdefault(k, v)
        return data
    except json.JSONDecodeError:
        return default


def load_results() -> List[Dict[str, Any]]:
    if not RESULTS_PATH.exists():
        return []
    try:
        return json.loads(RESULTS_PATH.read_text())
    except json.JSONDecodeError:
        return []


def run_experiment_thread(config: Dict):
    """Run experiment in background thread."""
    global EXPERIMENT_STATUS
    EXPERIMENT_STATUS["running"] = True
    EXPERIMENT_STATUS["current"] = config
    EXPERIMENT_STATUS["log"] = [f"Starting experiment with {config['agg_method']}..."]
    
    try:
        # Build command
        cmd = [
            sys.executable, "run_single_experiment.py",
            "--agg-method", config["agg_method"],
            "--num-rounds", str(config["num_rounds"]),
            "--num-clients", str(config["num_clients"]),
            "--model", config["model"],
            "--partition-method", config["partition_method"],
        ]
        if config["partition_method"] == "noniid":
            cmd.extend(["--noniid-alpha", str(config["noniid_alpha"])])
        
        EXPERIMENT_STATUS["log"].append(f"Running: {' '.join(cmd)}")
        
        # Run subprocess
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        
        EXPERIMENT_STATUS["log"].append(result.stdout)
        if result.returncode != 0:
            EXPERIMENT_STATUS["log"].append(f"Error: {result.stderr}")
        else:
            EXPERIMENT_STATUS["log"].append("Experiment completed successfully!")
            
            # Auto-generate explanations after successful experiment
            EXPERIMENT_STATUS["log"].append("Generating feature explanations...")
            try:
                explain_cmd = [
                    sys.executable, "explain.py",
                    "--model-weights", f"state/model_{config['agg_method']}.npz",
                    "--model", config["model"] if config["model"] != "dp-mlp" else "mlp",
                    "--top-k", "12",
                    "--output", "state/explanations.json",
                ]
                explain_result = subprocess.run(
                    explain_cmd, capture_output=True, text=True, timeout=60
                )
                if explain_result.returncode == 0:
                    EXPERIMENT_STATUS["log"].append("Explanations generated successfully!")
                else:
                    EXPERIMENT_STATUS["log"].append(f"Explain error: {explain_result.stderr}")
            except Exception as ex:
                EXPERIMENT_STATUS["log"].append(f"Explain exception: {str(ex)}")
            
    except Exception as e:
        EXPERIMENT_STATUS["log"].append(f"Exception: {str(e)}")
    finally:
        EXPERIMENT_STATUS["running"] = False


@app.route("/api/results")
def api_results() -> Response:
    return jsonify(load_results())


@app.route("/api/status")
def api_status() -> Response:
    return jsonify(EXPERIMENT_STATUS)


@app.route("/api/explanations")
def api_explanations() -> Response:
    return jsonify(load_explanations())


@app.route("/api/generate_explanations", methods=["POST"])
def api_generate_explanations() -> Response:
    """Manually trigger explanation generation."""
    if EXPERIMENT_STATUS["running"]:
        return jsonify({"error": "Experiment running, please wait"}), 400
    
    try:
        # Find ALL model files and pick the most recently modified
        model_files = list(Path("state").glob("model_*.npz"))
        latest_model = Path("state/latest_model.npz")
        if latest_model.exists():
            model_files.append(latest_model)
        
        if not model_files:
            return jsonify({"error": "No model weights found"}), 404
        
        # Use the most recently modified model
        model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        
        cmd = [
            sys.executable, "explain.py",
            "--model-weights", str(model_path),
            "--top-k", "12",
            "--output", "state/explanations.json",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            return jsonify({"status": "success", "message": f"Explanations generated from {model_path.name}"})
        else:
            return jsonify({"error": result.stderr}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/run", methods=["POST"])
def api_run() -> Response:
    if EXPERIMENT_STATUS["running"]:
        return jsonify({"error": "Experiment already running"}), 400
    
    config = request.json
    thread = threading.Thread(target=run_experiment_thread, args=(config,))
    thread.start()
    
    return jsonify({"status": "started", "config": config})


@app.route("/")
def index() -> Response:
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>FL Interactive Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" onload="window.plotlyReady=true;"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js" onload="window.chartjsReady=true;"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { 
      font-family: 'Segoe UI', sans-serif; 
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); 
      color: #e2e8f0; 
      min-height: 100vh;
      padding: 20px;
    }
    .container { max-width: 1400px; margin: 0 auto; }
    h1 { 
      text-align: center;
      font-size: 26px;
      background: linear-gradient(90deg, #38bdf8, #a3e635);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 20px;
    }
    
    .grid { display: grid; grid-template-columns: 320px 1fr; gap: 20px; }
    
    /* Config Panel */
    .config-panel {
      background: #1e293b;
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    .config-title { 
      font-size: 16px; 
      font-weight: 600; 
      color: #38bdf8; 
      margin-bottom: 16px;
      border-bottom: 1px solid #334155;
      padding-bottom: 8px;
    }
    .form-group { margin-bottom: 16px; }
    .form-group label { 
      display: block; 
      font-size: 13px; 
      color: #94a3b8; 
      margin-bottom: 6px;
    }
    .form-group select, .form-group input {
      width: 100%;
      padding: 10px 12px;
      background: #0f172a;
      border: 1px solid #334155;
      border-radius: 8px;
      color: #e2e8f0;
      font-size: 14px;
    }
    .form-group select:focus, .form-group input:focus {
      outline: none;
      border-color: #38bdf8;
    }
    .btn {
      width: 100%;
      padding: 12px;
      background: linear-gradient(90deg, #0ea5e9, #22c55e);
      border: none;
      border-radius: 8px;
      color: white;
      font-weight: 600;
      cursor: pointer;
      font-size: 14px;
      transition: transform 0.1s, opacity 0.2s;
    }
    .btn:hover { transform: translateY(-1px); }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    
    .status-box {
      margin-top: 16px;
      padding: 12px;
      background: #0f172a;
      border-radius: 8px;
      font-size: 12px;
      max-height: 150px;
      overflow-y: auto;
    }
    .status-running { color: #fbbf24; }
    .status-idle { color: #22c55e; }
    
    /* Results Panel */
    .results-panel {
      background: #1e293b;
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    
    .summary-table {
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 20px;
      font-size: 13px;
    }
    .summary-table th {
      background: linear-gradient(90deg, #0ea5e9, #22c55e);
      color: white;
      padding: 10px;
      text-align: center;
    }
    .summary-table td {
      padding: 10px;
      text-align: center;
      border-bottom: 1px solid #334155;
    }
    .summary-table tr:hover td { background: #334155; }
    .method-badge {
      display: inline-block;
      padding: 3px 10px;
      border-radius: 12px;
      font-weight: 600;
      font-size: 12px;
    }
    .fedavg { background: #38bdf8; color: #0f172a; }
    .median { background: #a3e635; color: #0f172a; }
    .krum { background: #fbbf24; color: #0f172a; }
    .trimmed { background: #f472b6; color: #0f172a; }
    .best { color: #22c55e; font-weight: 700; }
    
    .charts-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 16px;
    }
    .chart-card { 
      background: #0f172a; 
      border-radius: 8px; 
      padding: 12px;
    }
    .chart-title { 
      font-size: 13px; 
      color: #94a3b8; 
      margin-bottom: 8px;
    }
    
    .noniid-group { display: none; }
    .noniid-group.visible { display: block; }
    
    /* Explainability Panel */
    .explain-section {
      margin-top: 20px;
      background: #1e293b;
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    .explain-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      flex-wrap: wrap;
      gap: 12px;
    }
    .explain-title {
      font-size: 18px;
      font-weight: 600;
      background: linear-gradient(90deg, #f472b6, #c084fc);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .explain-meta { color: #94a3b8; font-size: 13px; }
    .explain-content {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }
    @media (max-width: 900px) {
      .explain-content { grid-template-columns: 1fr; }
    }
    .explain-chart-card { background: #0f172a; border-radius: 8px; padding: 12px; }
    .explain-table-card { background: #0f172a; border-radius: 8px; padding: 12px; overflow-x: auto; }
    .explain-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    .explain-table th {
      background: linear-gradient(90deg, #f472b6, #c084fc);
      color: white;
      padding: 8px;
      text-align: left;
    }
    .explain-table td {
      padding: 8px;
      border-bottom: 1px solid #334155;
    }
    .explain-table tr:hover td { background: #334155; }
    .btn-small {
      padding: 8px 14px;
      background: linear-gradient(90deg, #f472b6, #c084fc);
      border: none;
      border-radius: 6px;
      color: white;
      font-weight: 500;
      cursor: pointer;
      font-size: 12px;
      transition: transform 0.1s;
    }
    .btn-small:hover { transform: translateY(-1px); }
    .btn-small:disabled { opacity: 0.5; cursor: not-allowed; }
    .explain-empty { color: #94a3b8; text-align: center; padding: 20px; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Federated Learning Malware Detection</h1>
    
    <div class="grid">
      <div class="config-panel">
        <div class="config-title">Experiment Configuration</div>
        
        <div class="form-group">
          <label>Aggregation Method</label>
          <select id="agg-method">
            <option value="fedavg">FedAvg (Standard)</option>
            <option value="median">Median (Byzantine-resilient)</option>
            <option value="krum">Krum (Byzantine-fault tolerant)</option>
            <option value="trimmed">Trimmed Mean</option>
          </select>
        </div>
        
        <div class="form-group">
          <label>Model Type</label>
          <select id="model">
            <option value="logreg">Logistic Regression (Fast)</option>
            <option value="mlp">MLP Neural Network</option>
            <option value="catboost">CatBoost (Gradient Boosting)</option>
            <option value="hybrid-quantum">Hybrid Quantum (Slow)</option>
          </select>
        </div>
        
        <div class="form-group">
          <label>Number of Rounds</label>
          <input type="number" id="num-rounds" value="5" min="1" max="50">
        </div>
        
        <div class="form-group">
          <label>Number of Clients</label>
          <input type="number" id="num-clients" value="2" min="1" max="10">
        </div>
        
        <div class="form-group">
          <label>Data Partition</label>
          <select id="partition-method" onchange="toggleNonIID()">
            <option value="iid">IID (Balanced)</option>
            <option value="noniid">Non-IID (Heterogeneous)</option>
          </select>
        </div>
        
        <div class="form-group noniid-group" id="noniid-group">
          <label>Non-IID Alpha (lower = more skewed)</label>
          <input type="number" id="noniid-alpha" value="0.5" min="0.01" max="100" step="0.1">
        </div>
        
        <button class="btn" id="run-btn" onclick="runExperiment()">
          Run Experiment
        </button>
        
        <div class="status-box" id="status-box">
          <div class="status-idle">Ready to run experiments.</div>
        </div>
      </div>
      
      <div class="results-panel">
        <table class="summary-table" id="summary-table">
          <thead>
            <tr>
              <th>Method</th>
              <th>Accuracy</th>
              <th>F1</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>Loss</th>
            </tr>
          </thead>
          <tbody id="summary-body"></tbody>
        </table>
        
        <div class="charts-grid">
          <div class="chart-card">
            <div class="chart-title">Accuracy Over Rounds</div>
            <div id="acc-chart"></div>
          </div>
          <div class="chart-card">
            <div class="chart-title">Loss Over Rounds</div>
            <div id="loss-chart"></div>
          </div>
          <div class="chart-card">
            <div class="chart-title">F1 Score Over Rounds</div>
            <div id="f1-chart"></div>
          </div>
          <div class="chart-card">
            <div class="chart-title">Final Comparison</div>
            <div id="bar-chart"></div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Explainability Section -->
    <div class="explain-section">
      <div class="explain-header">
        <div>
          <div class="explain-title">Feature Explainability (XAI)</div>
          <div class="explain-meta" id="explain-meta">Awaiting explanations...</div>
        </div>
        <button class="btn-small" id="gen-explain-btn" onclick="generateExplanations()">
          Generate Explanations
        </button>
      </div>
      <div class="explain-content" id="explain-content">
        <div class="explain-chart-card">
          <canvas id="explain-chart" style="max-height: 400px;"></canvas>
        </div>
        <div class="explain-table-card">
          <table class="explain-table" id="explain-table" style="display:none;">
            <thead>
              <tr>
                <th>#</th>
                <th>Feature</th>
                <th>Importance</th>
                <th>Relative</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
          <div class="explain-empty" id="explain-empty">
            No explanations yet. Run an experiment or click "Generate Explanations".
          </div>
        </div>
      </div>
    </div>
    
    <!-- Model Comparison Section -->
    <div class="explain-section" style="margin-top: 1.5rem;">
      <div class="explain-header">
        <div>
          <div class="explain-title">Model Comparison</div>
          <div class="explain-meta">FL vs Centralized • Quantum vs Classical</div>
        </div>
      </div>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; padding: 1rem;">
        <div style="background: #1e293b; padding: 1rem; border-radius: 8px;">
          <h4 style="margin: 0 0 0.75rem 0; color: #38bdf8;">Available Models</h4>
          <table style="width: 100%; font-size: 0.85rem;">
            <tr><td>Logistic Regression</td><td style="text-align:right; color: #a3e635;">99.81%</td></tr>
            <tr><td>MLP Neural Network</td><td style="text-align:right; color: #a3e635;">99.8%+</td></tr>
            <tr><td>CatBoost (Best)</td><td style="text-align:right; color: #fbbf24;">99.96%</td></tr>
            <tr><td>Hybrid Quantum</td><td style="text-align:right; color: #f472b6;">99.01%</td></tr>
          </table>
        </div>
        <div style="background: #1e293b; padding: 1rem; border-radius: 8px;">
          <h4 style="margin: 0 0 0.75rem 0; color: #f472b6;">Quantum: FL vs Centralized</h4>
          <table style="width: 100%; font-size: 0.85rem;">
            <tr><td>Centralized (30 epochs)</td><td style="text-align:right; color: #94a3b8;">~93%</td></tr>
            <tr><td>Federated (5 rounds)</td><td style="text-align:right; color: #a3e635;">99.01%</td></tr>
            <tr><td colspan="2" style="color: #38bdf8; padding-top: 0.5rem;">
              ✅ FL is 6% more accurate with fewer iterations
            </td></tr>
          </table>
        </div>
      </div>
    </div>
  </div>

  <script>
    const colors = { fedavg: '#38bdf8', median: '#a3e635', krum: '#fbbf24', trimmed: '#f472b6' };
    const baseLayout = {
      paper_bgcolor: '#0f172a',
      plot_bgcolor: '#0f172a',
      font: { color: '#e2e8f0', size: 11 },
      margin: { l: 40, r: 20, t: 20, b: 40 },
      legend: { orientation: 'h', y: -0.2 },
      xaxis: { gridcolor: '#334155' },
      yaxis: { gridcolor: '#334155' }
    };
    
    function toggleNonIID() {
      const method = document.getElementById('partition-method').value;
      document.getElementById('noniid-group').classList.toggle('visible', method === 'noniid');
    }
    
    async function runExperiment() {
      const btn = document.getElementById('run-btn');
      const statusBox = document.getElementById('status-box');
      
      const config = {
        agg_method: document.getElementById('agg-method').value,
        model: document.getElementById('model').value,
        num_rounds: parseInt(document.getElementById('num-rounds').value),
        num_clients: parseInt(document.getElementById('num-clients').value),
        partition_method: document.getElementById('partition-method').value,
        noniid_alpha: parseFloat(document.getElementById('noniid-alpha').value),
      };
      
      btn.disabled = true;
      btn.textContent = 'Running...';
      statusBox.innerHTML = '<div class="status-running">Starting experiment...</div>';
      
      try {
        const res = await fetch('/api/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(config)
        });
        
        if (!res.ok) {
          const err = await res.json();
          statusBox.innerHTML = `<div style="color:#ef4444">Error: ${err.error}</div>`;
          return;
        }
        
        // Poll for status
        const poll = setInterval(async () => {
          const statusRes = await fetch('/api/status');
          const status = await statusRes.json();
          
          statusBox.innerHTML = status.log.map(l => `<div>${l}</div>`).join('');
          statusBox.scrollTop = statusBox.scrollHeight;
          
          if (!status.running) {
            clearInterval(poll);
            btn.disabled = false;
            btn.textContent = 'Run Experiment';
            loadData();
          }
        }, 1000);
        
      } catch (e) {
        statusBox.innerHTML = `<div style="color:#ef4444">Error: ${e.message}</div>`;
        btn.disabled = false;
        btn.textContent = 'Run Experiment';
      }
    }
    
    async function loadData() {
      try {
        const res = await fetch('/api/results');
        const data = await res.json();
        
        if (!data || data.length === 0) {
          document.getElementById('summary-body').innerHTML = 
            '<tr><td colspan="6" style="color:#94a3b8">No results yet. Run an experiment!</td></tr>';
          return;
        }
        
        // Find best values
        let bestAcc = 0, bestF1 = 0;
        data.forEach(d => {
          if (d.accuracy && d.accuracy.length > 0) {
            const acc = d.accuracy[d.accuracy.length - 1];
            const f1 = d.f1[d.f1.length - 1];
            if (acc > bestAcc) bestAcc = acc;
            if (f1 > bestF1) bestF1 = f1;
          }
        });
        
        // Populate table
        const tbody = document.getElementById('summary-body');
        tbody.innerHTML = '';
        data.forEach(d => {
          if (!d.accuracy || d.accuracy.length === 0) return;
          const lastAcc = d.accuracy[d.accuracy.length - 1];
          const lastF1 = d.f1[d.f1.length - 1];
          const lastPrec = d.precision[d.precision.length - 1];
          const lastRecall = d.recall[d.recall.length - 1];
          const lastLoss = d.loss[d.loss.length - 1];
          
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td><span class="method-badge ${d.method}">${d.method.toUpperCase()}</span></td>
            <td class="${lastAcc >= bestAcc ? 'best' : ''}">${(lastAcc * 100).toFixed(2)}%</td>
            <td class="${lastF1 >= bestF1 ? 'best' : ''}">${(lastF1 * 100).toFixed(2)}%</td>
            <td>${(lastPrec * 100).toFixed(2)}%</td>
            <td>${(lastRecall * 100).toFixed(2)}%</td>
            <td>${lastLoss.toFixed(4)}</td>
          `;
          tbody.appendChild(tr);
        });
        
        // Draw charts
        const validData = data.filter(d => d.rounds && d.rounds.length > 0);
        if (validData.length === 0) return;
        
        const accTraces = validData.map(d => ({
          x: d.rounds, y: d.accuracy.map(v => v * 100),
          type: 'scatter', mode: 'lines+markers', name: d.method.toUpperCase(),
          line: { color: colors[d.method] || '#888', width: 2 }, marker: { size: 6 }
        }));
        Plotly.newPlot('acc-chart', accTraces, { ...baseLayout, yaxis: { ...baseLayout.yaxis, title: '%' } }, {displayModeBar: false});
        
        const lossTraces = validData.map(d => ({
          x: d.rounds, y: d.loss,
          type: 'scatter', mode: 'lines+markers', name: d.method.toUpperCase(),
          line: { color: colors[d.method] || '#888', width: 2 }, marker: { size: 6 }
        }));
        Plotly.newPlot('loss-chart', lossTraces, baseLayout, {displayModeBar: false});
        
        const f1Traces = validData.map(d => ({
          x: d.rounds, y: d.f1.map(v => v * 100),
          type: 'scatter', mode: 'lines+markers', name: d.method.toUpperCase(),
          line: { color: colors[d.method] || '#888', width: 2 }, marker: { size: 6 }
        }));
        Plotly.newPlot('f1-chart', f1Traces, { ...baseLayout, yaxis: { ...baseLayout.yaxis, title: '%' } }, {displayModeBar: false});
        
        // Bar chart
        const methods = validData.map(d => d.method.toUpperCase());
        const barTraces = [
          { x: methods, y: validData.map(d => d.accuracy[d.accuracy.length - 1] * 100), name: 'Acc', type: 'bar', marker: { color: '#38bdf8' } },
          { x: methods, y: validData.map(d => d.f1[d.f1.length - 1] * 100), name: 'F1', type: 'bar', marker: { color: '#a3e635' } },
        ];
        Plotly.newPlot('bar-chart', barTraces, { ...baseLayout, barmode: 'group', xaxis: { type: 'category' }, yaxis: { range: [99, 100.1] } }, {displayModeBar: false});
        
      } catch (e) {
        console.error(e);
      }
    }
    
    // loadData() is now called in initAll after Plotly is ready
    
    // Explainability functions
    let lastExplainData = null;
    let explainChart = null;
    
    async function loadExplanations() {
      try {
        const res = await fetch('/api/explanations');
        const data = await res.json();
        const features = data.top_features || [];
        const explainTable = document.getElementById('explain-table');
        const explainEmpty = document.getElementById('explain-empty');
        const explainMeta = document.getElementById('explain-meta');
        
        if (features.length === 0) {
          explainTable.style.display = 'none';
          explainEmpty.style.display = 'block';
          explainMeta.textContent = 'Awaiting explanations...';
          if (explainChart) {
            explainChart.destroy();
            explainChart = null;
          }
          lastExplainData = null;
          return;
        }
        
        // Check if data has changed to avoid unnecessary re-renders
        const dataHash = JSON.stringify(features);
        if (dataHash === lastExplainData) {
          return; // Data unchanged, skip re-render
        }
        lastExplainData = dataHash;
        
        // Update metadata
        const round = data.round === null ? 'latest' : data.round;
        const method = data.method || 'auto';
        explainMeta.textContent = `Model: ${data.model_type || 'unknown'} | Round: ${round} | Method: ${method}`;
        
        // Build table
        const tbody = explainTable.querySelector('tbody');
        tbody.innerHTML = '';
        const maxScore = features.reduce((m, f) => Math.max(m, Math.abs(f.score || 0)), 0);
        features.forEach((f, idx) => {
          const tr = document.createElement('tr');
          const sci = Number(f.score || 0).toExponential(2);
          const rel = maxScore > 0 ? ((Math.abs(f.score) / maxScore) * 100).toFixed(1) : '0.0';
          tr.innerHTML = `<td>${idx + 1}</td><td>${f.feature}</td><td>${sci}</td><td>${rel}%</td>`;
          tbody.appendChild(tr);
        });
        explainTable.style.display = 'table';
        explainEmpty.style.display = 'none';
        
        // Chart.js horizontal bar chart
        const ctx = document.getElementById('explain-chart');
        const labels = features.map(f => f.feature).reverse();
        const relativeScores = features.map(f => maxScore > 0 ? (Math.abs(f.score) / maxScore) * 100 : 0).reverse();
        const barColors = labels.map((_, i) => `hsl(${300 + (i * 8) % 60}, 70%, 60%)`);
        
        if (explainChart) {
          explainChart.destroy();
        }
        
        explainChart = new Chart(ctx, {
          type: 'bar',
          data: {
            labels: labels,
            datasets: [{
              label: 'Relative Importance (%)',
              data: relativeScores,
              backgroundColor: barColors,
              borderColor: barColors,
              borderWidth: 1
            }]
          },
          options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              title: {
                display: true,
                text: 'Relative Feature Importances',
                color: '#e2e8f0',
                font: { size: 14 }
              },
              legend: { display: false }
            },
            scales: {
              x: {
                beginAtZero: true,
                max: 105,
                title: { display: true, text: 'Relative Importance (%)', color: '#94a3b8' },
                grid: { color: '#334155' },
                ticks: { color: '#e2e8f0' }
              },
              y: {
                grid: { color: '#334155' },
                ticks: { color: '#e2e8f0', font: { size: 11 } }
              }
            }
          }
        });
        
      } catch (e) {
        console.error('Error loading explanations:', e);
      }
    }
    
    async function generateExplanations() {
      const btn = document.getElementById('gen-explain-btn');
      btn.disabled = true;
      btn.textContent = 'Generating...';
      
      try {
        const res = await fetch('/api/generate_explanations', { method: 'POST' });
        const data = await res.json();
        
        if (!res.ok) {
          alert('Error: ' + (data.error || 'Failed to generate explanations'));
        } else {
          // Reset cache to force re-render
          lastExplainData = null;
          loadExplanations();
        }
      } catch (e) {
        alert('Error: ' + e.message);
      } finally {
        btn.disabled = false;
        btn.textContent = 'Generate Explanations';
      }
    }
    
    // Ensure libraries are fully ready before loading charts
    function waitForLibraries(callback) {
      if (typeof Plotly !== 'undefined' && window.plotlyReady && 
          typeof Chart !== 'undefined' && window.chartjsReady) {
        callback();
      } else {
        setTimeout(() => waitForLibraries(callback), 50);
      }
    }
    
    function initAll() {
      waitForLibraries(() => {
        loadData();
        loadExplanations();
        setInterval(loadExplanations, 5000);
      });
    }
    
    // Start initialization when DOM is ready
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', initAll);
    } else {
      initAll();
    }
  </script>
</body>
</html>
"""
    return Response(html, mimetype="text/html")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8503)
    args = parser.parse_args()
    print(f"Interactive Dashboard running at http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
