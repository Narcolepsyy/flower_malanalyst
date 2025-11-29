"""
Flask-based dashboard to visualize metrics written by the Flower server.
Serves a small SPA that polls /api/metrics every 2 seconds and renders
loss/accuracy charts with Plotly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from flask import Flask, Response, jsonify

app = Flask(__name__)

# Defaults; can be overridden in main()
METRICS_PATH = Path("state/metrics.json")
EXPLANATION_PATH = Path("state/explanations.json")


def load_metrics() -> Dict[str, Any]:
    empty = {"rounds": [], "loss": [], "accuracy": [], "precision": [], "recall": [], "f1": []}
    if not METRICS_PATH.exists():
        return empty
    try:
        data = json.loads(METRICS_PATH.read_text())
        for k, v in empty.items():
            data.setdefault(k, v)
        return data
    except json.JSONDecodeError:
        return empty


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


@app.route("/api/metrics")
def api_metrics() -> Response:
    return jsonify(load_metrics())


@app.route("/api/explanations")
def api_explanations() -> Response:
    return jsonify(load_explanations())


@app.route("/")
def index() -> Response:
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Federated Malware Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 24px; background: #0f172a; color: #e2e8f0; }}
    h1 {{ margin-top: 0; }}
    .cards {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
    .card {{ background: #1e293b; padding: 16px 20px; border-radius: 10px; min-width: 180px; box-shadow: 0 8px 20px rgba(0,0,0,0.3); }}
    #charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    .chart {{ background: #1e293b; padding: 12px; border-radius: 10px; box-shadow: 0 8px 20px rgba(0,0,0,0.25); }}
    .explain {{ margin-top: 20px; background: #1e293b; padding: 16px; border-radius: 10px; box-shadow: 0 8px 20px rgba(0,0,0,0.25); }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #334155; }}
    th {{ color: #cbd5e1; font-weight: 600; }}
  </style>
</head>
<body>
  <h1> Federated Malware Detection</h1>
  <p>Polling metrics from <code>{METRICS_PATH}</code> every 2 seconds.</p>
  <div class="cards">
    <div class="card"><div>Latest Round</div><div id="round-val" style="font-size:24px;font-weight:bold;">—</div></div>
    <div class="card"><div>Accuracy</div><div id="acc-val" style="font-size:24px;font-weight:bold;">—</div></div>
    <div class="card"><div>F1</div><div id="f1-val" style="font-size:24px;font-weight:bold;">—</div></div>
    <div class="card"><div>Model</div><div id="model-val" style="font-size:20px;font-weight:bold;">—</div></div>
  </div>
  <div id="charts">
    <div class="chart"><div id="loss-chart"></div></div>
    <div class="chart"><div id="acc-chart"></div></div>
    <div class="chart"><div id="f1-chart"></div></div>
  </div>
  <div class="explain">
    <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
      <div>
        <div style="font-size:18px;font-weight:600;">Explainability</div>
        <div id="explain-meta" style="color:#cbd5e1;">Awaiting explanations...</div>
      </div>
      <div style="color:#cbd5e1;">Polling <code>{EXPLANATION_PATH}</code> every 5 seconds.</div>
    </div>
    <table id="explain-table" style="display:none;">
      <thead><tr><th>#</th><th>Feature</th><th>Importance (sci)</th><th>Rel (%)</th></tr></thead>
      <tbody></tbody>
    </table>
    <div id="explain-empty" style="margin-top:8px;color:#94a3b8;">Run <code>python explain.py</code> after training to populate feature attributions.</div>
  </div>
  <script>
    const roundEl = document.getElementById('round-val');
    const accEl = document.getElementById('acc-val');
    const f1El = document.getElementById('f1-val');
    const modelEl = document.getElementById('model-val');
    const lossDiv = document.getElementById('loss-chart');
    const accDiv = document.getElementById('acc-chart');
    const f1Div = document.getElementById('f1-chart');
    const explainTable = document.getElementById('explain-table');
    const explainBody = explainTable.querySelector('tbody');
    const explainEmpty = document.getElementById('explain-empty');
    const explainMeta = document.getElementById('explain-meta');

    async function fetchMetrics() {{
      try {{
        const res = await fetch('/api/metrics');
        const data = await res.json();
        const rounds = data.rounds || [];
        const loss = data.loss || [];
        const acc = data.accuracy || [];
        const f1 = data.f1 || [];

        if (rounds.length > 0) {{
          roundEl.textContent = rounds[rounds.length - 1];
          const latestAcc = acc.length > 0 ? acc[acc.length - 1] : 0;
          accEl.textContent = latestAcc.toFixed(4);
          const latestF1 = f1.length > 0 ? f1[f1.length - 1] : 0;
          f1El.textContent = latestF1.toFixed(4);
          modelEl.textContent = modelEl.textContent !== '—' ? modelEl.textContent : 'Updating...';
        }} else {{
          roundEl.textContent = '—';
          accEl.textContent = '—';
          f1El.textContent = '—';
          modelEl.textContent = '—';
        }}

        const lossTrace = {{ x: rounds, y: loss, type: 'scatter', mode: 'lines+markers', name: 'Loss', line: {{color: '#38bdf8'}} }};
        const accTrace = {{ x: rounds, y: acc, type: 'scatter', mode: 'lines+markers', name: 'Accuracy', line: {{color: '#a3e635'}} }};
        const f1Trace = {{ x: rounds, y: f1, type: 'scatter', mode: 'lines+markers', name: 'F1', line: {{color: '#fbbf24'}} }};
        const baseLayout = {{ paper_bgcolor:'#1e293b', plot_bgcolor:'#0f172a', font:{{color:'#e2e8f0'}} }};
        Plotly.newPlot(lossDiv, [lossTrace], {{ ...baseLayout, title: 'Aggregated Loss' }}, {{displayModeBar:false}});
        Plotly.newPlot(accDiv, [accTrace], {{ ...baseLayout, title: 'Aggregated Accuracy' }}, {{displayModeBar:false}});
        Plotly.newPlot(f1Div, [f1Trace], {{ ...baseLayout, title: 'Aggregated F1' }}, {{displayModeBar:false}});
      }} catch (err) {{
        console.error(err);
      }}
    }}

    fetchMetrics();
    setInterval(fetchMetrics, 2000);

    async function fetchExplanations() {{
      try {{
        const res = await fetch('/api/explanations');
        const data = await res.json();
        const top = data.top_features || [];
        if (top.length === 0) {{
          explainTable.style.display = 'none';
          explainEmpty.style.display = 'block';
          explainMeta.textContent = 'Awaiting explanations...';
          return;
        }}
        explainBody.innerHTML = '';
        const maxScore = top.reduce((m, item) => Math.max(m, Math.abs(item.score || 0)), 0);
        top.forEach((item, idx) => {{
          const tr = document.createElement('tr');
          const sci = Number(item.score || 0).toExponential(3);
          const rel = maxScore > 0 ? ((Math.abs(item.score || 0) / maxScore) * 100).toFixed(2) : '0.00';
          tr.innerHTML = `<td>${{idx + 1}}</td><td>${{item.feature}}</td><td>${{sci}}</td><td>${{rel}}</td>`;
          explainBody.appendChild(tr);
        }});
        explainTable.style.display = 'table';
        explainEmpty.style.display = 'none';
        const round = data.round === null ? 'latest' : data.round;
        explainMeta.textContent = `Model: ${{data.model_type || 'unknown'}} | Round: ${{round}}`;
        modelEl.textContent = data.model_type || '—';
      }} catch (err) {{
        console.error(err);
      }}
    }}

    fetchExplanations();
    setInterval(fetchExplanations, 5000);
  </script>
</body>
</html>
"""
    return Response(html, mimetype="text/html")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flask dashboard for FL metrics")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8501, help="Port to bind")
    parser.add_argument("--metrics", type=str, default="state/metrics.json", help="Metrics JSON path")
    parser.add_argument(
        "--explanations",
        type=str,
        default="state/explanations.json",
        help="Explainability JSON path produced by explain.py",
    )
    return parser.parse_args()


def main() -> None:
    global METRICS_PATH, EXPLANATION_PATH
    args = parse_args()
    METRICS_PATH = Path(args.metrics)
    EXPLANATION_PATH = Path(args.explanations)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
