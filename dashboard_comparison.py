"""
Comparison Dashboard - Shows results from all aggregation methods.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, Response, jsonify

app = Flask(__name__)

RESULTS_PATH = Path("state/experiment_results.json")


def load_results() -> List[Dict[str, Any]]:
    if not RESULTS_PATH.exists():
        return []
    try:
        return json.loads(RESULTS_PATH.read_text())
    except json.JSONDecodeError:
        return []


@app.route("/api/results")
def api_results() -> Response:
    return jsonify(load_results())


@app.route("/")
def index() -> Response:
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>FL Aggregation Comparison Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    * { box-sizing: border-box; }
    body { 
      font-family: 'Segoe UI', Arial, sans-serif; 
      margin: 0; 
      padding: 24px; 
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); 
      color: #e2e8f0; 
      min-height: 100vh;
    }
    .header {
      text-align: center;
      margin-bottom: 32px;
    }
    h1 { 
      margin: 0;
      font-size: 28px;
      background: linear-gradient(90deg, #38bdf8, #a3e635);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .subtitle { color: #94a3b8; margin-top: 8px; }
    
    .summary-table {
      width: 100%;
      max-width: 1000px;
      margin: 0 auto 32px;
      border-collapse: collapse;
      background: #1e293b;
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    .summary-table th {
      background: linear-gradient(90deg, #0ea5e9, #22c55e);
      color: white;
      padding: 16px;
      text-align: center;
      font-weight: 600;
    }
    .summary-table td {
      padding: 14px 16px;
      text-align: center;
      border-bottom: 1px solid #334155;
    }
    .summary-table tr:last-child td { border-bottom: none; }
    .summary-table tr:hover td { background: #334155; }
    .method-badge {
      display: inline-block;
      padding: 4px 12px;
      border-radius: 20px;
      font-weight: 600;
      font-size: 14px;
    }
    .fedavg { background: #38bdf8; color: #0f172a; }
    .median { background: #a3e635; color: #0f172a; }
    .krum { background: #fbbf24; color: #0f172a; }
    .best { 
      color: #22c55e; 
      font-weight: 700;
    }
    
    .charts-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
      gap: 24px;
      max-width: 1400px;
      margin: 0 auto;
    }
    .chart-card {
      background: #1e293b;
      padding: 20px;
      border-radius: 12px;
      box-shadow: 0 10px 40px rgba(0,0,0,0.25);
    }
    .chart-title {
      font-size: 16px;
      font-weight: 600;
      margin-bottom: 12px;
      color: #cbd5e1;
    }
    
    .footer {
      text-align: center;
      margin-top: 32px;
      color: #64748b;
      font-size: 14px;
    }
  </style>
</head>
<body>
  <div class="header">
    <h1>🛡️ Federated Learning Aggregation Comparison</h1>
    <div class="subtitle">Malware Detection using Obfuscated-MalMem2022 Dataset</div>
  </div>
  
  <table class="summary-table" id="summary-table">
    <thead>
      <tr>
        <th>Method</th>
        <th>Final Accuracy</th>
        <th>Final F1</th>
        <th>Final Precision</th>
        <th>Final Recall</th>
        <th>Final Loss</th>
      </tr>
    </thead>
    <tbody id="summary-body"></tbody>
  </table>
  
  <div class="charts-grid">
    <div class="chart-card">
      <div class="chart-title">📈 Accuracy Over Rounds</div>
      <div id="acc-chart"></div>
    </div>
    <div class="chart-card">
      <div class="chart-title">📉 Loss Over Rounds</div>
      <div id="loss-chart"></div>
    </div>
    <div class="chart-card">
      <div class="chart-title">🎯 F1 Score Over Rounds</div>
      <div id="f1-chart"></div>
    </div>
    <div class="chart-card">
      <div class="chart-title">⚖️ Final Metrics Comparison</div>
      <div id="bar-chart"></div>
    </div>
  </div>
  
  <div class="footer">
    Experiment: 5 rounds, 2 clients, Logistic Regression model, batch_size=32, lr=0.05
  </div>

  <script>
    const colors = {
      fedavg: '#38bdf8',
      median: '#a3e635',
      krum: '#fbbf24'
    };
    
    const baseLayout = {
      paper_bgcolor: '#1e293b',
      plot_bgcolor: '#0f172a',
      font: { color: '#e2e8f0' },
      margin: { l: 50, r: 30, t: 30, b: 40 },
      legend: { orientation: 'h', y: -0.15 },
      xaxis: { gridcolor: '#334155', title: 'Round' },
      yaxis: { gridcolor: '#334155' }
    };

    async function loadData() {
      const res = await fetch('/api/results');
      const data = await res.json();
      
      // Find best values
      let bestAcc = 0, bestF1 = 0;
      data.forEach(d => {
        const lastAcc = d.accuracy[d.accuracy.length - 1];
        const lastF1 = d.f1[d.f1.length - 1];
        if (lastAcc > bestAcc) bestAcc = lastAcc;
        if (lastF1 > bestF1) bestF1 = lastF1;
      });
      
      // Populate summary table
      const tbody = document.getElementById('summary-body');
      tbody.innerHTML = '';
      data.forEach(d => {
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
      
      // Accuracy chart
      const accTraces = data.map(d => ({
        x: d.rounds,
        y: d.accuracy.map(v => v * 100),
        type: 'scatter',
        mode: 'lines+markers',
        name: d.method.toUpperCase(),
        line: { color: colors[d.method], width: 3 },
        marker: { size: 8 }
      }));
      Plotly.newPlot('acc-chart', accTraces, {
        ...baseLayout,
        yaxis: { ...baseLayout.yaxis, title: 'Accuracy (%)' }
      }, {displayModeBar: false});
      
      // Loss chart
      const lossTraces = data.map(d => ({
        x: d.rounds,
        y: d.loss,
        type: 'scatter',
        mode: 'lines+markers',
        name: d.method.toUpperCase(),
        line: { color: colors[d.method], width: 3 },
        marker: { size: 8 }
      }));
      Plotly.newPlot('loss-chart', lossTraces, {
        ...baseLayout,
        yaxis: { ...baseLayout.yaxis, title: 'Loss' }
      }, {displayModeBar: false});
      
      // F1 chart
      const f1Traces = data.map(d => ({
        x: d.rounds,
        y: d.f1.map(v => v * 100),
        type: 'scatter',
        mode: 'lines+markers',
        name: d.method.toUpperCase(),
        line: { color: colors[d.method], width: 3 },
        marker: { size: 8 }
      }));
      Plotly.newPlot('f1-chart', f1Traces, {
        ...baseLayout,
        yaxis: { ...baseLayout.yaxis, title: 'F1 Score (%)' }
      }, {displayModeBar: false});
      
      // Bar chart - show all key metrics side by side
      if (data.length > 0) {
        const methods = data.map(d => d.method.toUpperCase());
        const accValues = data.map(d => (d.accuracy && d.accuracy.length > 0) ? d.accuracy[d.accuracy.length - 1] * 100 : 0);
        const f1Values = data.map(d => (d.f1 && d.f1.length > 0) ? d.f1[d.f1.length - 1] * 100 : 0);
        const recallValues = data.map(d => (d.recall && d.recall.length > 0) ? d.recall[d.recall.length - 1] * 100 : 0);
        
        const barTraces = [
          {
            x: methods,
            y: accValues,
            name: 'Accuracy',
            type: 'bar',
            marker: { color: '#38bdf8' },
            text: accValues.map(v => v.toFixed(2) + '%'),
            textposition: 'outside',
            textfont: { color: '#e2e8f0', size: 11 }
          },
          {
            x: methods,
            y: f1Values,
            name: 'F1',
            type: 'bar',
            marker: { color: '#a3e635' },
            text: f1Values.map(v => v.toFixed(2) + '%'),
            textposition: 'outside',
            textfont: { color: '#e2e8f0', size: 11 }
          },
          {
            x: methods,
            y: recallValues,
            name: 'Recall',
            type: 'bar',
            marker: { color: '#fbbf24' },
            text: recallValues.map(v => v.toFixed(2) + '%'),
            textposition: 'outside',
            textfont: { color: '#e2e8f0', size: 11 }
          }
        ];
        Plotly.newPlot('bar-chart', barTraces, {
          paper_bgcolor: '#1e293b',
          plot_bgcolor: '#0f172a',
          font: { color: '#e2e8f0' },
          margin: { l: 50, r: 30, t: 30, b: 60 },
          legend: { orientation: 'h', y: -0.25 },
          barmode: 'group',
          xaxis: { 
            type: 'category',
            title: 'Aggregation Method',
            gridcolor: '#334155',
            tickfont: { size: 12 }
          },
          yaxis: { 
            title: 'Score (%)', 
            gridcolor: '#334155',
            range: [99.6, 100.05]
          }
        }, {displayModeBar: false});
      }
    }
    
    loadData();
  </script>
</body>
</html>
"""
    return Response(html, mimetype="text/html")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8502)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
