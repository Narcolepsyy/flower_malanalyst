"""
Streamlit dashboard that reads aggregated metrics from state/metrics.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

METRICS_FILE = Path("state/metrics.json")


def load_metrics() -> dict | None:
    if not METRICS_FILE.exists():
        return None
    try:
        return json.loads(METRICS_FILE.read_text())
    except json.JSONDecodeError:
        return None


def render() -> None:
    st.set_page_config(page_title="Federated Malware Dashboard", layout="wide")
    st.title("🛡️ Federated Malware Detection Dashboard")
    st.caption("Metrics streamed from the Flower server after each round.")

    col1, col2 = st.columns(2)
    round_metric = col1.metric("Latest round", value="0")
    acc_metric = col2.metric("Accuracy", value="0.0")

    chart_col1, chart_col2 = st.columns(2)
    loss_plot = chart_col1.empty()
    acc_plot = chart_col2.empty()
    table_placeholder = st.empty()

    while True:
        data = load_metrics()
        if data and data.get("rounds"):
            df = pd.DataFrame(data)
            df = df.sort_values("rounds")
            round_metric.metric("Latest round", value=str(int(df["rounds"].iloc[-1])))
            acc_metric.metric("Accuracy", value=f"{df['accuracy'].iloc[-1]:.3f}")

            loss_fig = px.line(df, x="rounds", y="loss", markers=True, title="Aggregated loss")
            acc_fig = px.line(df, x="rounds", y="accuracy", markers=True, title="Aggregated accuracy")
            loss_plot.plotly_chart(loss_fig, use_container_width=True)
            acc_plot.plotly_chart(acc_fig, use_container_width=True)
            table_placeholder.dataframe(df.tail(10))
        time.sleep(2)


if __name__ == "__main__":
    render()
