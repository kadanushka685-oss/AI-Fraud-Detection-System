"""Plotly chart builders used by the Streamlit pages.

Kept separate from app.py so the layout code stays readable. Every function returns a Figure.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import TARGET_COLUMN

FRAUD_COLOR = "#B42318"
LEGIT_COLOR = "#0F766E"
RISK_COLORS = {"Low": "#2F855A", "Medium": "#B7791F", "High": "#B42318"}
# Accuracy is shown in the tables but left out of the chart: with rare fraud it flatters weak models.
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}
_LAYOUT = dict(template="simple_white", margin=dict(l=20, r=20, t=60, b=20))


def class_balance_chart(stats):
    fig = go.Figure(go.Pie(
        labels=["Legitimate", "Fraudulent"],
        values=[stats["legit"], stats["fraud"]],
        hole=0.6,
        sort=False,
        marker=dict(colors=[LEGIT_COLOR, FRAUD_COLOR]),
    ))
    fig.update_layout(title="Fraud vs legitimate transactions", **_LAYOUT)
    return fig


def weekly_trend_chart(trend):
    fig = px.line(
        trend, x="timestamp", y="fraud_rate_pct", markers=True,
        labels={"timestamp": "Week", "fraud_rate_pct": "Fraud rate (%)"},
    )
    fig.update_traces(line_color=FRAUD_COLOR)
    fig.update_layout(title="Weekly fraud rate", **_LAYOUT)
    return fig


def amount_box_chart(df):
    labelled = df.assign(Class=df[TARGET_COLUMN].map({0: "Legitimate", 1: "Fraudulent"}))
    fig = px.box(
        labelled, x="Class", y="amount", color="Class", log_y=True, points=False,
        color_discrete_map={"Legitimate": LEGIT_COLOR, "Fraudulent": FRAUD_COLOR},
        labels={"amount": "Amount (log scale)"},
    )
    fig.update_layout(title="Transaction amounts by class", showlegend=False, **_LAYOUT)
    return fig


def category_rate_chart(category_rates):
    ordered = category_rates.sort_values("fraud_rate_pct")
    fig = px.bar(
        ordered, x="fraud_rate_pct", y="merchant_category", orientation="h",
        labels={"fraud_rate_pct": "Fraud rate (%)", "merchant_category": "Merchant category"},
    )
    fig.update_traces(marker_color=FRAUD_COLOR)
    fig.update_layout(title="Fraud rate by merchant category", **_LAYOUT)
    return fig


def risk_distribution_chart(counts):
    fig = go.Figure(go.Bar(
        x=list(counts.index),
        y=[int(v) for v in counts.values],
        marker_color=[RISK_COLORS[level] for level in counts.index],
    ))
    fig.update_layout(title="Risk level distribution", yaxis_title="Transactions", **_LAYOUT)
    return fig


def model_comparison_chart(models):
    rows = [
        {"Model": name, "Metric": label, "Score": metrics[key]}
        for name, metrics in models.items()
        for key, label in METRIC_LABELS.items()
        if key != "accuracy" and metrics.get(key) is not None
    ]
    fig = px.bar(pd.DataFrame(rows), x="Metric", y="Score", color="Model", barmode="group")
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(title="Model comparison on the held-out test set", **_LAYOUT)
    return fig


def confusion_matrix_chart(matrix, model_name):
    z = [[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]]
    fig = go.Figure(go.Heatmap(
        z=z,
        x=["Predicted legitimate", "Predicted fraud"],
        y=["Actual legitimate", "Actual fraud"],
        colorscale="Blues",
        showscale=False,
        text=z,
        texttemplate="%{text}",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=f"Confusion matrix: {model_name}", **_LAYOUT)
    return fig


def _auc_text(value):
    return f"{value:.3f}" if value is not None else "n/a"


def curves_chart(models):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ROC curve", "Precision-recall curve"))
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#94A3B8"),
                   name="Random guess", showlegend=False),
        row=1, col=1,
    )
    for name, metrics in models.items():
        curves = metrics.get("curves")
        if not curves:
            continue
        fig.add_trace(
            go.Scatter(x=curves["fpr"], y=curves["tpr"], mode="lines",
                       name=f"{name} (AUC {_auc_text(metrics.get('roc_auc'))})"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=curves["recall"], y=curves["precision"], mode="lines",
                       name=f"{name} (AP {_auc_text(metrics.get('pr_auc'))})", showlegend=False),
            row=1, col=2,
        )
    fig.update_xaxes(title_text="False positive rate", row=1, col=1)
    fig.update_yaxes(title_text="True positive rate", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)
    fig.update_layout(title="Test-set ranking curves", **_LAYOUT)
    return fig
