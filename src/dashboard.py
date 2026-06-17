import json
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pathlib import Path

matplotlib.use("Agg")

STATS_DIR = Path(__file__).parent / "models" / "stats"
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#020202"]
LINESTYLES = ["-", "--", "-.", ":"]

st.set_page_config(page_title="Catan DQN Dashboard", layout="wide")
st.title("Catan DQN Training Dashboard")

st.sidebar.header("Settings")

stat_files = sorted(STATS_DIR.glob("*.json"))
if not stat_files:
    st.error(f"No stats files found in {STATS_DIR}")
    st.stop()

selected = st.sidebar.multiselect(
    "Stat files",
    options=[f.name for f in stat_files],
    default=[f.name for f in stat_files],
)
window = st.sidebar.slider("Smoothing window (episodes)", min_value=10, max_value=500, value=100, step=10)

if not selected:
    st.warning("Select at least one file.")
    st.stop()

@st.cache_data
def load(path: str) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            row = {
                "episode": e["episode"],
                "total_loss": e["total_loss_for_game"],
                "epsilon": e.get("epsilon", None),
                "total_reward": e.get("total_reward", None),
                "mean_max_q": e.get("mean_max_q", None),
                **e["stats"],
            }
            row["loss_per_turn"] = row["total_loss"] / max(row["game_turn"], 1)
            row["mp_won"] = int(row["mp_won"])
            row["mp_has_road"] = int(row["mp_has_road"])
            row["mp_has_army"] = int(row["mp_has_army"])
            rows.append(row)
    return pd.DataFrame(rows)

datasets = {name: load(str(STATS_DIR / name)) for name in selected}

def roll(df: pd.DataFrame, col: str) -> pd.Series:
    return df.set_index("episode")[col].rolling(window, min_periods=1).mean()

st.subheader(f"Last {window} episodes")
metric_cols = st.columns(len(datasets) * 4)
for i, (name, df) in enumerate(datasets.items()):
    tail = df.tail(window)
    base = i * 4
    metric_cols[base + 0].metric("Win rate",   f"{tail['mp_won'].mean():.1%}",              help=name)
    metric_cols[base + 1].metric("Avg VPs",    f"{tail['mp_actual_vps'].mean():.1f}",       help=name)
    metric_cols[base + 2].metric("Avg turns",  f"{tail['game_turn'].mean():.0f}",           help=name)
    metric_cols[base + 3].metric("Loss/turn",  f"{tail['loss_per_turn'].mean():.2f}",       help=name)

st.caption("  |  ".join(selected))
st.divider()

def chart_single(title: str, ylabel: str, col: str):
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    for i, (name, df) in enumerate(datasets.items()):
        s = roll(df, col)
        ax.plot(s.index, s.values, label=name, color=COLORS[i % len(COLORS)], linewidth=1.5)
    if len(datasets) > 1:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def chart_multi(title: str, ylabel: str, cols_data: list, labels: list):
    """Multiple columns per chart; linestyle distinguishes datasets, color distinguishes columns."""
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    for i, (name, df) in enumerate(datasets.items()):
        ls = LINESTYLES[i % len(LINESTYLES)]
        for j, (col, label) in enumerate(zip(cols_data, labels)):
            s = roll(df, col)
            lbl = label if len(datasets) == 1 else f"{label} ({name[:12]})"
            ax.plot(s.index, s.values, label=lbl, color=COLORS[j % len(COLORS)], linestyle=ls, linewidth=1.5)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

r1c1, r1c2 = st.columns(2)
with r1c1:
    st.pyplot(chart_single("Win Rate (rolling)", "Win rate", "mp_won"))
with r1c2:
    st.pyplot(chart_single("Loss per Turn (rolling)", "Loss", "loss_per_turn"))

r2c1, r2c2 = st.columns(2)
with r2c1:
    st.pyplot(chart_single("Actual VPs at Game End", "VPs", "mp_actual_vps"))
with r2c2:
    st.pyplot(chart_single("Game Length", "Turns", "game_turn"))

r3c1, r3c2 = st.columns(2)
with r3c1:
    st.pyplot(chart_multi(
        "Building Progress", "Count",
        ["mp_cities", "mp_settlements", "mp_roads"],
        ["Cities", "Settlements", "Roads"],
    ))
with r3c2:
    st.pyplot(chart_multi(
        "Special Bonuses Rate", "Rate",
        ["mp_has_road", "mp_has_army"],
        ["Longest Road", "Largest Army"],
    ))

r4c1, r4c2 = st.columns(2)
with r4c1:
    st.pyplot(chart_single("Resources at Game End", "Total resources", "mp_total_resources"))
with r4c2:
    st.pyplot(chart_single("Dev Cards Played", "Cards played", "mp_dev_cards_played"))

st.divider()
st.subheader("Training Signal")

def chart_single_optional(title: str, ylabel: str, col: str):
    for df in datasets.values():
        if col not in df.columns:
            st.caption(f"'{col}' not in older stat files — skipped.")
            return None
    return chart_single(title, ylabel, col)

r5c1, r5c2, r5c3 = st.columns(3)
with r5c1:
    fig = chart_single_optional("Epsilon (Exploration)", "Epsilon", "epsilon")
    if fig:
        st.pyplot(fig)
with r5c2:
    fig = chart_single_optional("Total Reward per Episode", "Reward", "total_reward")
    if fig:
        st.pyplot(fig)
with r5c3:
    fig = chart_single_optional("Mean Max-Q per Episode", "Mean max-Q", "mean_max_q")
    if fig:
        st.pyplot(fig)
