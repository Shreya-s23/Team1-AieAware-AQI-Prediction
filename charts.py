"""
charts.py — Shared matplotlib chart functions for AirAware
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from utils import AQI_LEVELS, classify_aqi


def _setup(C):
    plt.rcParams.update({
        "figure.facecolor": C["mpl_bg"],
        "axes.facecolor":   C["mpl_bg"],
        "axes.edgecolor":   C["border"],
        "axes.labelcolor":  C["text2"],
        "axes.titlecolor":  C["text"],
        "axes.titlesize":   9.5,
        "axes.labelsize":   8,
        "xtick.color":      C["text3"],
        "ytick.color":      C["text3"],
        "xtick.labelsize":  7.5,
        "ytick.labelsize":  7.5,
        "text.color":       C["text"],
        "grid.color":       C["border"],
        "grid.linestyle":   "--",
        "grid.alpha":       0.45,
        "legend.facecolor": C["mpl_bg"],
        "legend.edgecolor": C["border"],
        "legend.labelcolor":C["text"],
        "legend.fontsize":  7.5,
        "font.family":      "DejaVu Sans",
        "font.size":        8.5,
    })


def chart_trend(df, col, C, title=""):
    _setup(C)
    fig, ax = plt.subplots(figsize=(10, 3))
    x, y = df["Datetime"], df[col]
    ax.fill_between(x, y, alpha=0.08, color=C["accent"])
    ax.plot(x, y, color=C["accent"], lw=1.5, alpha=0.9)
    if len(y) > 12:
        ma = y.rolling(12, center=True).mean()
        ax.plot(x, ma, color=C["accent3"], lw=2, alpha=0.75, linestyle="--", label="12-pt avg")
        ax.legend(framealpha=0.3, loc="upper right")
    ax.set_title(title or col, pad=8, fontweight='semibold')
    ax.set_ylabel(col, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, alpha=0.25)
    plt.tight_layout(pad=0.8)
    return fig


def chart_rolling(df, TARGET, C):
    _setup(C)
    fig, ax = plt.subplots(figsize=(10, 3))
    roll7  = df[TARGET].rolling(7).mean()
    roll24 = df[TARGET].rolling(24).mean()
    ax.fill_between(df["Datetime"], df[TARGET], alpha=0.04, color=C["accent"])
    ax.plot(df["Datetime"], df[TARGET], color=C["text3"], lw=0.6, alpha=0.45, label="Raw")
    ax.plot(df["Datetime"], roll7,  color=C["accent"],   lw=1.8, alpha=0.9, label="7-pt avg")
    ax.plot(df["Datetime"], roll24, color=C["moderate"], lw=1.5, alpha=0.8, linestyle="--", label="24-pt avg")
    ax.set_title("Rolling Averages", pad=8, fontweight='semibold')
    ax.legend(framealpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, alpha=0.25)
    plt.tight_layout(pad=0.8)
    return fig


def chart_monthly_bar(df, TARGET, C):
    _setup(C)
    df2 = df.copy()
    df2["month"] = df2["Datetime"].dt.month
    monthly = df2.groupby("month")[TARGET].mean()
    labels  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    mlabels = [labels[m-1] for m in monthly.index]
    q33 = np.percentile(monthly.values, 33)
    q66 = np.percentile(monthly.values, 66)
    colors = [C["good"] if v < q33 else (C["moderate"] if v < q66 else C["bad"]) for v in monthly.values]
    fig, ax = plt.subplots(figsize=(9, 3))
    bars = ax.bar(mlabels, monthly.values, color=colors, width=0.55, alpha=0.85, edgecolor='none')
    for bar, val in zip(bars, monthly.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.0f}", ha='center', va='bottom', fontsize=7.5, color=C["text2"])
    ax.set_title("Monthly Average AQI", pad=8, fontweight='semibold')
    ax.spines[["top","right","left"]].set_visible(False)
    ax.grid(True, axis='y', alpha=0.25)
    ax.tick_params(left=False)
    plt.tight_layout(pad=0.5)
    return fig


def chart_heatmap(df, TARGET, C):
    _setup(C)
    df2 = df.copy()
    df2["hour"] = df2["Datetime"].dt.hour
    df2["day"]  = df2["Datetime"].dt.day_name()
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = df2.pivot_table(index="day", columns="hour", values=TARGET, aggfunc="mean")
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])
    fig, ax = plt.subplots(figsize=(12, 3))
    cmap = LinearSegmentedColormap.from_list("aqi",["#059669","#d97706","#dc2626","#7c3aed"])
    sns.heatmap(pivot, ax=ax, cmap=cmap, annot=False,
                linewidths=0.3, linecolor=C["bg"],
                cbar_kws={"shrink": 0.7, "label": "Avg AQI"})
    ax.set_title("AQI by Hour and Day of Week", pad=8, fontweight='semibold')
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("")
    plt.tight_layout(pad=0.5)
    return fig


def chart_radar(df, C):
    cols = [c for c in ["PM2.5","PM10","NO2(GT)","CO(GT)","Temperature","Humidity"] if c in df.columns]
    if len(cols) < 3:
        return None
    _setup(C)
    means = df[cols].mean()
    maxes = df[cols].max().replace(0, 1)
    vals  = (means / maxes).values.tolist()
    N = len(cols)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vals   += vals[:1]
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles, vals, color=C["accent"], lw=2)
    ax.fill(angles, vals, color=C["accent"], alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cols, fontsize=7.5, color=C["text2"])
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%","50%","75%","100%"], fontsize=6, color=C["text3"])
    ax.grid(color=C["border"], linestyle='--', alpha=0.4)
    ax.spines['polar'].set_color(C["border"])
    ax.set_title("Pollutant Profile", fontsize=8.5, color=C["text"], pad=12, fontweight='semibold')
    plt.tight_layout()
    return fig


def chart_forecast(y_test, preds, C):
    _setup(C)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), gridspec_kw={"width_ratios":[2.5,1]})
    ax1 = axes[0]
    idx = np.arange(len(y_test))
    ax1.fill_between(idx, y_test.values, alpha=0.08, color=C["accent"])
    ax1.plot(idx, y_test.values, color=C["accent"], lw=1.5, label="Actual", alpha=0.9)
    ax1.plot(idx, preds, color=C["moderate"], lw=1.5, linestyle="--", label="Predicted", alpha=0.9)
    ax1.set_title("Actual vs Predicted", pad=8, fontweight='semibold')
    ax1.legend(framealpha=0.35)
    ax1.spines[["top","right"]].set_visible(False)
    ax1.grid(True, alpha=0.25)
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("AQI")
    ax2 = axes[1]
    ax2.scatter(y_test.values, preds, color=C["accent"], alpha=0.3, s=6)
    mn = min(y_test.min(), preds.min())
    mx = max(y_test.max(), preds.max())
    ax2.plot([mn,mx],[mn,mx], color=C["bad"], lw=1.5, linestyle="--", alpha=0.6)
    ax2.set_title("Predicted vs Actual", pad=8, fontweight='semibold')
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    ax2.spines[["top","right"]].set_visible(False)
    ax2.grid(True, alpha=0.25)
    plt.tight_layout(pad=0.8)
    return fig


def chart_importance(model, feature_names, C, top_n=12):
    _setup(C)
    imp = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
    imp = imp.sort_values("Importance", ascending=True).tail(top_n)
    q33 = imp["Importance"].quantile(0.33)
    q66 = imp["Importance"].quantile(0.66)
    colors = [C["accent"] if v >= q66 else (C["accent3"] if v >= q33 else C["text3"]) for v in imp["Importance"]]
    fig, ax = plt.subplots(figsize=(7, top_n * 0.36 + 0.8))
    ax.barh(imp["Feature"], imp["Importance"], color=colors, alpha=0.85, height=0.55)
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance", pad=8, fontweight='semibold')
    ax.spines[["top","right","left"]].set_visible(False)
    ax.grid(True, axis='x', alpha=0.25)
    ax.tick_params(left=False)
    plt.tight_layout(pad=0.8)
    return fig


def chart_correlation(df, cols, C):
    _setup(C)
    avail = [c for c in cols if c in df.columns]
    if len(avail) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    cmap = "RdBu_r" if not C["is_dark"] else LinearSegmentedColormap.from_list(
        "corr", ["#8b5cf6","#1f2937","#10b981"])
    sns.heatmap(df[avail].corr(), annot=True, fmt=".2f", ax=ax, cmap=cmap, center=0,
                square=True, linewidths=0.4, linecolor=C["bg"],
                annot_kws={"size": 7.5}, cbar_kws={"shrink": 0.7})
    ax.set_title("Correlation Matrix", pad=8, fontweight='semibold')
    plt.tight_layout(pad=0.5)
    return fig


def chart_distribution(df, TARGET, C):
    _setup(C)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    data = df[TARGET].dropna()
    n, bins, patches = ax.hist(data, bins=30, edgecolor='none', alpha=0.8)
    for patch, x in zip(patches, bins[:-1]):
        _, color = classify_aqi(x)
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.axvline(data.mean(), color=C["text2"], lw=1.5, linestyle="--", label=f"Mean: {data.mean():.1f}")
    ax.axvline(data.median(), color=C["accent"], lw=1.5, linestyle=":", label=f"Median: {data.median():.1f}")
    ax.legend(framealpha=0.35)
    ax.set_title("AQI Distribution", pad=8, fontweight='semibold')
    ax.set_xlabel("AQI")
    ax.set_ylabel("Frequency")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(pad=0.5)
    return fig


def chart_alerts(df, TARGET, threshold, C):
    _setup(C)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(df["Datetime"], df[TARGET], color=C["text3"], lw=0.8, alpha=0.6)
    ax.fill_between(df["Datetime"], df[TARGET], threshold,
                    where=df[TARGET] > threshold, color=C["bad"], alpha=0.18, label="Alert Zone")
    above = df[df[TARGET] > threshold]
    if len(above):
        ax.scatter(above["Datetime"], above[TARGET], color=C["bad"], s=12, zorder=5, alpha=0.7)
    ax.axhline(threshold, color=C["bad"], lw=1.5, linestyle="--", alpha=0.8, label=f"Threshold ({threshold:.0f})")
    ax.set_title("Alert Monitor", pad=8, fontweight='semibold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(framealpha=0.35)
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, alpha=0.25)
    plt.tight_layout(pad=0.8)
    return fig


def chart_hourly(df, TARGET, C):
    _setup(C)
    df2 = df.copy()
    df2["hour"] = df2["Datetime"].dt.hour
    hourly = df2.groupby("hour")[TARGET].agg(["mean","std"]).reset_index()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(hourly["hour"],
                    hourly["mean"] - hourly["std"],
                    hourly["mean"] + hourly["std"],
                    alpha=0.1, color=C["accent"])
    ax.plot(hourly["hour"], hourly["mean"], color=C["accent"], lw=2, marker='o', markersize=4)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average AQI")
    ax.set_title("Hourly AQI Profile (with std deviation band)", pad=8, fontweight='semibold')
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.25)
    plt.tight_layout(pad=0.8)
    return fig