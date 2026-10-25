"""Electricity Consumption Optimizer - Streamlit dashboard.

Run:  streamlit run app/dashboard.py
Pages: Monthly Summary, Daily / Weekly / Monthly Analysis, Insights, Bill Prediction.
Everything shown is calculated from data/daily_clean.csv (and the trained model).
"""
import inspect
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data_preprocessing import (DAILY_FILE, DATA_DIR, HOURLY_PROFILE_FILE,  # noqa: E402
                                calculate_bill)
from forecast import LSTMForecaster, load_metrics, trend_estimate  # noqa: E402

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Electricity Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }

    section[data-testid="stSidebar"] {
        background: var(--background-color);
        border-right: 1px solid rgba(128,128,128,0.15);
        min-width: 220px !important;
        max-width: 220px !important;
    }
    section[data-testid="stSidebar"] .stButton button {
        width: 100%;
        text-align: left;
        padding: 10px 14px;
        border-radius: 8px;
        border: none;
        background: transparent;
        font-size: 14px;
        color: var(--text-color);
        margin-bottom: 4px;
        cursor: pointer;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background: rgba(55,138,221,0.1);
        color: #378ADD;
    }

    div[data-testid="metric-container"] {
        background: var(--background-color);
        border: 1px solid rgba(128,128,128,0.18);
        border-radius: 10px;
        padding: 12px 16px;
    }
    div[data-testid="metric-container"] label { font-size: 12px !important; color: gray; }

    .section-header {
        font-size: 15px; font-weight: 600; padding: 6px 0 10px 0;
        border-bottom: 1px solid rgba(128,128,128,0.2); margin-bottom: 14px;
    }
    .page-title { font-size: 20px; font-weight: 600; margin-bottom: 4px; }
    .page-sub { font-size: 13px; color: gray; margin-bottom: 18px; }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=40, b=10, l=10, r=10),
)

# Optional columns: shown only when the data file actually contains them.
OPTIONAL_METRICS = [
    # column,         card label,        number format,  how to combine over a period
    ("voltage",       "Avg Voltage (V)",   "{:.1f}",       "mean"),
    ("current",       "Avg Current (A)",   "{:.1f}",       "mean"),
    ("power",         "Avg Power (W)",     "{:.0f}",       "mean"),
    ("power_factor",  "Power Factor",      "{:.2f}",       "mean"),
    ("frequency",     "Frequency (Hz)",    "{:.1f}",       "mean"),
    ("run_hours",     "Running Hours",     "{:.0f} hr",    "sum"),
]


# ── DATA LOADING ──────────────────────────────────────────────
@st.cache_data
def load_data():
    """Daily table written by the preprocessing step (None if it doesn't exist)."""
    if not os.path.exists(DAILY_FILE):
        return None
    df = pd.read_csv(DAILY_FILE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if "is_anomaly" not in df.columns:
        df["is_anomaly"] = df["usage_kwh"] > df["usage_kwh"].mean() + 2 * df["usage_kwh"].std()
    return df


@st.cache_data
def load_hourly_profile():
    """Average kWh for each hour of the day (None if not available)."""
    if not os.path.exists(HOURLY_PROFILE_FILE):
        return None
    prof = pd.read_csv(HOURLY_PROFILE_FILE).set_index("hour")["avg_kwh"]
    return prof.reindex(range(24)).fillna(0)


@st.cache_resource
def load_forecaster():
    return LSTMForecaster.load()


def data_note():
    path = os.path.join(DATA_DIR, "DATA_NOTE.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


# ── SMALL HELPERS ─────────────────────────────────────────────
def show(fig):
    """Draw a Plotly chart at full width (works on old and new Streamlit)."""
    if "width" in inspect.signature(st.plotly_chart).parameters:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def chart_grid(figures):
    """Lay charts out two per row."""
    for i in range(0, len(figures), 2):
        cols = st.columns(2)
        for col, fig in zip(cols, figures[i:i + 2]):
            with col:
                show(fig)


def metric_row(cards):
    """cards: list of (label, value, delta, delta_color), shown four per row."""
    for i in range(0, len(cards), 4):
        cols = st.columns(4)
        for col, (label, value, delta, color) in zip(cols, cards[i:i + 4]):
            col.metric(label, value, delta, delta_color=color)


def pct_delta(new, old, suffix=" vs prev"):
    if not old:
        return "N/A"
    return f"{(new - old) / old * 100:+.1f}%{suffix}"


def period_bill(kwh, n_days):
    """Full months use the monthly slabs; shorter periods get pro-rated slabs."""
    return calculate_bill(kwh, days=30 if n_days >= 28 else max(n_days, 1))


def header(title, subtitle):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def hour_label(h):
    h = h % 24
    return f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}"


def peak_window(profile, width=3):
    """The `width` consecutive hours that use the most energy."""
    values = profile.to_numpy()
    sums = [values[i:i + width].sum() for i in range(24 - width + 1)]
    start = int(np.argmax(sums))
    share = sums[start] / values.sum() * 100 if values.sum() else 0
    return start, start + width, share


def monthly_table(df):
    """One row per calendar month, using complete months when there are enough."""
    d = df.copy()
    d["ym"] = d["date"].dt.to_period("M")
    agg = {"usage_kwh": "sum", "date": "count"}
    for col in ("voltage", "current"):
        if col in d.columns:
            agg[col] = "mean"
    m = d.groupby("ym").agg(agg).rename(columns={"date": "days"}).reset_index()
    m["complete"] = m["days"] >= m["ym"].dt.days_in_month
    if m["complete"].sum() >= 2:
        m = m[m["complete"]]
    m["label"] = m["ym"].astype(str)
    m["cost"] = [period_bill(k, n) for k, n in zip(m["usage_kwh"], m["days"])]
    return m.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════
def render_sidebar(df):
    with st.sidebar:
        st.markdown("## ⚡ Electricity\nOptimizer")
        st.markdown("---")

        st.markdown("### 📊 Overview")
        if st.button("🏠  Monthly Summary"):
            st.session_state.page = "summary"

        st.markdown("### 📈 Analysis")
        if st.button("📅  Daily Analysis"):
            st.session_state.page = "daily"
        if st.button("📆  Weekly Analysis"):
            st.session_state.page = "weekly"
        if st.button("🗓️  Monthly Analysis"):
            st.session_state.page = "monthly"

        st.markdown("### 🤖 Smart Features")
        if st.button("💡  Insights"):
            st.session_state.page = "insights"
        if st.button("🔮  Bill Prediction"):
            st.session_state.page = "prediction"

        st.markdown("---")
        st.caption(f"Data: {len(df)} days, {df['date'].min():%d %b %Y} to {df['date'].max():%d %b %Y}")
        note = data_note()
        if note:
            st.caption(f"ℹ️ {note}")
        st.caption("Built with Python · Streamlit · TensorFlow")


# ══════════════════════════════════════════════════════════════
#  PAGE: MONTHLY SUMMARY
# ══════════════════════════════════════════════════════════════
def page_summary(df):
    today = df["date"].max()
    cur = df[df["date"].dt.to_period("M") == today.to_period("M")]
    prev = df[df["date"].dt.to_period("M") == today.to_period("M") - 1]

    cur_kwh, prev_kwh = cur["usage_kwh"].sum(), prev["usage_kwh"].sum()
    cur_bill = period_bill(cur_kwh, len(cur))
    prev_bill = period_bill(prev_kwh, len(prev)) if len(prev) else 0

    header("🏠 Monthly Summary",
           f"Overview for {today:%B %Y}, compared with the previous month")

    section("⚡ Electrical Parameters")
    bill_delta = f"{'+' if cur_bill >= prev_bill else '-'}₹{abs(cur_bill - prev_bill):,.0f} vs prev"
    cards = [
        ("Total Energy (kWh)", f"{cur_kwh:.1f}", pct_delta(cur_kwh, prev_kwh), "inverse"),
        ("Current Bill (₹)", f"₹{cur_bill:,.0f}", bill_delta if len(prev) else "N/A", "inverse"),
    ]
    for col, label, fmt, how in OPTIONAL_METRICS:
        if col in df.columns:
            c_val = cur[col].mean() if how == "mean" else cur[col].sum()
            p_val = (prev[col].mean() if how == "mean" else prev[col].sum()) if len(prev) else 0
            cards.append((label, fmt.format(c_val), pct_delta(c_val, p_val), "off"))
    metric_row(cards)

    st.markdown("####")
    section("📊 This Month vs Previous Month")
    col_l, col_r = st.columns(2)
    with col_l:
        fig = go.Figure(go.Bar(
            x=["Previous Month", "Current Month"], y=[prev_kwh, cur_kwh],
            marker_color=["#BA7517", "#378ADD"],
            text=[f"{prev_kwh:.1f} kWh", f"{cur_kwh:.1f} kWh"], textposition="outside",
        ))
        fig.update_layout(title="Energy Consumption Comparison (kWh)", **PLOTLY_LAYOUT)
        show(fig)
    with col_r:
        fig2 = go.Figure(go.Pie(
            labels=["Previous Bill", "Current Bill"], values=[prev_bill, cur_bill],
            hole=0.55, marker_colors=["#BA7517", "#378ADD"],
        ))
        fig2.update_layout(
            title="Bill Comparison (₹)",
            annotations=[dict(text=f"₹{cur_bill:,.0f}", x=0.5, y=0.5, font_size=16, showarrow=False)],
            legend=dict(orientation="h", y=-0.1), **PLOTLY_LAYOUT)
        show(fig2)

    last30 = df.tail(30).copy()
    last30["label"] = last30["date"].dt.strftime("%d %b")
    fig3 = px.area(last30, x="label", y="usage_kwh",
                   title="Last 30 Days: Daily Consumption Trend (kWh)")
    fig3.update_traces(line_color="#378ADD", fillcolor="rgba(55,138,221,0.15)")
    fig3.update_layout(**PLOTLY_LAYOUT)
    show(fig3)


# ══════════════════════════════════════════════════════════════
#  PAGE: DAILY ANALYSIS
# ══════════════════════════════════════════════════════════════
def page_daily(df):
    last14 = df.tail(14).copy()
    last14["label"] = last14["date"].dt.strftime("%d %b")
    last14["cost"] = last14["usage_kwh"].apply(lambda u: calculate_bill(u, days=1))
    profile = load_hourly_profile()

    header("📅 Daily Analysis", "Last 14 days: energy, cost and usage by hour")

    section("📌 Key Metrics")
    if profile is not None:
        start, end, _ = peak_window(profile)
        peak_value, peak_note = f"{hour_label(start)} – {hour_label(end)}", "Highest-usage hours"
    else:
        peak_value, peak_note = "N/A", "No hourly data"
    metric_row([
        ("Highest Day", f"{last14['usage_kwh'].max():.1f} kWh",
         last14.loc[last14["usage_kwh"].idxmax(), "label"], "off"),
        ("Lowest Day", f"{last14['usage_kwh'].min():.1f} kWh",
         last14.loc[last14["usage_kwh"].idxmin(), "label"], "off"),
        ("Peak Hours", peak_value, peak_note, "off"),
        ("Avg Daily Cost", f"₹{last14['cost'].mean():.0f}", "Last 14 days", "off"),
    ])

    st.markdown("####")
    section("📈 Daily Graphs")
    figs = []
    f1 = px.bar(last14, x="label", y="usage_kwh", title="Daily Energy Consumption (kWh)",
                color="usage_kwh", color_continuous_scale="Blues")
    f1.update_layout(coloraxis_showscale=False, **PLOTLY_LAYOUT)
    figs.append(f1)
    for col, title, color in (("voltage", "Daily Voltage Trend (V)", "#1D9E75"),
                              ("current", "Daily Current Trend (A)", "#7F77DD")):
        if col in last14.columns:
            f = px.line(last14, x="label", y=col, title=title, markers=True)
            f.update_traces(line_color=color, marker_color=color)
            f.update_layout(**PLOTLY_LAYOUT)
            figs.append(f)
    f4 = px.bar(last14, x="label", y="cost", title="Daily Cost Analysis (₹)",
                color="cost", color_continuous_scale="Oranges")
    f4.update_layout(coloraxis_showscale=False, **PLOTLY_LAYOUT)
    figs.append(f4)
    chart_grid(figs)
    st.caption("Daily cost uses the monthly slab tariff with the slab limits divided by 30.")

    if profile is not None:
        section("🕐 Peak Usage Hours")
        top = profile.max()
        fig5 = px.bar(x=[f"{h}:00" for h in range(24)], y=profile.to_numpy(),
                      title="Average Hourly Usage (kWh)", labels={"x": "Hour", "y": "kWh"})
        fig5.update_traces(marker_color=["#E24B4A" if v >= 0.8 * top else "#378ADD"
                                         for v in profile.to_numpy()])
        fig5.update_layout(**PLOTLY_LAYOUT)
        show(fig5)
        st.caption("🔴 Red bars: hours within 20% of the busiest hour. Averaged over the whole dataset.")


# ══════════════════════════════════════════════════════════════
#  PAGE: WEEKLY ANALYSIS
# ══════════════════════════════════════════════════════════════
def page_weekly(df):
    n_weeks = min(4, len(df) // 7)
    header("📆 Weekly Analysis", f"Last {n_weeks} weeks: energy trends and cost")
    if n_weeks < 2:
        st.info("At least two weeks of data are needed for the weekly view.")
        return

    df_w = df.tail(n_weeks * 7).copy().reset_index(drop=True)
    df_w["week_label"] = "Week " + (df_w.index // 7 + 1).astype(str)
    agg = {"usage_kwh": "sum"}
    for col in ("voltage", "current"):
        if col in df_w.columns:
            agg[col] = "mean"
    weekly = df_w.groupby("week_label").agg(agg).reset_index()
    weekly["cost"] = weekly["usage_kwh"].apply(lambda u: calculate_bill(u, days=7))

    wk_avg = weekly["usage_kwh"].mean()
    peak_wk = weekly.loc[weekly["usage_kwh"].idxmax(), "week_label"]
    wow = (weekly["usage_kwh"].iloc[-1] - weekly["usage_kwh"].iloc[-2]) / weekly["usage_kwh"].iloc[-2] * 100

    section("📌 Key Metrics")
    metric_row([
        ("Peak Week", peak_wk, "Highest consumption", "off"),
        ("Avg Weekly Use", f"{wk_avg:.0f} kWh", f"Last {n_weeks} weeks", "off"),
        ("Week-on-Week", f"{wow:+.1f}%", "↑ increase" if wow > 0 else "↓ decrease", "off"),
        ("Avg Weekly Cost", f"₹{weekly['cost'].mean():,.0f}", f"Last {n_weeks} weeks", "off"),
    ])

    st.markdown("####")
    section("📈 Weekly Graphs")
    figs = []
    f1 = px.bar(weekly, x="week_label", y="usage_kwh", title="Weekly Energy Consumption (kWh)",
                color="usage_kwh", color_continuous_scale="Blues")
    f1.update_layout(coloraxis_showscale=False, **PLOTLY_LAYOUT)
    figs.append(f1)
    for col, title, color in (("voltage", "Weekly Voltage Trend (V)", "#1D9E75"),
                              ("current", "Weekly Current Trend (A)", "#7F77DD")):
        if col in weekly.columns:
            f = px.line(weekly, x="week_label", y=col, title=title, markers=True)
            f.update_traces(line_color=color, marker_color=color)
            f.update_layout(**PLOTLY_LAYOUT)
            figs.append(f)

    # Same weeks one year earlier (52 weeks back keeps weekdays aligned) - real data only.
    series = df.set_index("date")["usage_kwh"]
    last_year = []
    for k in range(n_weeks):
        days = df_w["date"].iloc[k * 7:(k + 1) * 7] - pd.Timedelta(days=364)
        vals = series.reindex(days)
        last_year.append(vals.sum() if vals.notna().all() else None)
    if all(v is not None for v in last_year):
        f4 = go.Figure(data=[
            go.Bar(name="This year", x=weekly["week_label"], y=weekly["usage_kwh"], marker_color="#378ADD"),
            go.Bar(name="Same weeks last year", x=weekly["week_label"], y=last_year, marker_color="#BA7517"),
        ])
        f4.update_layout(barmode="group", title="Year-on-Year Comparison (kWh)",
                         legend=dict(orientation="h", y=-0.2), **PLOTLY_LAYOUT)
        figs.append(f4)
    chart_grid(figs)

    f5 = px.bar(weekly, x="week_label", y="cost", title="Weekly Cost Analysis (₹)",
                color="cost", color_continuous_scale="Oranges")
    f5.update_layout(coloraxis_showscale=False, **PLOTLY_LAYOUT)
    show(f5)

    section("📊 Average Weekly Usage Pattern")
    f6 = go.Figure()
    f6.add_trace(go.Bar(x=weekly["week_label"], y=weekly["usage_kwh"],
                        name="Weekly Usage", marker_color="#378ADD"))
    f6.add_trace(go.Scatter(x=weekly["week_label"], y=[wk_avg] * len(weekly), name="Average",
                            mode="lines", line=dict(color="#E24B4A", dash="dash", width=2)))
    f6.update_layout(title="Weekly Usage vs Average (kWh)",
                     legend=dict(orientation="h", y=-0.2), **PLOTLY_LAYOUT)
    show(f6)


# ══════════════════════════════════════════════════════════════
#  PAGE: MONTHLY ANALYSIS
# ══════════════════════════════════════════════════════════════
def page_monthly(df):
    all_months = monthly_table(df)
    monthly = all_months.tail(6).reset_index(drop=True)

    header("🗓️ Monthly Analysis", f"Last {len(monthly)} months: energy, cost and comparison")
    if len(monthly) < 2:
        st.info("At least two months of data are needed for the monthly view.")
        return

    avg_monthly = monthly["usage_kwh"].mean()
    peak_month = monthly.loc[monthly["usage_kwh"].idxmax(), "label"]
    mom = (monthly["usage_kwh"].iloc[-1] - monthly["usage_kwh"].iloc[-2]) / monthly["usage_kwh"].iloc[-2] * 100

    section("📌 Key Metrics")
    metric_row([
        ("Peak Month", peak_month, "Highest consumption", "off"),
        ("Avg Monthly Use", f"{avg_monthly:.0f} kWh", f"Last {len(monthly)} months", "off"),
        ("Month-on-Month", f"{mom:+.1f}%", "↑ increase" if mom > 0 else "↓ decrease", "off"),
        ("Avg Monthly Bill", f"₹{monthly['cost'].mean():,.0f}", f"Last {len(monthly)} months", "off"),
    ])

    st.markdown("####")
    section("📈 Monthly Graphs")
    f1 = px.bar(monthly, x="label", y="usage_kwh", title="Monthly Energy Consumption (kWh)",
                color="usage_kwh", color_continuous_scale="Blues")
    f1.update_layout(coloraxis_showscale=False, **PLOTLY_LAYOUT)
    show(f1)

    figs = []
    for col, title, color in (("voltage", "Monthly Voltage Trend (V)", "#1D9E75"),
                              ("current", "Monthly Current Trend (A)", "#7F77DD")):
        if col in monthly.columns:
            f = px.line(monthly, x="label", y=col, title=title, markers=True)
            f.update_traces(line_color=color, marker_color=color)
            f.update_layout(**PLOTLY_LAYOUT)
            figs.append(f)
    if figs:
        chart_grid(figs)

    # Same months one year earlier, taken from the data itself.
    by_period = all_months.set_index("ym")["usage_kwh"]
    last_year = [by_period.get(p - 12) for p in (pd.Period(m) for m in monthly["label"])]
    if all(v is not None for v in last_year):
        f4 = go.Figure(data=[
            go.Bar(name="This year", x=monthly["label"], y=monthly["usage_kwh"], marker_color="#378ADD"),
            go.Bar(name="Same month last year", x=monthly["label"], y=last_year, marker_color="#BA7517"),
        ])
        f4.update_layout(barmode="group", title="Year-on-Year Comparison (kWh)",
                         legend=dict(orientation="h", y=-0.2), **PLOTLY_LAYOUT)
        show(f4)
    else:
        st.caption("Year-on-year comparison needs data from the same months one year earlier.")


# ══════════════════════════════════════════════════════════════
#  PAGE: INSIGHTS
# ══════════════════════════════════════════════════════════════
def build_insights(df, profile):
    """Rule-based insights calculated from the data. Savings are 'what if' estimates:
    the assumed reduction is stated on each card and priced with the slab tariff."""
    month_kwh = df["usage_kwh"].tail(30).sum()
    month_bill = calculate_bill(month_kwh)

    def saving(kwh_cut):
        kwh_cut = min(max(kwh_cut, 0), month_kwh)
        return kwh_cut, month_bill - calculate_bill(month_kwh - kwh_cut)

    insights = []

    if profile is not None and profile.sum() > 0:
        start, end, share = peak_window(profile)
        insights.append(dict(
            title=f"Move flexible loads out of {hour_label(start)} – {hour_label(end)}",
            desc=f"These three hours use {share:.0f}% of a typical day's energy. Run the washing "
                 "machine, dishwasher or water heater at other times. With a slab tariff this "
                 "does not change the bill, but it lowers peak demand and helps if your utility "
                 "offers time-of-day rates.",
            assumption=None, kwh=0, inr=0))

        base_daily = profile.min() * 24
        base_share = base_daily / profile.sum() * 100
        kwh, inr = saving(base_daily * 30 * 0.25)
        insights.append(dict(
            title="Reduce always-on (standby) load",
            desc=f"Even in the quietest hour the home draws about {profile.min():.2f} kW. Held for "
                 f"24 hours, that is about {base_share:.0f}% of a typical day's energy. Unplug idle "
                 "TVs, set-top boxes and chargers, and check for appliances left running.",
            assumption="Assumes the always-on load is cut by 25%", kwh=kwh, inr=inr))

    by_month = df.groupby(df["date"].dt.month)["usage_kwh"].mean()
    if len(df) >= 180 and by_month.min() > 0:
        hi, lo = by_month.idxmax(), by_month.idxmin()
        gap = (by_month.max() / by_month.min() - 1) * 100
        kwh, inr = saving(by_month.max() * 30 * 0.05)
        names = pd.Series(pd.date_range("2000-01-01", periods=12, freq="MS").strftime("%B"),
                          index=range(1, 13))
        insights.append(dict(
            title=f"Keep cooling use in check in {names[hi]}",
            desc=f"{names[hi]} is the busiest month at {by_month.max():.1f} kWh/day, "
                 f"{gap:.0f}% above {names[lo]}. Service the AC before the season and raise the "
                 "thermostat setting by a degree or two.",
            assumption="Assumes a 5% cut in that month's usage", kwh=kwh, inr=inr))

    wk = df.groupby("is_weekend")["usage_kwh"].mean()
    if len(wk) == 2 and wk[0] > 0 and wk[1] > wk[0] * 1.05:
        kwh, inr = saving((wk[1] - wk[0]) * 8)
        insights.append(dict(
            title="Weekends use more energy than weekdays",
            desc=f"Weekend days average {wk[1]:.1f} kWh against {wk[0]:.1f} kWh on weekdays "
                 f"({(wk[1] / wk[0] - 1) * 100:.0f}% higher).",
            assumption="Assumes weekend usage is brought down to the weekday level", kwh=kwh, inr=inr))

    n_anom = int(df["is_anomaly"].sum())
    if n_anom:
        top = df.loc[df["usage_kwh"].idxmax()]
        insights.append(dict(
            title="Check unusually high-usage days",
            desc=f"{n_anom} day(s) used more than the mean + 2 standard deviations. The highest was "
                 f"{top['date']:%d %b %Y} at {top['usage_kwh']:.1f} kWh. Look for appliances left on.",
            assumption=None, kwh=0, inr=0))

    for ins in insights:
        pct = ins["inr"] / month_bill * 100 if month_bill else 0
        ins["impact"] = "🔴 High" if pct >= 5 else "🟡 Medium" if pct >= 2 else \
                        "🟢 Low" if ins["inr"] > 0 else "⚪ Demand / awareness"
    return insights


def page_insights(df):
    header("💡 Energy Insights & Recommendations",
           "Rule-based insights calculated from your consumption data")
    insights = build_insights(df, load_hourly_profile())
    if not insights:
        st.info("Not enough data to generate insights yet.")
        return

    for ins in insights:
        savings_html = ""
        if ins["inr"] > 0:
            savings_html = (f"<span><b>Energy saved:</b> {ins['kwh']:.0f} kWh/month</span>"
                            f"<span><b>Cost saved:</b> ₹{ins['inr']:,.0f}/month</span>")
        note_html = ""
        if ins["assumption"]:
            note_html = f"<div style='font-size:12px;color:gray;margin-bottom:8px;'>{ins['assumption']}</div>"

        # Built as one unindented line: Markdown treats a blank line followed by
        # indented HTML as a code block, which would show the raw tags.
        card = (
            "<div style='border:1px solid rgba(128,128,128,0.25);border-radius:10px;"
            "padding:14px 18px;margin-bottom:12px;'>"
            f"<div style='font-size:15px;font-weight:600;margin-bottom:6px;'>{ins['title']}</div>"
            f"<div style='font-size:13px;color:gray;margin-bottom:8px;'>{ins['desc']}</div>"
            f"{note_html}"
            "<div style='display:flex;gap:32px;flex-wrap:wrap;'>"
            f"<span><b>Impact:</b> {ins['impact']}</span>{savings_html}"
            "</div></div>"
        )
        st.markdown(card, unsafe_allow_html=True)

    with_savings = [i for i in insights if i["inr"] > 0]
    if with_savings:
        st.markdown("####")
        section("📊 Estimated Savings per Recommendation")
        fig = px.bar(x=[i["title"][:38] + ("..." if len(i["title"]) > 38 else "") for i in with_savings],
                     y=[i["inr"] for i in with_savings],
                     title="Estimated Monthly Savings if the Assumption Holds (₹)",
                     labels={"x": "Recommendation", "y": "₹ Saved"},
                     color=[i["inr"] for i in with_savings], color_continuous_scale="Greens")
        fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-15, **PLOTLY_LAYOUT)
        show(fig)
        st.caption("Savings are estimates from the slab tariff, based on the assumption shown on each card. "
                   "They are not measured results.")


# ══════════════════════════════════════════════════════════════
#  PAGE: BILL PREDICTION
# ══════════════════════════════════════════════════════════════
def page_prediction(df):
    usage = df["usage_kwh"].to_numpy()
    forecaster = load_forecaster()
    metrics = load_metrics()

    daily_forecast = None
    if forecaster is not None and len(usage) >= forecaster.sequence_length:
        daily_forecast = forecaster.forecast(usage, steps=30)
        pred_kwh, method = float(daily_forecast.sum()), "LSTM (30-day forecast, one day at a time)"
    else:
        pred_kwh, method = trend_estimate(usage), "Trend estimate (trained model not found)"

    last30_kwh = usage[-30:].sum()
    cur_bill = calculate_bill(last30_kwh)
    pred_bill = calculate_bill(pred_kwh)
    change_pct = (pred_kwh - last30_kwh) / last30_kwh * 100 if last30_kwh else 0
    today = df["date"].max()

    header("🔮 Next Month Bill Prediction", f"Forecast method: {method}")

    section("📌 Prediction Summary")
    acc = f"{metrics['accuracy_percent']:.1f}%" if metrics and forecaster is not None else "N/A"
    metric_row([
        ("Predicted Consumption", f"{pred_kwh:,.0f} kWh", f"{change_pct:+.1f}% vs last 30 days", "inverse"),
        ("Predicted Bill", f"₹{pred_bill:,.0f}",
         f"{'+' if pred_bill >= cur_bill else '-'}₹{abs(pred_bill - cur_bill):,.0f} vs last 30 days", "inverse"),
        ("Expected Change", "Increase" if change_pct > 0 else "Decrease",
         f"{abs(change_pct):.1f}% {'higher' if change_pct > 0 else 'lower'}", "off"),
        ("Model Accuracy (1 day ahead)", acc, "Test period, 100 - MAPE", "off"),
    ])
    st.caption("Accuracy is measured on held-out days for one-day-ahead predictions. A 30-day forecast "
               "repeats that step 30 times, so it is less certain than the figure above.")

    st.markdown("####")
    section("📈 Consumption and Forecast")

    monthly = monthly_table(df).tail(6)
    next_label = str(today.to_period("M") + 1)
    labels = monthly["label"].tolist() + [next_label]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["label"], y=monthly["usage_kwh"], name="Actual",
                             line=dict(color="#378ADD", width=2), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=[labels[-2], labels[-1]], y=[monthly["usage_kwh"].iloc[-1], pred_kwh],
                             name="Forecast", line=dict(color="#E24B4A", width=2, dash="dash"),
                             mode="lines+markers"))
    fig.update_layout(title="Monthly Consumption + Next Month Forecast (kWh)",
                      legend=dict(orientation="h", y=-0.2), **PLOTLY_LAYOUT)
    show(fig)

    if daily_forecast is not None:
        hist = df.tail(30)
        future_dates = pd.date_range(today + pd.Timedelta(days=1), periods=30, freq="D")
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=hist["date"], y=hist["usage_kwh"], name="Last 30 days (actual)",
                                   line=dict(color="#378ADD", width=2)))
        fig_d.add_trace(go.Scatter(x=future_dates, y=daily_forecast, name="Next 30 days (forecast)",
                                   line=dict(color="#E24B4A", width=2, dash="dash")))
        fig_d.update_layout(title="Daily Consumption and LSTM Forecast (kWh)",
                            legend=dict(orientation="h", y=-0.2), **PLOTLY_LAYOUT)
        show(fig_d)

    monthly = monthly.assign(bill=monthly["cost"])
    fig2 = px.line(monthly, x="label", y="bill", title="Monthly Bill Trend (₹)", markers=True)
    fig2.update_traces(line_color="#BA7517", marker_color="#BA7517")
    fig2.add_scatter(x=[next_label], y=[pred_bill], mode="markers",
                     marker=dict(color="#E24B4A", size=12, symbol="star"), name="Next Month Forecast")
    fig2.update_layout(legend=dict(orientation="h", y=-0.2), **PLOTLY_LAYOUT)
    show(fig2)


# ══════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════
def main():
    df = load_data()
    if df is None:
        st.error("No data found. Run `python main.py` (or `python data/generate_data.py` and "
                 "`python data_preprocessing.py`) first, then reload this page.")
        st.stop()

    if "page" not in st.session_state:
        st.session_state.page = "summary"
    render_sidebar(df)

    pages = {"summary": page_summary, "daily": page_daily, "weekly": page_weekly,
             "monthly": page_monthly, "insights": page_insights, "prediction": page_prediction}
    pages.get(st.session_state.page, page_summary)(df)


if __name__ == "__main__":
    main()
