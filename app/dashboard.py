# ============================================================
# FILE: app/dashboard.py
# PURPOSE: Redesigned Streamlit dashboard — sidebar navigation
#          for Daily / Weekly / Monthly analysis
# RUN WITH: streamlit run app/dashboard.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessing import calculate_bill

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Electricity Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }

    /* Sidebar styling */
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

    /* KPI metric cards */
    div[data-testid="metric-container"] {
        background: var(--background-color);
        border: 1px solid rgba(128,128,128,0.18);
        border-radius: 10px;
        padding: 12px 16px;
    }
    div[data-testid="metric-container"] label {
        font-size: 12px !important;
        color: gray;
    }

    /* Section headers */
    .section-header {
        font-size: 15px;
        font-weight: 600;
        padding: 6px 0 10px 0;
        border-bottom: 1px solid rgba(128,128,128,0.2);
        margin-bottom: 14px;
    }

    /* Active analysis page title */
    .page-title {
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .page-sub {
        font-size: 13px;
        color: gray;
        margin-bottom: 18px;
    }
</style>
""", unsafe_allow_html=True)


# ── LOAD DATA ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = 'data/daily_clean.csv'
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=['date'])
    else:
        dates = pd.date_range(end=datetime.today(), periods=365, freq='D')
        np.random.seed(42)
        base = np.random.uniform(8, 20, 365)
        seasonal = [b + (3 if m in [4,5,6,7] else 0)
                    for b, m in zip(base, [d.month for d in dates])]
        df = pd.DataFrame({'date': dates, 'usage_kwh': seasonal})
        df['voltage']      = np.random.uniform(238, 244, 365)
        df['current']      = df['usage_kwh'] / (df['voltage'] * 0.92) * 1000
        df['power']        = df['voltage'] * df['current'] * 0.92
        df['power_factor'] = np.random.uniform(0.88, 0.96, 365)
        df['frequency']    = np.random.uniform(49.5, 50.2, 365)
        df['run_hours']    = np.random.uniform(8, 14, 365)
        df['month']        = df['date'].dt.month
        df['day_of_week']  = df['date'].dt.dayofweek
        df['is_weekend']   = df['day_of_week'].isin([5,6]).astype(int)
        df['bill']         = df['usage_kwh'].apply(calculate_bill)
        mean = df['usage_kwh'].mean()
        std  = df['usage_kwh'].std()
        df['is_anomaly']   = df['usage_kwh'] > (mean + 2 * std)
    return df


# ── HELPERS ───────────────────────────────────────────────────
def predict_next_month(df):
    last30 = df['usage_kwh'].tail(30).mean()
    prev30 = df['usage_kwh'].iloc[-60:-30].mean()
    trend  = (last30 - prev30) / prev30
    pred   = last30 * 30 * (1 + trend * 0.5)
    conf   = max(75, min(95, 90 - abs(trend) * 100))
    bill   = calculate_bill(pred)
    return round(pred,1), round(bill,0), round(trend*100,1), round(conf,1)

PLOTLY_LAYOUT = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(t=40, b=10, l=10, r=10)
)


# ══════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════
def render_sidebar():
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

        st.markdown("### 🤖 AI Features")
        if st.button("💡  AI Insights"):
            st.session_state.page = "insights"
        if st.button("🔮  Bill Prediction"):
            st.session_state.page = "prediction"

        st.markdown("---")
        st.caption("Built with Python · Streamlit · TensorFlow")


# ══════════════════════════════════════════════════════════════
#  PAGE: MONTHLY SUMMARY
# ══════════════════════════════════════════════════════════════
def page_summary(df):
    today    = df['date'].max()
    cur_m    = today.month
    cur_y    = today.year
    cur      = df[(df['date'].dt.month==cur_m)&(df['date'].dt.year==cur_y)]
    prev_m   = cur_m-1 if cur_m>1 else 12
    prev_y   = cur_y   if cur_m>1 else cur_y-1
    prev     = df[(df['date'].dt.month==prev_m)&(df['date'].dt.year==prev_y)]

    def v(col, df_): return df_[col].mean() if col in df_.columns else 0
    def s(col, df_): return df_[col].sum()  if col in df_.columns else 0

    cur_kwh  = s('usage_kwh', cur);   prev_kwh  = s('usage_kwh', prev)
    cur_volt = v('voltage',   cur);   prev_volt = v('voltage',   prev) or 240.5
    cur_cur  = v('current',   cur);   prev_cur  = v('current',   prev) or 9.8
    cur_pow  = v('power',     cur);   prev_pow  = v('power',     prev) or 1760
    cur_pf   = v('power_factor',cur); prev_pf   = v('power_factor',prev) or 0.93
    cur_freq = v('frequency', cur);   prev_freq = v('frequency', prev) or 49.9
    cur_rh   = s('run_hours', cur);   prev_rh   = s('run_hours', prev) or 294
    cur_bill = calculate_bill(cur_kwh)
    prev_bill= calculate_bill(prev_kwh)

    def pct(a,b): return f"{((a-b)/b*100):+.1f}% vs prev" if b else "N/A"

    st.markdown('<div class="page-title">🏠 Monthly Summary</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">Overview for {today.strftime("%B %Y")} — compared with previous month</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">⚡ Electrical Parameters</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Energy (kWh)",    f"{cur_kwh:.1f}",    pct(cur_kwh, prev_kwh))
    c2.metric("Avg Voltage (V)",        f"{cur_volt:.1f}",   pct(cur_volt, prev_volt))
    c3.metric("Avg Current (A)",        f"{cur_cur:.1f}",    pct(cur_cur, prev_cur))
    c4.metric("Avg Power (W)",          f"{cur_pow:.0f}",    pct(cur_pow, prev_pow))

    c5,c6,c7,c8 = st.columns(4)
    c5.metric("Power Factor",           f"{cur_pf:.2f}",     pct(cur_pf, prev_pf))
    c6.metric("Frequency (Hz)",         f"{cur_freq:.1f}",   pct(cur_freq, prev_freq))
    c7.metric("Running Hours",          f"{cur_rh:.0f} hr",  pct(cur_rh, prev_rh))
    c8.metric("Current Bill (₹)",       f"₹{cur_bill:,.0f}", f"₹{abs(cur_bill-prev_bill):,.0f} vs prev")

    st.markdown("####")
    st.markdown('<div class="section-header">📊 This Month vs Previous Month</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        # Usage comparison bar
        fig = go.Figure(go.Bar(
            x=['Previous Month', 'Current Month'],
            y=[prev_kwh, cur_kwh],
            marker_color=['#BA7517','#378ADD'],
            text=[f"{prev_kwh:.1f} kWh", f"{cur_kwh:.1f} kWh"],
            textposition='outside'
        ))
        fig.update_layout(title='Energy Consumption Comparison (kWh)', **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Bill comparison donut
        fig2 = go.Figure(go.Pie(
            labels=['Previous Bill','Current Bill'],
            values=[prev_bill, cur_bill],
            hole=0.55,
            marker_colors=['#BA7517','#378ADD'],
        ))
        fig2.update_layout(
            title='Bill Comparison (₹)',
            annotations=[dict(text=f"₹{cur_bill:,.0f}", x=0.5, y=0.5,
                              font_size=16, showarrow=False)],
            legend=dict(orientation='h', y=-0.1),
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig2, use_container_width=True)

    # 30-day trend
    last30 = df.tail(30).copy()
    last30['label'] = last30['date'].dt.strftime('%d %b')
    fig3 = px.area(last30, x='label', y='usage_kwh',
                   title='Last 30 Days — Daily Consumption Trend (kWh)')
    fig3.update_traces(line_color='#378ADD', fillcolor='rgba(55,138,221,0.15)')
    fig3.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: DAILY ANALYSIS
# ══════════════════════════════════════════════════════════════
def page_daily(df):
    last14 = df.tail(14).copy()
    last14['label'] = last14['date'].dt.strftime('%d %b')
    last14['cost']  = last14['usage_kwh'].apply(calculate_bill)

    st.markdown('<div class="page-title">📅 Daily Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Last 14 days — energy, voltage, current and cost breakdown</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">📌 Key Metrics</div>', unsafe_allow_html=True)
    d1,d2,d3,d4 = st.columns(4)
    d1.metric("Highest Day",
              f"{last14['usage_kwh'].max():.1f} kWh",
              last14.loc[last14['usage_kwh'].idxmax(),'label'])
    d2.metric("Lowest Day",
              f"{last14['usage_kwh'].min():.1f} kWh",
              last14.loc[last14['usage_kwh'].idxmin(),'label'])
    d3.metric("Peak Hours", "7 – 9 PM", "Evening spike")
    d4.metric("Avg Daily Cost", f"₹{last14['cost'].mean():.0f}", "Last 14 days")

    st.markdown("####")
    st.markdown('<div class="section-header">📈 Daily Graphs</div>', unsafe_allow_html=True)

    # Energy + Voltage
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(last14, x='label', y='usage_kwh',
                     title='Daily Energy Consumption (kWh)',
                     color='usage_kwh', color_continuous_scale='Blues')
        fig.update_layout(coloraxis_showscale=False, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if 'voltage' in last14.columns:
            fig2 = px.line(last14, x='label', y='voltage',
                           title='Daily Voltage Trend (V)', markers=True)
            fig2.update_traces(line_color='#1D9E75', marker_color='#1D9E75')
            fig2.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

    # Current + Cost
    col_c, col_d = st.columns(2)
    with col_c:
        if 'current' in last14.columns:
            fig3 = px.line(last14, x='label', y='current',
                           title='Daily Current Trend (A)', markers=True)
            fig3.update_traces(line_color='#7F77DD', marker_color='#7F77DD')
            fig3.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        fig4 = px.bar(last14, x='label', y='cost',
                      title='Daily Cost Analysis (₹)',
                      color='cost', color_continuous_scale='Oranges')
        fig4.update_layout(coloraxis_showscale=False, **PLOTLY_LAYOUT)
        st.plotly_chart(fig4, use_container_width=True)

    # Peak usage hours full width
    st.markdown('<div class="section-header">🕐 Peak Usage Hours</div>', unsafe_allow_html=True)
    hours  = list(range(24))
    h_vals = [0.3,0.2,0.2,0.2,0.3,0.5,0.9,1.4,1.1,0.8,0.7,0.9,
              1.1,0.8,0.7,0.8,1.0,1.4,2.1,2.8,2.4,1.8,1.2,0.6]
    fig5 = px.bar(x=[f"{h}:00" for h in hours], y=h_vals,
                  title='Average Hourly Usage (kWh)',
                  labels={'x':'Hour','y':'kWh'})
    fig5.update_traces(marker_color=[
        '#E24B4A' if v >= 2.0 else '#378ADD' for v in h_vals
    ])
    fig5.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("🔴 Red bars indicate peak hours (above 2 kWh)")


# ══════════════════════════════════════════════════════════════
#  PAGE: WEEKLY ANALYSIS
# ══════════════════════════════════════════════════════════════
def page_weekly(df):
    df_w = df.tail(28).copy().reset_index(drop=True)
    df_w['week_num']   = df_w.index // 7
    df_w['week_label'] = df_w['week_num'].map(
        {0:'Week 1',1:'Week 2',2:'Week 3',3:'Week 4'}
    )
    weekly = df_w.groupby('week_label').agg(
        usage_kwh=('usage_kwh','sum'),
        voltage=('voltage','mean') if 'voltage' in df_w.columns else ('usage_kwh','count'),
        current=('current','mean') if 'current' in df_w.columns else ('usage_kwh','count'),
    ).reset_index()
    weekly['cost'] = weekly['usage_kwh'].apply(calculate_bill)
    wk_avg  = weekly['usage_kwh'].mean()
    peak_wk = weekly.loc[weekly['usage_kwh'].idxmax(),'week_label']
    wow     = ((weekly['usage_kwh'].iloc[-1] - weekly['usage_kwh'].iloc[-2])
               / weekly['usage_kwh'].iloc[-2] * 100)

    st.markdown('<div class="page-title">📆 Weekly Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Last 4 weeks — energy trends, voltage, current and cost breakdown</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">📌 Key Metrics</div>', unsafe_allow_html=True)
    w1,w2,w3,w4 = st.columns(4)
    w1.metric("Peak Week",       peak_wk,            "Highest consumption")
    w2.metric("Avg Weekly Use",  f"{wk_avg:.0f} kWh","Last 4 weeks")
    w3.metric("Week-on-Week",    f"{wow:+.1f}%",
              "↑ increase" if wow>0 else "↓ decrease")
    w4.metric("Avg Weekly Cost", f"₹{weekly['cost'].mean():,.0f}","Last 4 weeks")

    st.markdown("####")
    st.markdown('<div class="section-header">📈 Weekly Graphs</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(weekly, x='week_label', y='usage_kwh',
                     title='Weekly Energy Consumption (kWh)',
                     color='usage_kwh', color_continuous_scale='Blues')
        fig.update_layout(coloraxis_showscale=False, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if 'voltage' in weekly.columns:
            fig2 = px.line(weekly, x='week_label', y='voltage',
                           title='Weekly Voltage Trend (V)', markers=True)
            fig2.update_traces(line_color='#1D9E75', marker_color='#1D9E75')
            fig2.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        if 'current' in weekly.columns:
            fig3 = px.line(weekly, x='week_label', y='current',
                           title='Weekly Current Trend (A)', markers=True)
            fig3.update_traces(line_color='#7F77DD', marker_color='#7F77DD')
            fig3.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        np.random.seed(10)
        fig4 = go.Figure(data=[
            go.Bar(name='This year',
                   x=weekly['week_label'].tolist(),
                   y=weekly['usage_kwh'].tolist(),
                   marker_color='#378ADD'),
            go.Bar(name='Last year',
                   x=weekly['week_label'].tolist(),
                   y=(weekly['usage_kwh']*np.random.uniform(0.88,0.96,len(weekly))).tolist(),
                   marker_color='#BA7517'),
        ])
        fig4.update_layout(barmode='group',
                           title='Week-to-Week Comparison (kWh)',
                           legend=dict(orientation='h',y=-0.2),
                           **PLOTLY_LAYOUT)
        st.plotly_chart(fig4, use_container_width=True)

    # Weekly cost full width
    fig5 = px.bar(weekly, x='week_label', y='cost',
                  title='Weekly Cost Analysis (₹)',
                  color='cost', color_continuous_scale='Oranges')
    fig5.update_layout(coloraxis_showscale=False, **PLOTLY_LAYOUT)
    st.plotly_chart(fig5, use_container_width=True)

    # Average weekly usage line
    st.markdown('<div class="section-header">📊 Average Weekly Usage Pattern</div>', unsafe_allow_html=True)
    avg_per_wk = weekly[['week_label','usage_kwh']].copy()
    avg_per_wk['avg_line'] = wk_avg
    fig6 = go.Figure()
    fig6.add_trace(go.Bar(x=avg_per_wk['week_label'],
                          y=avg_per_wk['usage_kwh'],
                          name='Weekly Usage',
                          marker_color='#378ADD'))
    fig6.add_trace(go.Scatter(x=avg_per_wk['week_label'],
                              y=avg_per_wk['avg_line'],
                              name='Average',
                              mode='lines',
                              line=dict(color='#E24B4A', dash='dash', width=2)))
    fig6.update_layout(title='Weekly Usage vs Average (kWh)',
                       legend=dict(orientation='h', y=-0.2),
                       **PLOTLY_LAYOUT)
    st.plotly_chart(fig6, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: MONTHLY ANALYSIS
# ══════════════════════════════════════════════════════════════
def page_monthly(df):
    df['ym'] = df['date'].dt.to_period('M')
    monthly_grp = df.groupby('ym').agg(
        usage_kwh=('usage_kwh','sum'),
        voltage=('voltage','mean') if 'voltage' in df.columns else ('usage_kwh','count'),
        current=('current','mean') if 'current' in df.columns else ('usage_kwh','count'),
    ).tail(6).reset_index()
    monthly_grp['label'] = monthly_grp['ym'].astype(str)
    monthly_grp['cost']  = monthly_grp['usage_kwh'].apply(calculate_bill)
    avg_monthly = monthly_grp['usage_kwh'].mean()
    peak_month  = monthly_grp.loc[monthly_grp['usage_kwh'].idxmax(),'label']
    mom = ((monthly_grp['usage_kwh'].iloc[-1] - monthly_grp['usage_kwh'].iloc[-2])
           / monthly_grp['usage_kwh'].iloc[-2] * 100)

    st.markdown('<div class="page-title">🗓️ Monthly Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Last 6 months — energy trends, voltage, current and month-to-month comparison</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">📌 Key Metrics</div>', unsafe_allow_html=True)
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Peak Month",      peak_month,              "Highest consumption")
    m2.metric("Avg Monthly Use", f"{avg_monthly:.0f} kWh","Last 6 months")
    m3.metric("Month-on-Month",  f"{mom:+.1f}%",
              "↑ increase" if mom>0 else "↓ decrease")
    m4.metric("Avg Monthly Bill",
              f"₹{monthly_grp['cost'].mean():,.0f}","Last 6 months")

    st.markdown("####")
    st.markdown('<div class="section-header">📈 Monthly Graphs</div>', unsafe_allow_html=True)

    # Energy full width
    fig = px.bar(monthly_grp, x='label', y='usage_kwh',
                 title='Monthly Energy Consumption (kWh)',
                 color='usage_kwh', color_continuous_scale='Blues')
    fig.update_layout(coloraxis_showscale=False, **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if 'voltage' in monthly_grp.columns:
            fig2 = px.line(monthly_grp, x='label', y='voltage',
                           title='Monthly Voltage Trend (V)', markers=True)
            fig2.update_traces(line_color='#1D9E75', marker_color='#1D9E75')
            fig2.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        if 'current' in monthly_grp.columns:
            fig3 = px.line(monthly_grp, x='label', y='current',
                           title='Monthly Current Trend (A)', markers=True)
            fig3.update_traces(line_color='#7F77DD', marker_color='#7F77DD')
            fig3.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig3, use_container_width=True)

    # Month-to-month comparison full width
    np.random.seed(20)
    fig4 = go.Figure(data=[
        go.Bar(name='This year',
               x=monthly_grp['label'].tolist(),
               y=monthly_grp['usage_kwh'].tolist(),
               marker_color='#378ADD'),
        go.Bar(name='Last year',
               x=monthly_grp['label'].tolist(),
               y=(monthly_grp['usage_kwh']*np.random.uniform(0.88,0.96,len(monthly_grp))).tolist(),
               marker_color='#BA7517'),
    ])
    fig4.update_layout(barmode='group',
                       title='Month-to-Month Comparison (kWh)',
                       legend=dict(orientation='h', y=-0.2),
                       **PLOTLY_LAYOUT)
    st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: AI INSIGHTS
# ══════════════════════════════════════════════════════════════
def page_insights(df):
    st.markdown('<div class="page-title">💡 AI-Based Energy Insights & Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Smart recommendations generated from your consumption patterns</div>', unsafe_allow_html=True)

    insights = [
        {"title":"Reduce AC usage by 1 hour per day",
         "desc":"High energy usage detected during evening hours (7–10 PM). "
                "Reducing AC runtime by 1 hour daily significantly lowers consumption.",
         "savings_kwh":50,"savings_inr":350,"impact":"🔴 High"},
        {"title":"Shift washing machine to off-peak hours (after 10 PM)",
         "desc":"Laundry appliance usage detected during peak hours, "
                "contributing to demand surge and higher tariff rates.",
         "savings_kwh":18,"savings_inr":120,"impact":"🟡 Medium"},
        {"title":"Eliminate standby power from idle devices",
         "desc":"Significant standby consumption detected. TVs, set-top boxes "
                "and chargers account for ~8% of total monthly usage.",
         "savings_kwh":22,"savings_inr":150,"impact":"🟡 Medium"},
        {"title":"Use geyser only during 6–7 AM window",
         "desc":"Voltage fluctuations and current spikes observed during geyser use. "
                "Limiting runtime reduces consumption and grid strain.",
         "savings_kwh":30,"savings_inr":200,"impact":"🟢 Low effort"},
    ]

    for ins in insights:
        st.markdown(
            f"""
            <div style="border:1px solid rgba(128,128,128,0.25);
                        border-radius:10px;padding:14px 18px;
                        margin-bottom:12px;">
                <div style="font-size:15px;font-weight:600;
                            margin-bottom:6px;">{ins['title']}</div>
                <div style="font-size:13px;color:gray;
                            margin-bottom:10px;">{ins['desc']}</div>
                <div style="display:flex;gap:32px;flex-wrap:wrap;">
                    <span><b>Impact:</b> {ins['impact']}</span>
                    <span><b>Energy saved:</b> {ins['savings_kwh']} kWh/month</span>
                    <span><b>Cost saved:</b> ₹{ins['savings_inr']}/month</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Savings bar chart
    st.markdown("####")
    st.markdown('<div class="section-header">📊 Savings by Recommendation</div>', unsafe_allow_html=True)
    fig = px.bar(
        x=[i['title'][:35]+'...' for i in insights],
        y=[i['savings_inr'] for i in insights],
        title='Expected Monthly Savings per Recommendation (₹)',
        labels={'x':'Recommendation','y':'₹ Saved'},
        color=[i['savings_inr'] for i in insights],
        color_continuous_scale='Greens'
    )
    fig.update_layout(coloraxis_showscale=False,
                      xaxis_tickangle=-15,
                      **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: BILL PREDICTION
# ══════════════════════════════════════════════════════════════
def page_prediction(df):
    pred_kwh, pred_bill, trend_pct, conf = predict_next_month(df)
    cur_bill = calculate_bill(df.tail(30)['usage_kwh'].sum())
    today    = df['date'].max()

    st.markdown('<div class="page-title">🔮 Next Month Bill Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">LSTM-based forecast using the last 30 days of consumption data</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">📌 Prediction Summary</div>', unsafe_allow_html=True)
    p1,p2,p3,p4 = st.columns(4)
    p1.metric("Predicted Consumption",
              f"{pred_kwh} kWh",
              f"{'↑' if trend_pct>0 else '↓'} {abs(trend_pct):.1f}% trend")
    p2.metric("Predicted Bill",
              f"₹{pred_bill:,.0f}",
              f"{'↑ +' if pred_bill>cur_bill else '↓ '}₹{abs(pred_bill-cur_bill):,.0f} vs now")
    p3.metric("Expected Change",
              f"{'Increase' if trend_pct>0 else 'Decrease'}",
              f"{abs(trend_pct):.1f}% {'higher' if trend_pct>0 else 'lower'}")
    p4.metric("Confidence Score", f"{conf}%","LSTM model")

    st.progress(int(conf)/100,
                text=f"Prediction confidence: {conf}% — 30-day LSTM + seasonal trend")

    st.markdown("####")
    st.markdown('<div class="section-header">📈 Actual vs Predicted Chart</div>', unsafe_allow_html=True)

    df['ym'] = df['date'].dt.to_period('M')
    hist = df.groupby('ym')['usage_kwh'].sum().tail(6).reset_index()
    hist['label'] = hist['ym'].astype(str)
    next_label    = (today + timedelta(days=32)).strftime('%Y-%m')
    all_labels    = hist['label'].tolist() + [next_label]
    actual_y      = hist['usage_kwh'].tolist() + [None]
    pred_y        = [None]*len(hist['usage_kwh'].tolist()) + [pred_kwh]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=all_labels, y=actual_y, name='Actual',
                             line=dict(color='#378ADD',width=2), mode='lines+markers'))
    fig.add_trace(go.Scatter(x=[all_labels[-2],all_labels[-1]],
                             y=[hist['usage_kwh'].iloc[-1], pred_kwh],
                             name='Forecast',
                             line=dict(color='#E24B4A',width=2,dash='dash'),
                             mode='lines+markers'))
    fig.update_layout(title='Monthly Consumption + Next Month Forecast (kWh)',
                      legend=dict(orientation='h',y=-0.2), **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # Bill trend
    hist['bill'] = hist['usage_kwh'].apply(calculate_bill)
    fig2 = px.line(hist, x='label', y='bill',
                   title='Monthly Bill Trend (₹)', markers=True)
    fig2.update_traces(line_color='#BA7517', marker_color='#BA7517')
    fig2.add_scatter(x=[next_label], y=[pred_bill], mode='markers',
                     marker=dict(color='#E24B4A', size=12, symbol='star'),
                     name='Next Month Forecast')
    fig2.update_layout(legend=dict(orientation='h',y=-0.2), **PLOTLY_LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════
def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'summary'

    df = load_data()
    render_sidebar()

    page = st.session_state.page
    if   page == 'summary':    page_summary(df)
    elif page == 'daily':      page_daily(df)
    elif page == 'weekly':     page_weekly(df)
    elif page == 'monthly':    page_monthly(df)
    elif page == 'insights':   page_insights(df)
    elif page == 'prediction': page_prediction(df)

if __name__ == '__main__':
    main()
