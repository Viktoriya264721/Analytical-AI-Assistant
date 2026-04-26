"""Streamlit dashboard components: KPI cards, charts, and layout helpers."""

import streamlit as st
import plotly.graph_objects as go


PRIMARY   = "#334663"
PRIMARY_L = "#6582AA"
ACCENT    = "#83A2CD"
DARK      = "#1e2d42"
GRAY      = "#6582AA"
BORDER    = "#E4ECF7"
SUCCESS   = "#4B9B7E"
DANGER    = "#C75450"
WARNING   = "#D4915E"

CHART_COLORS = ["#6582AA", "#83A2CD", "#4B9B7E", "#D4915E", "#7B6FAA"]


def inject_custom_css() -> None:
    """Inject global CSS for KPI cards, tabs, efficiency bars, and responsive layout."""
    st.markdown("""
    <style>
    /* ---- KPI Grid ---- */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 1.6rem;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E4ECF7;
        border-radius: 16px;
        padding: 18px 20px 14px;
        box-shadow: 0 2px 10px rgba(51,70,99,0.06);
        transition: box-shadow 0.2s ease, transform 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #83A2CD, #A3C5F1);
        border-radius: 16px 16px 0 0;
    }
    .kpi-card:hover {
        box-shadow: 0 6px 20px rgba(51,70,99,0.13);
        transform: translateY(-2px);
    }
    .kpi-header {
        display: flex;
        align-items: center;
        gap: 7px;
        margin-bottom: 10px;
    }
    .kpi-icon { font-size: 15px; line-height: 1; }
    .kpi-label {
        font-size: 11px;
        color: #8BA4C8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: 700;
        color: #1e2d42;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    .kpi-value-primary {
        font-size: 30px;
        font-weight: 800;
        color: #1e2d42;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    .kpi-unit {
        font-size: 14px;
        font-weight: 500;
        color: #8BA4C8;
        margin-left: 3px;
    }
    .kpi-delta {
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .kpi-delta.positive { color: #4B9B7E; }
    .kpi-delta.negative { color: #C75450; }
    .kpi-delta.neutral  { color: #8BA4C8; }

    /* ---- Section header ---- */
    .fin-section {
        margin: 1.4rem 0 0.8rem;
        line-height: 1.4;
    }
    .fin-section-title {
        font-size: 16px;
        font-weight: 700;
        color: #334663;
    }
    .fin-section-sub {
        font-size: 13px;
        color: #9EB3CB;
        font-weight: 400;
        margin-left: 6px;
    }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 3px !important;
        background: #F2F6FB !important;
        border-radius: 10px !important;
        padding: 3px !important;
        border-bottom: none !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 6px 20px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #6582AA !important;
        background: transparent !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
        color: #334663 !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 4px rgba(51,70,99,0.12) !important;
    }
    .stTabs [data-baseweb="tab-border"]    { display: none !important; }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }

    /* ---- Efficiency list ---- */
    .eff-list { padding: 2px 0; }
    .eff-item { margin-bottom: 14px; }
    .eff-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 5px;
    }
    .eff-square {
        width: 10px; height: 10px;
        border-radius: 2px;
        flex-shrink: 0;
        display: inline-block;
    }
    .eff-name {
        font-size: 13px;
        color: #334663;
        font-weight: 500;
        flex: 1;
    }
    .eff-pct {
        font-size: 13px;
        font-weight: 700;
        color: #334663;
        min-width: 34px;
        text-align: right;
    }
    .eff-track {
        height: 7px;
        background: #EDF1F7;
        border-radius: 4px;
        overflow: hidden;
    }
    .eff-fill { height: 100%; border-radius: 4px; }

    /* ---- Summary row ---- */
    .summary-row {
        font-size: 13px;
        color: #6582AA;
        margin-bottom: 10px;
    }
    .summary-row b { color: #1e2d42; }

    /* ---- Responsive ---- */
    @media (max-width: 700px) {
        .kpi-grid { grid-template-columns: repeat(2, 1fr); }
        .kpi-value { font-size: 20px; }
        .kpi-value-primary { font-size: 24px; }
    }
    </style>
    """, unsafe_allow_html=True)


def _num(value) -> float:
    """Safely cast *value* to float; return 0.0 on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt(value, decimals: int = 0) -> str:
    """Format a number with a narrow-space thousands separator."""
    v = _num(value)
    if decimals == 0:
        return f"{v:,.0f}".replace(",", "\u202f")
    return f"{v:,.{decimals}f}".replace(",", "\u202f")


def _bar_label(v: float) -> str:
    """Return a compact label string (e.g. '12K', '1.2M') for chart bar annotations."""
    if not v:
        return ""
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v / 1_000:.0f}K"
    return f"{v:.0f}"


def _sparkline(values: list, width: int = 120, height: int = 36,
               color: str = "#83A2CD") -> str:
    """Generate an inline SVG sparkline area chart.

    Args:
        values: Numeric series (last 10 points are used).
        width: SVG canvas width in pixels.
        height: SVG canvas height in pixels.
        color: Stroke and fill base colour (hex).

    Returns:
        SVG string, or empty string when fewer than 2 data points are available.
    """
    vals = [float(v or 0) for v in (values or [])]
    vals = vals[-10:]
    if len(vals) < 2:
        return ""
    mn, mx = min(vals), max(vals)
    rng = mx - mn if abs(mx - mn) > 1e-9 else max(abs(mx), 1)
    pad = 3
    usable_h = height - pad * 2
    n = len(vals)
    step_x = width / (n - 1)
    pts = [
        (round(i * step_x, 1), round(pad + usable_h * (1 - (v - mn) / rng), 1))
        for i, v in enumerate(vals)
    ]
    path = "M " + " L ".join(f"{x},{y}" for x, y in pts)
    lx, _ = pts[-1]
    fill_d = f"{path} L {lx},{height} L 0,{height} Z"
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block;">'
        f'<path d="{fill_d}" fill="{color}1A"/>'
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.8" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f'</svg>'
    )


def _delta_html(delta, delta_pct=None, is_margin: bool = False,
                inverse: bool = False) -> str:
    """Render a delta indicator as an HTML badge.

    Args:
        delta: Absolute change value.
        delta_pct: Optional percentage change.
        is_margin: When True formats the value as percentage points.
        inverse: When True, a positive delta is coloured red (e.g. for expenses).

    Returns:
        HTML string with the appropriate CSS class (positive / negative / neutral).
    """
    if delta is None or delta == 0:
        return '<div class="kpi-delta neutral">—</div>'
    is_pos = delta > 0
    if inverse:
        is_pos = not is_pos
    cls   = "positive" if is_pos else "negative"
    arrow = "▲" if delta > 0 else "▼"
    if is_margin:
        sign = "+" if delta > 0 else ""
        text = f"{arrow} {sign}{_fmt(abs(delta), 1)} п.п."
    else:
        abs_str = _fmt(abs(delta))
        if delta_pct is not None:
            sign = "+" if delta_pct > 0 else ""
            text = f"{arrow} {abs_str} грн ({sign}{delta_pct:.1f}%)"
        else:
            text = f"{arrow} {abs_str} грн"
    return f'<div class="kpi-delta {cls}">{text}</div>'


def _section_header(title: str, subtitle: str = "") -> None:
    """Render a styled section heading with an optional subtitle.

    Args:
        title: Main heading text.
        subtitle: Optional parenthetical subtitle shown in lighter weight.
    """
    sub = f'<span class="fin-section-sub">({subtitle})</span>' if subtitle else ""
    st.markdown(
        f'<div class="fin-section">'
        f'<span class="fin-section-title">{title}</span>{sub}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_kpi_cards(kpi: dict, history: dict) -> None:
    """Render the four top-level KPI cards (Revenue, EBIT, Cash Flow, Margin).

    Each card displays the current value, a month-over-month delta badge,
    and a sparkline built from historical data.

    Args:
        kpi: KPI dict from :func:`build_dashboard_data`.
        history: History dict with per-metric time series lists.
    """
    cards = [
        dict(label="Дохід",            icon="💰", primary=True,
             value=kpi.get("revenue", 0),
             delta=kpi.get("revenue_delta"), delta_pct=kpi.get("revenue_delta_pct"),
             unit="грн", is_margin=False, spark="revenue"),
        dict(label="Операц. прибуток", icon="📈", primary=False,
             value=kpi.get("ebit", 0),
             delta=kpi.get("ebit_delta"), delta_pct=kpi.get("ebit_delta_pct"),
             unit="грн", is_margin=False, spark="ebit"),
        dict(label="Грошовий потік",   icon="💳", primary=False,
             value=kpi.get("operating_cf", 0),
             delta=kpi.get("operating_cf_delta"), delta_pct=kpi.get("operating_cf_delta_pct"),
             unit="грн", is_margin=False, spark="operating_cf"),
        dict(label="Операц. маржа",    icon="🎯", primary=False,
             value=kpi.get("operating_margin_pct", 0),
             delta=kpi.get("operating_margin_pct_delta"), delta_pct=None,
             unit="%", is_margin=True, spark="operating_margin"),
    ]

    html = '<div class="kpi-grid">'
    for c in cards:
        val_cls = "kpi-value-primary" if c["primary"] else "kpi-value"
        val_str = f"{_num(c['value']):.1f}" if c["is_margin"] else _fmt(c["value"])
        delta_block = _delta_html(c["delta"], c["delta_pct"], c["is_margin"])

        d = c["delta"]
        spark_color = (
            "#4B9B7E" if (d and d > 0)
            else "#C75450" if (d and d < 0)
            else "#83A2CD"
        )
        spark_svg = _sparkline(history.get(c["spark"], []),
                               width=140, height=34, color=spark_color)

        html += f"""
        <div class="kpi-card">
          <div class="kpi-header">
            <span class="kpi-label">{c['label']}</span>
          </div>
          <div class="{val_cls}">{val_str}<span class="kpi-unit">{c['unit']}</span></div>
          {delta_block}
          {spark_svg}
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_finances_line_chart(history: dict, last_n: int = 6) -> None:
    """Render a multi-line chart for Revenue, EBIT, and Operating CF.

    Args:
        history: History dict with ``months``, ``revenue``, ``ebit``,
            and ``operating_cf`` lists.
        last_n: Number of most recent months to display.
    """
    months     = history.get("months",       [])[-last_n:]
    revenue    = history.get("revenue",      [])[-last_n:]
    ebit       = history.get("ebit",         [])[-last_n:]
    net_cf     = history.get("operating_cf", [])[-last_n:]

    if not months:
        st.info("Недостатньо даних.")
        return

    fig = go.Figure()
    traces = [
        ("Дохід",     revenue, "#5B8FF9", True,  "rgba(91,143,249,0.07)"),
        ("EBIT",      ebit,    "#4B9B7E", False, None),
        ("Cash Flow", net_cf,  "#9B7BB0", False, None),
    ]
    for name, vals, color, with_fill, fill_color in traces:
        if not vals or all(v == 0 for v in vals):
            continue
        fig.add_trace(go.Scatter(
            x=months, y=vals, name=name,
            mode="lines+markers",
            line=dict(color=color, width=2.5, shape="spline", smoothing=0.4),
            marker=dict(size=7, color=color, line=dict(color="#fff", width=1.5)),
            fill="tozeroy" if with_fill else "none",
            fillcolor=fill_color,
            hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:,.0f}} грн<extra></extra>",
        ))

    all_vals = [v for s in [revenue, ebit, net_cf] for v in (s or []) if v]
    y_max = max(all_vals) * 1.15 if all_vals else 1

    fig.update_layout(
        height=320,
        margin=dict(l=0, r=20, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=12)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        font=dict(color=DARK, size=12),
        yaxis=dict(range=[0, y_max]),
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=11))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(51,70,99,0.07)",
                     tickformat=",.0f", tickfont=dict(size=11))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_revenue_expenses_chart(history: dict, last_n: int = 6) -> None:
    """Render a grouped bar chart comparing Revenue vs Expenses.

    Args:
        history: History dict with ``months``, ``revenue``, and ``expenses`` lists.
        last_n: Number of most recent months to display.
    """
    months   = history.get("months",   [])[-last_n:]
    revenue  = history.get("revenue",  [])[-last_n:]
    expenses = history.get("expenses", [])[-last_n:]

    if not months:
        st.info("Недостатньо даних для графіка.")
        return

    fig = go.Figure()
    if revenue and any(v != 0 for v in revenue):
        fig.add_trace(go.Bar(
            x=months, y=revenue, name="Дохід",
            marker_color="#5B8FF9", marker_line_width=0,
            text=[_bar_label(v) for v in revenue],
            textposition="outside", textfont=dict(size=11, color=DARK),
            hovertemplate="<b>Дохід</b><br>%{x}: %{y:,.0f} грн<extra></extra>",
        ))
    if expenses and any(v != 0 for v in expenses):
        fig.add_trace(go.Bar(
            x=months, y=expenses, name="Витрати",
            marker_color="#FF7875", marker_line_width=0,
            text=[_bar_label(v) for v in expenses],
            textposition="outside", textfont=dict(size=11, color=DARK),
            hovertemplate="<b>Витрати</b><br>%{x}: %{y:,.0f} грн<extra></extra>",
        ))

    all_vals = [v for s in [revenue, expenses] for v in (s or []) if v]
    y_max = max(all_vals) * 1.22 if all_vals else 1

    fig.update_layout(
        barmode="group",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5, font=dict(size=12)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=DARK, size=12),
        hovermode="x unified",
        yaxis=dict(range=[0, y_max]),
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=11))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(51,70,99,0.07)",
                     tickfont=dict(size=11), tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_margin_chart(history: dict, last_n: int = 6) -> None:
    """Render a colour-coded bar chart of operating margin percentages.

    Bars are green (≥20%), yellow (≥10%), or red (<10%).

    Args:
        history: History dict with ``months`` and ``operating_margin`` lists.
        last_n: Number of most recent months to display.
    """
    months  = history.get("months",           [])[-last_n:]
    margins = history.get("operating_margin", [])[-last_n:]

    if not months or not margins or all(v == 0 for v in margins):
        st.info("Немає даних про маржинальність.")
        return

    colors = ["#52C41A" if v >= 20 else "#FAAD14" if v >= 10 else "#FF4D4F"
              for v in margins]
    y_max = max((v for v in margins if v), default=1) * 1.25

    fig = go.Figure(go.Bar(
        x=months, y=margins,
        marker_color=colors, marker_line_width=0,
        text=[f"{v:.1f}%" if v else "" for v in margins],
        textposition="outside", textfont=dict(size=12, color=DARK),
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=DARK, size=12),
        yaxis=dict(range=[0, y_max]),
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=11))
    fig.update_yaxes(ticksuffix="%", showgrid=True,
                     gridcolor="rgba(51,70,99,0.07)", tickfont=dict(size=11))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_efficiency_chart(efficiency: dict) -> None:
    """Render HTML horizontal progress bars for per-specialist efficiency.

    Args:
        efficiency: Mapping of specialist name to efficiency percentage (0–100).
    """
    if not efficiency:
        st.info("Немає даних про ефективність.")
        return

    items = sorted(efficiency.items(), key=lambda x: x[1], reverse=True)
    html = '<div class="eff-list">'
    for name, value in items:
        color = "#4B9B7E" if value >= 80 else "#D4915E" if value >= 50 else "#C75450"
        w = min(float(value), 100)
        html += f"""
        <div class="eff-item">
          <div class="eff-header">
            <span class="eff-square" style="background:{color}"></span>
            <span class="eff-name">{name}</span>
            <span class="eff-pct">{value:.0f}%</span>
          </div>
          <div class="eff-track">
            <div class="eff-fill" style="width:{w:.0f}%;background:{color}"></div>
          </div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_service_types_chart(service_types: dict) -> None:
    """Render a vertical bar chart of service unit counts by activity type.

    Args:
        service_types: Mapping of activity type name to unit count.
    """
    if not service_types:
        st.info("Немає даних про типи послуг.")
        return

    items  = sorted(service_types.items(), key=lambda x: _num(x[1]), reverse=True)
    labels = [i[0] for i in items]
    values = [_num(i[1]) for i in items]

    bar_colors = ["#7B6FAA", "#9B8FC0", "#6582AA", "#83A2CD", "#A78BBF", "#C4A8D8"]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker=dict(
            color=bar_colors[:len(labels)],
            line=dict(width=0),
        ),
        text=[f"{v:.0f}" for v in values],
        textposition="outside",
        textfont=dict(size=10, color=DARK),
        hovertemplate="%{x}: %{y:.0f} од.<extra></extra>",
    ))
    y_max = max(values) * 1.3 if values else 1
    fig.update_layout(
        height=270,
        margin=dict(l=0, r=0, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=DARK, size=11),
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(range=[0, y_max], showgrid=True,
                   gridcolor="rgba(51,70,99,0.07)", tickfont=dict(size=10)),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_expenses_pie(expenses: dict) -> None:
    """Render a donut chart for the top-3 expense categories.

    Args:
        expenses: Dict with ``total`` and ``top_3`` keys
            (as returned by :func:`build_dashboard_data`).
    """
    top3  = expenses.get("top_3", {})
    total = expenses.get("total", 0)
    if not top3:
        st.info("Немає даних про витрати.")
        return

    labels = list(top3.keys())
    values = list(top3.values())

    expense_colors = ["#C75450", "#D4915E", "#E8B84B", "#E07060", "#C49A6C"]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55,
        textinfo="label+percent",
        textfont=dict(size=11),
        marker=dict(colors=expense_colors[:len(labels)],
                    line=dict(color="#fff", width=2)),
        hovertemplate="%{label}: %{value:,.0f} грн (%{percent})<extra></extra>",
    ))
    fig.add_annotation(
        text=(f"<b>{_fmt(total)}</b><br>"
              f"<span style='font-size:10px;color:{GRAY}'>грн</span>"),
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color=DARK), align="center",
    )
    fig.update_layout(
        height=270, margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
        font=dict(color=DARK, size=12),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_dashboard(data: dict) -> None:
    """Render the full financial dashboard for a selected month.

    Displays KPI cards, tabbed financial charts (finances / expenses / margin),
    and three side-by-side panels for expenses, efficiency, and service types.

    Args:
        data: Dashboard data dict returned by :func:`build_dashboard_data`.
    """
    inject_custom_css()

    if data.get("empty"):
        st.warning("Немає даних для обраного місяця.")
        return

    if data.get("names_missing"):
        st.warning(
            "Таблиця осіб (persons) порожня — відображаються анонімні ID замість реальних імен. "
            "Запустіть pipeline (`python main.py`), щоб відновити маппінг імен."
        )

    target_month = data["target_month"]
    history      = data["history"]

    st.markdown(
        f'<div style="font-size:22px;font-weight:700;color:#334663;margin-bottom:2px;">'
        f'Огляд: {target_month}</div>'
        f'<div style="font-size:13px;color:#6582AA;margin-bottom:1.2rem;">'
        f'Порівняння з попереднім місяцем</div>',
        unsafe_allow_html=True,
    )

    render_kpi_cards(data["kpi"], history)

    _section_header("Фінансова динаміка", "Дохід, Прибуток, Cash Flow")
    tab1, tab2, tab3 = st.tabs(["Фінанси", "Витрати", "Маржа"])
    with tab1:
        render_finances_line_chart(history)
    with tab2:
        render_revenue_expenses_chart(history)
    with tab3:
        render_margin_chart(history)

    st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        _section_header("Структура витрат")
        render_expenses_pie(data["expenses_by_category"])

    with col2:
        _section_header("Ефективність")
        render_efficiency_chart(data["efficiency"])

    with col3:
        _section_header("Типи послуг")
        render_service_types_chart(data.get("service_types", {}))
