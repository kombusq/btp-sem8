"""
Streamlit dashboard for the Autonomous Trading & Portfolio Optimization Agent.
User inputs a stock ticker and sees the committee's decision plus full explainability (agent outputs).
"""

import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent

# Load .env so MODEL and API keys (GROQ, OpenAI, etc.) are available
_env_file = root / ".env"
if _env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_file, override=True)

# Ensure src is on path when running from project root
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

# Disable CrewAI telemetry to avoid signal handler threading issues
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

# Use a writable project directory for CrewAI task-output DB (avoids "readonly database" errors)
_crewai_storage = root / ".crewai_storage"
_crewai_storage.mkdir(parents=True, exist_ok=True)
import crewai.utilities.paths as _crewai_paths
_original_db_storage_path = _crewai_paths.db_storage_path
def _writable_db_storage_path():
    return str(_crewai_storage)
_crewai_paths.db_storage_path = _writable_db_storage_path

import html
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go

def _escape_html(s: str) -> str:
    return html.escape(s).replace("\n", "<br/>") if s else ""

st.set_page_config(
    page_title="Autonomous Trading & Portfolio Optimization Agent",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for a clean, professional dashboard
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #5a6c7d;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .recommendation-box {
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .recommendation-buy { background: #d4edda; color: #155724; border-left: 4px solid #28a745; }
    .recommendation-sell { background: #f8d7da; color: #721c24; border-left: 4px solid #dc3545; }
    .recommendation-hold { background: #fff3cd; color: #856404; border-left: 4px solid #ffc107; }
    .agent-block {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    .agent-title { font-weight: 600; color: #1e3a5f; margin-bottom: 0.5rem; }
    .stButton > button {
        background: #1e3a5f;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
    }
    .stButton > button:hover { background: #2c5282; color: white; }
    .workflow-step { padding: 0.5rem 0.75rem; margin: 0.25rem 0; border-radius: 8px; font-size: 0.9rem; }
    .workflow-step.pending { color: #868e96; background: transparent; }
    .workflow-step.running { color: #0d6efd; background: #cfe2ff; border-left: 4px solid #0d6efd; font-weight: 600; }
    .workflow-step.done { color: #198754; background: #d1e7dd; border-left: 4px solid #198754; }
    .comm-panel { background: #f1f3f5; border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 0.5rem; border-left: 4px solid #1e3a5f; font-size: 0.85rem; max-height: 140px; overflow-y: auto; }
    .comm-panel .comm-agent { font-weight: 700; color: #1e3a5f; margin-bottom: 0.25rem; }
    .comm-panel.researcher { border-left-color: #0d6efd; }
    .comm-panel.quant { border-left-color: #6f42c1; }
    .comm-panel.pm { border-left-color: #198754; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">💹 Autonomous Trading & Portfolio Optimization Agent</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Multi-Agent Investment Committee</p>',
    unsafe_allow_html=True,
)

_model = (os.environ.get("MODEL") or "groq/llama-3.1-70b-versatile").strip()
_api_key = (os.environ.get("API_KEY") or "").strip()
_is_local = _model.lower().startswith("ollama/")
if not _api_key and not _is_local:
    st.warning(
        "**API_KEY** is not set in `.env`. Add it for your chosen provider (see `.env` comments)."
    )
st.sidebar.caption(f"LLM: `{_model}`" + (" (local)" if _is_local else ""))
if _is_local:
    st.sidebar.caption("Ensure **Ollama** is running (open the app or `ollama serve`).")

@st.cache_data(ttl=3600, show_spinner=False)
def get_company_info(t: str):
    import yfinance as yf
    import requests
    info = {"name": "", "logo": ""}
    if not t:
        return info
    try:
        data = yf.Ticker(t).info
        info["name"] = data.get("longName", "")
    except Exception:
        pass
    
    urls = [
        f"https://financialmodelingprep.com/image-stock/{t}.png",
        f"https://assets.parqet.com/logos/symbol/{t}?format=png",
        f"https://companiesmarketcap.com/img/company-logos/64/{t}.webp"
    ]
    for url in urls:
        try:
            r = requests.head(url, timeout=2)
            if r.status_code == 200:
                info["logo"] = url
                break
        except Exception:
            continue
    return info

ticker = st.sidebar.text_input(
    "Stock ticker",
    value="AAPL",
    placeholder="e.g. AAPL, MSFT, GOOGL",
    help="Enter a valid stock ticker symbol.",
).strip().upper()

if ticker:
    c_info = get_company_info(ticker)
    if c_info["name"] or c_info["logo"]:
        cols = st.sidebar.columns([1, 4], vertical_alignment="center")
        with cols[0]:
            if c_info["logo"]:
                st.image(c_info["logo"], width="stretch")
            else:
                st.markdown("<h3 style='margin:0; padding:0; text-align:center;'>🏢</h3>", unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"**{c_info['name'] or ticker}**")

st.sidebar.markdown("---")
st.sidebar.markdown("**Workflow**")

# ── Stock Dashboard (price chart + key metrics) ────────────────────
@st.cache_data(ttl=900, show_spinner=False)
def get_stock_dashboard_data(t: str, period: str = "6mo"):
    """Fetch OHLCV + indicators for the dashboard chart."""
    import yfinance as yf
    import pandas as pd
    try:
        tk = yf.Ticker(t)
        hist = tk.history(period=period, interval="1d")
        if hist.empty:
            return None, {}
        df = hist.copy()
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["SMA_50"] = df["Close"].rolling(50).mean()
        # Key metrics
        latest = df["Close"].iloc[-1]
        prev_close = df["Close"].iloc[-2] if len(df) > 1 else latest
        day_change = latest - prev_close
        day_change_pct = (day_change / prev_close * 100) if prev_close else 0
        high_52w = df["Close"].max()
        low_52w = df["Close"].min()
        vol_latest = int(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0
        volatility = df["Close"].pct_change().rolling(20).std().iloc[-1] * (252 ** 0.5) * 100 if len(df) >= 21 else 0
        metrics = {
            "price": latest,
            "change": day_change,
            "change_pct": day_change_pct,
            "high": high_52w,
            "low": low_52w,
            "volume": vol_latest,
            "volatility": volatility,
        }
        return df, metrics
    except Exception:
        return None, {}

# ── PDF Report Generator ──────────────────────────────────────────
def generate_pdf_report(ticker_sym: str, result: dict, metrics: dict) -> bytes:
    """Generate a professional PDF report and return as bytes."""
    from fpdf import FPDF

    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(30, 58, 95)
            self.cell(0, 12, "Autonomous Trading & Portfolio Optimization Agent - Report", align="C", new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(30, 58, 95)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(6)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f"Page {self.page_no()} | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(30, 58, 95)
    pdf.cell(0, 14, f"{ticker_sym} - Investment Analysis", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 8, f"Report generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # Recommendation box
    rec = result.get("recommendation", "HOLD")
    if rec == "BUY":
        pdf.set_fill_color(212, 237, 218)
        pdf.set_text_color(21, 87, 36)
    elif rec == "SELL":
        pdf.set_fill_color(248, 215, 218)
        pdf.set_text_color(114, 28, 36)
    else:
        pdf.set_fill_color(255, 243, 205)
        pdf.set_text_color(133, 100, 4)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 14, f"  Recommendation: {rec}", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Market metrics
    pdf.set_text_color(30, 58, 95)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Market Overview", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)
    if metrics:
        pdf.cell(95, 7, f"Price: ${metrics.get('price', 0):.2f}  |  Change: {metrics.get('change_pct', 0):+.2f}%")
        pdf.cell(95, 7, f"Period High: ${metrics.get('high', 0):.2f}  |  Low: ${metrics.get('low', 0):.2f}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(95, 7, f"Volatility (ann.): {metrics.get('volatility', 0):.1f}%")
        pdf.cell(95, 7, f"Volume: {metrics.get('volume', 0):,}", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.cell(0, 7, "Market data not available.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Agent outputs
    sections = [
        ("Researcher - Sentiment Report", result.get("research_output", "")),
        ("Quant - Technical Analysis", result.get("quant_output", "")),
        ("Portfolio Manager - Decision & Justification", result.get("decision_output", "")),
    ]
    for title, body in sections:
        pdf.set_text_color(30, 58, 95)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(50, 50, 50)
        body_clean = (body or "(no output)").strip()
        if len(body_clean) > 2500:
            body_clean = body_clean[:2500] + "\n... [truncated]"
        pdf.multi_cell(0, 5, body_clean, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    return bytes(pdf.output())

if ticker:
    with st.spinner("Loading stock data..."):
        _period_options = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y"}
        _sel_period = st.radio("Chart period", list(_period_options.keys()), index=2, horizontal=True)
        df_chart, metrics = get_stock_dashboard_data(ticker, _period_options[_sel_period])

    if df_chart is not None and not df_chart.empty:
        # Key metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Price", f"${metrics['price']:.2f}", f"{metrics['change']:+.2f} ({metrics['change_pct']:+.2f}%)")
        m2.metric("Period High", f"${metrics['high']:.2f}")
        m3.metric("Period Low", f"${metrics['low']:.2f}")
        m4.metric("Volatility (ann.)", f"{metrics['volatility']:.1f}%")

        # Candlestick chart with SMA overlays
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_chart.index, open=df_chart["Open"], high=df_chart["High"],
            low=df_chart["Low"], close=df_chart["Close"], name="Price",
            increasing_line_color="#28a745", decreasing_line_color="#dc3545",
        ))
        if "SMA_20" in df_chart.columns:
            fig.add_trace(go.Scatter(
                x=df_chart.index, y=df_chart["SMA_20"], mode="lines",
                name="SMA 20", line=dict(color="#0d6efd", width=1.5),
            ))
        if "SMA_50" in df_chart.columns:
            fig.add_trace(go.Scatter(
                x=df_chart.index, y=df_chart["SMA_50"], mode="lines",
                name="SMA 50", line=dict(color="#ff8c00", width=1.5),
            ))
        fig.update_layout(
            title=f"{ticker} — Price & Moving Averages",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            height=420,
            margin=dict(l=40, r=20, t=50, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, width="stretch")

        # Volume bar chart
        vol_colors = ["#28a745" if c >= o else "#dc3545" for c, o in zip(df_chart["Close"], df_chart["Open"])]
        fig_vol = go.Figure(go.Bar(x=df_chart.index, y=df_chart["Volume"], marker_color=vol_colors, name="Volume"))
        fig_vol.update_layout(
            title="Volume",
            yaxis_title="Shares",
            template="plotly_white",
            height=180,
            margin=dict(l=40, r=20, t=40, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_vol, width="stretch")
    elif ticker:
        st.warning(f"Could not load chart data for **{ticker}**. Check the ticker symbol.")

    st.markdown("---")

WORKFLOW_STEPS = [
    ("1", "Researcher", "News & sentiment"),
    ("2", "Quant", "Technical analysis"),
    ("3", "Portfolio Manager", "Final decision"),
    ("4", "Explainability", "Full agent reasoning"),
]

def render_workflow_steps(current: int):
    """current: 0 = none, 1–3 = that step running, 4 = all done."""
    html = ""
    for i, (num, name, desc) in enumerate(WORKFLOW_STEPS, 1):
        if i < current:
            cls = "done"
            prefix = "✓ "
        elif i == current:
            cls = "running"
            prefix = "▶ "
        else:
            cls = "pending"
            prefix = ""
        html += f'<div class="workflow-step {cls}">{prefix}{num}. {name} — {desc}</div>'
    return html

_workflow_placeholder = st.sidebar.empty()
_workflow_placeholder.markdown(render_workflow_steps(0), unsafe_allow_html=True)

run = st.sidebar.button("Run committee", width="stretch")

# Sidebar agent communication section — uses placeholders for live updates
st.sidebar.markdown("---")
st.sidebar.markdown("**Agent communication**")
_AGENT_PANELS = [
    ("Researcher (sentiment)", "researcher", "researcher"),
    ("Quant (technical)", "quant", "quant"),
    ("Portfolio Manager (decision)", "portfolio_manager", "pm"),
]
_comm_placeholders = [st.sidebar.empty() for _ in _AGENT_PANELS]

def _render_comm_panels(outputs: dict):
    """Update sidebar comm panels from an outputs dict (keys: researcher, quant, portfolio_manager)."""
    for ph, (label, key, css) in zip(_comm_placeholders, _AGENT_PANELS):
        text = (outputs.get(key) or "").strip()
        if text:
            if len(text) > 500:
                text = text[:497] + "..."
            ph.markdown(
                f'<div class="comm-panel {css}"><div class="comm-agent">{label}</div>{_escape_html(text)}</div>',
                unsafe_allow_html=True,
            )
        else:
            ph.markdown(
                f'<div class="comm-panel {css}" style="opacity:0.4"><div class="comm-agent">{label}</div><em>waiting...</em></div>',
                unsafe_allow_html=True,
            )

# Show previous run results or empty panels
if "last_crew_result" in st.session_state and not (run and ticker):
    _prev = st.session_state["last_crew_result"]
    _render_comm_panels({
        "researcher": _prev.get("research_output", ""),
        "quant": _prev.get("quant_output", ""),
        "portfolio_manager": _prev.get("decision_output", ""),
    })
else:
    for ph, (label, _key, css) in zip(_comm_placeholders, _AGENT_PANELS):
        ph.caption("Run committee to see agent messages.")

if run and ticker:
    import json as _json
    import threading
    import time
    from trading_agent.crew import run_crew
    _progress_file = root / ".crewai_storage" / "workflow_progress.txt"
    _outputs_file = root / ".crewai_storage" / "workflow_progress_outputs.json"
    _progress_file.parent.mkdir(parents=True, exist_ok=True)
    _result_holder = []
    _error_holder = []

    # Show waiting state in comm panels
    _render_comm_panels({})

    def _run():
        try:
            _result_holder.append(run_crew(ticker, api_key=_api_key, progress_file=_progress_file))
        except Exception as e:
            _error_holder.append(e)

    th = threading.Thread(target=_run)
    th.start()
    while th.is_alive():
        try:
            step = int(_progress_file.read_text(encoding="utf-8").strip() or "0")
        except Exception:
            step = 0
        _workflow_placeholder.markdown(render_workflow_steps(step), unsafe_allow_html=True)
        # Live-update agent comm panels from outputs file
        try:
            if _outputs_file.exists():
                _live_outputs = _json.loads(_outputs_file.read_text(encoding="utf-8"))
                _render_comm_panels(_live_outputs)
        except Exception:
            pass
        time.sleep(0.6)
    th.join()

    # Final update of comm panels
    try:
        if _outputs_file.exists():
            _live_outputs = _json.loads(_outputs_file.read_text(encoding="utf-8"))
            _render_comm_panels(_live_outputs)
    except Exception:
        pass

    try:
        _workflow_placeholder.markdown(render_workflow_steps(4), unsafe_allow_html=True)
    except Exception:
        pass

    if _error_holder:
        e = _error_holder[0]
        err_str = str(e).lower()
        if "connection refused" in err_str or ("ollama" in err_str and "refused" in err_str):
            st.error(
                "**Ollama is not running.** Start Ollama first, then try again:\n\n"
                "1. **Open the Ollama app** from Applications (macOS), or\n"
                "2. In a terminal run: `ollama serve`\n\n"
                "Keep Ollama running in the background and click **Run committee** again."
            )
        else:
            st.error(f"Committee run failed: {e}")
        raise
    result = _result_holder[0]
    st.session_state["last_crew_result"] = result
    # Update sidebar comm panels with final data
    _render_comm_panels({
        "researcher": result.get("research_output", ""),
        "quant": result.get("quant_output", ""),
        "portfolio_manager": result.get("decision_output", ""),
    })
    st.success("Committee run completed.")

    rec = result.get("recommendation", "HOLD")
    css_class = "recommendation-buy" if rec == "BUY" else "recommendation-sell" if rec == "SELL" else "recommendation-hold"
    st.markdown(f'<div class="recommendation-box {css_class}">Recommendation: {rec}</div>', unsafe_allow_html=True)
    st.markdown("**Justification**")
    st.write(result.get("justification", ""))

    st.markdown("---")
    st.markdown("### Explainability — Committee reasoning (XAI)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="agent-block"><span class="agent-title">🔬 Researcher (Sentiment)</span></div>', unsafe_allow_html=True)
        st.text_area("Research output", value=result.get("research_output", ""), height=220, disabled=True, label_visibility="collapsed")
    with col2:
        st.markdown('<div class="agent-block"><span class="agent-title">📐 Quant (Technical)</span></div>', unsafe_allow_html=True)
        st.text_area("Quant output", value=result.get("quant_output", ""), height=220, disabled=True, label_visibility="collapsed")
    with col3:
        st.markdown('<div class="agent-block"><span class="agent-title">📋 Portfolio Manager (Decision)</span></div>', unsafe_allow_html=True)
        st.text_area("Decision output", value=result.get("decision_output", ""), height=220, disabled=True, label_visibility="collapsed")

    st.markdown("---")
    with st.expander("View full raw outputs"):
        st.json({k: v for k, v in result.items() if k != "justification" or True})

    # ── PDF Report Download ────────────────────────────────────────
    _pdf_metrics = metrics if "metrics" in dir() else {}
    try:
        pdf_bytes = generate_pdf_report(ticker, result, _pdf_metrics)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"{ticker}_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
        )
    except Exception as _pdf_err:
        st.warning(f"Could not generate PDF: {_pdf_err}")

elif run and not ticker:
    st.warning("Please enter a stock ticker in the sidebar.")

else:
    st.info("Enter a stock ticker in the sidebar and click **Run committee** to get a Buy/Sell/Hold recommendation and full explainability.")
