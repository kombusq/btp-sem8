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
    load_dotenv(_env_file)
    with open(_env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" not in line or line.startswith("#"):
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'").strip("\r\n")
            if k in ("GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "MODEL") and v:
                os.environ[k] = v

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
import streamlit as st

def _escape_html(s: str) -> str:
    return html.escape(s).replace("\n", "<br/>") if s else ""

st.set_page_config(
    page_title="Digital Investment Office",
    page_icon="📊",
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

st.markdown('<p class="main-header">📊 Digital Investment Office</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Autonomous Trading & Portfolio Optimization Agent — Multi-Agent Investment Committee</p>',
    unsafe_allow_html=True,
)

_model = (os.environ.get("MODEL") or "groq/llama-3.1-70b-versatile").lower()
_needs_groq = _model.startswith("groq/")
if _needs_groq and not os.environ.get("GROQ_API_KEY"):
    st.warning(
        "**GROQ_API_KEY** is not set (required for current MODEL). Add it to `.env` or set [console.groq.com/keys](https://console.groq.com/keys)."
    )
elif not _needs_groq:
    _m = os.environ.get("MODEL", "groq/llama-3.1-70b-versatile")
    st.sidebar.caption(f"LLM: `{_m}`" + (" (local)" if _m.startswith("ollama/") else ""))
    if _m.startswith("ollama/"):
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

if run and ticker:
    import threading
    import time
    from trading_agent.crew import run_crew
    _progress_file = root / ".crewai_storage" / "workflow_progress.txt"
    _progress_file.parent.mkdir(parents=True, exist_ok=True)
    _result_holder = []
    _error_holder = []

    def _run():
        try:
            _result_holder.append(run_crew(ticker, api_key=os.environ.get("GROQ_API_KEY"), progress_file=_progress_file))
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
        time.sleep(0.6)
    th.join()

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

elif run and not ticker:
    st.warning("Please enter a stock ticker in the sidebar.")

else:
    st.info("Enter a stock ticker in the sidebar and click **Run committee** to get a Buy/Sell/Hold recommendation and full explainability.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Agent communication**")
if "last_crew_result" in st.session_state:
    r = st.session_state["last_crew_result"]
    for label, key, css in [
        ("Researcher (sentiment)", "research_output", "researcher"),
        ("Quant (technical)", "quant_output", "quant"),
        ("Portfolio Manager (decision)", "decision_output", "pm"),
    ]:
        text = (r.get(key) or "").strip() or "(no output)"
        if len(text) > 500:
            text = text[:497] + "..."
        st.sidebar.markdown(
            f'<div class="comm-panel {css}"><div class="comm-agent">{label}</div>{_escape_html(text)}</div>',
            unsafe_allow_html=True,
        )
else:
    st.sidebar.caption("Run committee to see agent messages.")
