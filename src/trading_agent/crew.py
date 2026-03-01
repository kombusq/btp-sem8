"""
Digital Investment Office — Crew definition.
Sequential pipeline: Researcher → Quant → Portfolio Manager.
"""

import os
from pathlib import Path
from typing import Any

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

from trading_agent.tools import MarketDataTool, NewsFetchTool

# Default key and model loaded at import (can be overridden by run_crew(api_key=...))
_crew_dir = Path(__file__).resolve().parent
_project_root = _crew_dir.parent.parent
if (_project_root / ".env").exists():
    load_dotenv(_project_root / ".env", override=True)

# Single MODEL and API_KEY from .env
_default_model = "groq/llama-3.1-70b-versatile"
_llm_model = (os.environ.get("MODEL") or _default_model).strip()

# Map model prefix → provider-specific env var that LiteLLM expects
_PROVIDER_KEY_MAP = {
    "groq":      "GROQ_API_KEY",
    "gemini":    "GOOGLE_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def _set_provider_key(model: str, api_key: str) -> None:
    """Set the provider-specific env var that LiteLLM/CrewAI expects, based on model prefix."""
    prefix = model.split("/")[0].lower() if "/" in model else model.lower()
    env_var = _PROVIDER_KEY_MAP.get(prefix)
    if env_var and api_key:
        os.environ[env_var] = api_key


@CrewBase
class TradingAgentCrew:
    """Autonomous Trading & Portfolio Optimization Agent — Multi-Agent Investment Committee."""

    # CrewBase uses these as paths relative to this file's directory (trading_agent/).
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self, api_key_override: str | None = None):
        model = (os.environ.get("MODEL", _llm_model) or _default_model).strip()
        if model.upper().startswith("MODEL="):
            model = model.split("=", 1)[-1].strip()
        model = model or _default_model

        # Resolve API key: explicit override > API_KEY env var
        api_key = (api_key_override or os.environ.get("API_KEY") or "").strip().strip('"').strip("'").strip("\r\n")

        is_local = model.lower().startswith("ollama/")
        if not api_key and not is_local:
            raise ValueError(
                "API_KEY is not set. Add it to .env (see .env comments for provider links)."
            )

        # Set the provider-specific env var so LiteLLM picks it up
        if api_key:
            _set_provider_key(model, api_key)

        llm_kwargs = dict(
            model=model,
            temperature=0.3,
            max_tokens=1024,
        )
        if api_key:
            llm_kwargs["api_key"] = api_key
        self._llm = LLM(**llm_kwargs)

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],
            llm=self._llm,
            verbose=True,
            tools=[NewsFetchTool()],
            max_iter=3,
            allow_delegation=False,
        )

    @agent
    def quant(self) -> Agent:
        return Agent(
            config=self.agents_config["quant"],
            llm=self._llm,
            verbose=True,
            tools=[MarketDataTool()],
            max_iter=3,
            allow_delegation=False,
        )

    @agent
    def portfolio_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["portfolio_manager"],
            llm=self._llm,
            verbose=True,
            max_iter=2,
            allow_delegation=False,
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],
        )

    @task
    def quant_task(self) -> Task:
        return Task(config=self.tasks_config["quant_task"], context=[])

    @task
    def decision_task(self) -> Task:
        return Task(config=self.tasks_config["decision_task"])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )


def run_crew(ticker: str, api_key: str | None = None, progress_file: str | Path | None = None) -> dict:
    """
    Run the investment committee crew for a given ticker.
    Returns dict with 'recommendation', 'justification', 'research_output', 'quant_output', 'decision_output'.
    If api_key is provided, it is used and set in os.environ so LiteLLM sees it.
    If progress_file is provided, writes current step (1, 2, 3) on task start for UI progress.
    Task outputs are streamed to a sibling JSON file (<progress_file_stem>_outputs.json)
    so the UI can display each agent's result as soon as it finishes.
    """
    import json as _json

    if api_key:
        api_key = api_key.strip().strip('"').strip("'").strip("\r\n")
        if api_key:
            os.environ["API_KEY"] = api_key
    inputs = {"ticker": ticker.strip().upper()}
    progress_path = Path(progress_file) if progress_file else None
    outputs_path = progress_path.with_name(progress_path.stem + "_outputs.json") if progress_path else None
    task_step = [0]  # mutable — incremented by _on_task_started for progress UI
    completed_step = [0]  # separate counter — incremented by _on_task_completed for output mapping
    _task_names = ["researcher", "quant", "portfolio_manager"]  # order matches sequential pipeline

    def _on_task_started(_source: Any, event: Any) -> None:
        task_step[0] += 1
        if progress_path:
            try:
                progress_path.write_text(str(task_step[0]), encoding="utf-8")
            except Exception:
                pass

    def _on_task_completed(_source: Any, event: Any) -> None:
        """Write each task's output to the shared JSON file as it finishes."""
        if not outputs_path:
            return
        try:
            output_text = getattr(event, "output", None)
            if output_text is None:
                output_text = str(event)
            elif hasattr(output_text, "raw"):
                output_text = output_text.raw
            else:
                output_text = str(output_text)
            # Use completion counter (not start counter) so outputs always map correctly
            step_idx = completed_step[0]
            completed_step[0] += 1
            agent_key = _task_names[step_idx] if step_idx < len(_task_names) else f"task_{step_idx}"
            # Read existing, append, write back
            existing = {}
            if outputs_path.exists():
                try:
                    existing = _json.loads(outputs_path.read_text(encoding="utf-8"))
                except Exception:
                    existing = {}
            existing[agent_key] = output_text
            outputs_path.write_text(_json.dumps(existing, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    if progress_path:
        try:
            progress_path.write_text("0", encoding="utf-8")
        except Exception:
            pass
        # Clear previous outputs
        if outputs_path and outputs_path.exists():
            try:
                outputs_path.unlink()
            except Exception:
                pass
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.types.task_events import TaskStartedEvent, TaskCompletedEvent
        crewai_event_bus.register_handler(TaskStartedEvent, _on_task_started)
        crewai_event_bus.register_handler(TaskCompletedEvent, _on_task_completed)
    try:
        crew_instance = TradingAgentCrew(api_key_override=api_key).crew()
        result = crew_instance.kickoff(inputs=inputs)
    finally:
        if progress_path:
            try:
                progress_path.write_text("3", encoding="utf-8")  # all done
            except Exception:
                pass

    # CrewAI result may have .tasks_output (list of task results with .raw) or .raw (final only)
    tasks_outputs = []
    if hasattr(result, "tasks_output") and result.tasks_output:
        for t in result.tasks_output:
            tasks_outputs.append(getattr(t, "raw", str(t)))
    elif hasattr(result, "raw"):
        tasks_outputs = [result.raw]

    research_output = tasks_outputs[0] if len(tasks_outputs) > 0 else ""
    quant_output = tasks_outputs[1] if len(tasks_outputs) > 1 else ""
    decision_output = tasks_outputs[2] if len(tasks_outputs) > 2 else (result.raw if hasattr(result, "raw") else "")

    # Parse final recommendation from decision_output
    import re as _re
    rec = "HOLD"
    # Strip markdown bold/italic markers before parsing
    _dec_clean = _re.sub(r"\*+", "", (decision_output or "")).upper()
    # Find ALL "Recommendation: BUY/SELL/HOLD" patterns and take the LAST one
    _all_rec = _re.findall(
        r"(?:FINAL\s+)?RECOMMENDATION\s*[:\-]\s*(BUY|SELL|HOLD)", _dec_clean
    )
    if _all_rec:
        rec = _all_rec[-1]
    else:
        # Fallback: last standalone BUY/SELL/HOLD word in the text
        _standalone = _re.findall(r"\b(BUY|SELL|HOLD)\b", _dec_clean)
        if _standalone:
            rec = _standalone[-1]

    return {
        "recommendation": rec,
        "justification": decision_output or "",
        "research_output": research_output,
        "quant_output": quant_output,
        "decision_output": decision_output or "",
        "ticker": inputs["ticker"],
    }
