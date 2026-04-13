"""
Bottleneck Detection Agent
---------------------------
Feed this agent a stream of status updates, standup notes, or project
check-ins over time. It builds a persistent picture of where work is
getting stuck, who is overloaded, and what patterns are causing delays.

Each session:
  1. Reads new status input
  2. Extracts blockers, overloaded owners, and stalled tasks
  3. Compares against accumulated history to detect repeating patterns
  4. Generates a prioritized intervention report

Over time the agent compounds: it gets better at spotting chronic
bottlenecks vs one-off delays, and surfaces trends a human reviewer
would miss across weeks of updates.

Demonstrates: persistent memory with pattern detection, multi-session
compounding intelligence, structured analysis output, and feedback
synthesis across a long-running operational workflow.
"""

import json
from datetime import datetime
from pathlib import Path
import anthropic

MODEL = "claude-opus-4-5"
STATE_FILE = Path("bottleneck_state.json")
CLIENT = anthropic.Anthropic()


# ── State management ──────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "sessions": [],
        "blocker_frequency": {},
        "owner_load": {},
        "stalled_tasks": {}
    }


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def build_history_context(state: dict) -> str:
    """Summarize accumulated state to inject as memory into the agent."""
    if not state["sessions"]:
        return "No prior sessions. This is the first update."

    lines = [f"Total sessions analyzed: {len(state['sessions'])}"]

    if state["blocker_frequency"]:
        chronic = {k: v for k, v in state["blocker_frequency"].items() if v >= 2}
        if chronic:
            top = sorted(chronic.items(), key=lambda x: -x[1])[:4]
            lines.append("Chronic blockers (appeared 2+ times): " +
                         ", ".join(f"{b} ({n}x)" for b, n in top))

    if state["owner_load"]:
        overloaded = sorted(state["owner_load"].items(), key=lambda x: -x[1])[:3]
        lines.append("Most frequently blocked owners: " +
                     ", ".join(f"{o} ({n} times)" for o, n in overloaded))

    if state["stalled_tasks"]:
        stalled = [(t, d) for t, d in state["stalled_tasks"].items() if d >= 2]
        if stalled:
            lines.append("Persistently stalled tasks: " +
                         ", ".join(f"{t} ({d} sessions)" for t, d in stalled[:3]))

    recent = state["sessions"][-2:]
    for s in recent:
        lines.append(f"[{s['timestamp']}] {s['blocker_count']} blockers, "
                     f"{s['overloaded_count']} overloaded owners")

    return "\n".join(lines)


# ── Agent ─────────────────────────────────────────────────────────────────────

def analyze_update(text: str, history: str) -> dict:
    """Extract blockers, overloaded owners, and stalled tasks from status input."""
    system = """You are a bottleneck detection agent for operational workflows.
Analyze status updates to identify where work is getting stuck.
Return ONLY valid JSON with no preamble or markdown fences:
{
  "snapshot_summary": "1-2 sentence overview of the current state of work",
  "blockers": [
    {
      "description": "what is blocked",
      "owner": "who owns the blocked item",
      "cause": "what is causing the block",
      "severity": "critical | high | medium"
    }
  ],
  "overloaded_owners": [
    {
      "name": "person's name",
      "reason": "why they appear overloaded"
    }
  ],
  "stalled_tasks": [
    {
      "task": "task name or description",
      "stall_reason": "why it is not moving"
    }
  ],
  "interventions": [
    {
      "action": "specific thing to do",
      "target": "who or what it addresses",
      "urgency": "immediate | this_week | monitor"
    }
  ],
  "trend_signal": "given the history provided, what pattern is this update reinforcing or breaking?"
}"""

    user_prompt = f"""Accumulated history:
{history}

New status update to analyze:
{text}"""

    resp = CLIENT.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=system,
        messages=[{"role": "user", "content": user_prompt}]
    )
    raw = resp.content[0].text.strip().strip("```json").strip("```").strip()
    return json.loads(raw)


def update_state(state: dict, analysis: dict):
    """Merge analysis results into accumulated state."""
    for blocker in analysis.get("blockers", []):
        cause = blocker.get("cause", "unknown")
        state["blocker_frequency"][cause] = state["blocker_frequency"].get(cause, 0) + 1
        owner = blocker.get("owner", "unknown")
        if owner != "unknown":
            state["owner_load"][owner] = state["owner_load"].get(owner, 0) + 1

    for overloaded in analysis.get("overloaded_owners", []):
        name = overloaded.get("name", "unknown")
        state["owner_load"][name] = state["owner_load"].get(name, 0) + 1

    for stalled in analysis.get("stalled_tasks", []):
        task = stalled.get("task", "unknown")
        state["stalled_tasks"][task] = state["stalled_tasks"].get(task, 0) + 1

    state["sessions"].append({
        "timestamp": datetime.now().isoformat(timespec="minutes"),
        "blocker_count": len(analysis.get("blockers", [])),
        "overloaded_count": len(analysis.get("overloaded_owners", [])),
        "snapshot": analysis.get("snapshot_summary", "")
    })


# ── Display ───────────────────────────────────────────────────────────────────

def display(analysis: dict, session_num: int):
    sev_icon = {"critical": "X", "high": "!", "medium": "-"}

    print("\n" + "=" * 60)
    print(f"  BOTTLENECK REPORT - Session {session_num}")
    print("=" * 60)
    print(f"\n  {analysis.get('snapshot_summary', '')}")

    blockers = analysis.get("blockers", [])
    if blockers:
        print(f"\n  BLOCKERS ({len(blockers)}):")
        for b in sorted(blockers, key=lambda x: ["critical","high","medium"].index(x.get("severity","medium"))):
            icon = sev_icon.get(b["severity"], "-")
            print(f"  [{icon}] {b['description']}")
            print(f"       Owner : {b['owner']}")
            print(f"       Cause : {b['cause']}")

    overloaded = analysis.get("overloaded_owners", [])
    if overloaded:
        print(f"\n  OVERLOADED OWNERS:")
        for o in overloaded:
            print(f"  * {o['name']}: {o['reason']}")

    stalled = analysis.get("stalled_tasks", [])
    if stalled:
        print(f"\n  STALLED TASKS:")
        for s in stalled:
            print(f"  > {s['task']}")
            print(f"    {s['stall_reason']}")

    interventions = analysis.get("interventions", [])
    if interventions:
        urgency_order = ["immediate", "this_week", "monitor"]
        print(f"\n  INTERVENTIONS:")
        for i in sorted(interventions, key=lambda x: urgency_order.index(x.get("urgency", "monitor"))):
            print(f"  [{i['urgency'].upper()}] {i['action']}")
            print(f"    Target: {i['target']}")

    trend = analysis.get("trend_signal", "")
    if trend:
        print(f"\n  TREND: {trend}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "=" * 60)
    print("  BOTTLENECK DETECTION AGENT")
    print("  Paste status update/standup notes, type END when done")
    print("  Compounds intelligence across sessions automatically")
    print("  Type 'summary' to see accumulated state | 'quit' to exit")
    print("=" * 60 + "\n")

    state = load_state()
    print(f"  Memory loaded: {len(state['sessions'])} prior session(s)\n")

    while True:
        print("-" * 60)
        print("Status update:")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            cmd = line.strip().lower()
            if cmd == "quit":
                save_state(state)
                print("State saved. Exiting.\n")
                return
            if cmd == "summary":
                print("\n" + build_history_context(state) + "\n")
                lines = []
                break
            if cmd == "end":
                break
            lines.append(line)

        if not lines:
            continue

        text = "\n".join(lines).strip()
        if not text:
            continue

        print("\n  Analyzing...", flush=True)
        try:
            history = build_history_context(state)
            analysis = analyze_update(text, history)
            update_state(state, analysis)
            save_state(state)
            display(analysis, len(state["sessions"]))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Processing error: {e}\n")


if __name__ == "__main__":
    run()
