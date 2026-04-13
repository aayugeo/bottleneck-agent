Bottleneck Detection Agent
A Python agent that analyzes a stream of status updates and standup notes over time to surface where work is getting stuck, who is overloaded, and which blockers are chronic vs one-off.
Builds a persistent state across sessions: blocker frequency map, owner load tracking, and stalled task registry. Each new update is analyzed against full history so the agent distinguishes repeating patterns from isolated delays. Outputs prioritized intervention reports with urgency tiers and a trend signal comparing the current update against accumulated history.
Setup:
pip install anthropic
export ANTHROPIC_API_KEY=your_key_here
python bottleneck_agent.py
