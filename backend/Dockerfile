# Example of the robust state schema I implement
from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    messages: List[dict]  # Chat history
    tool_outputs: List[dict] # Results from tools
    next_action: str # 'continue', 'human_review', 'end'
    # Note: No objects, no connections. 100% JSON serializable.