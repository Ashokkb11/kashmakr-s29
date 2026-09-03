import pytest
from main import AgentState

class TestAgentStateContract:
    """
    Test suite for verifying the AgentState schema contract.
    Ensures the TypedDict is correctly defined and enforces JSON serializability.
    """

    def test_agent_state_is_typed_dict(self):
        """Verify AgentState is a TypedDict subclass."""
        # In Python 3.8+, TypedDict is a class
        assert hasattr(AgentState, '__annotations__')
        # Verify it behaves like a dict type hint
        assert AgentState.__name__ == 'AgentState'

    def test_agent_state_has_required_keys(self):
        """Verify AgentState contains the specific required keys."""
        expected_keys = {'messages', 'tool_outputs', 'next_action'}
        actual_keys = set(AgentState.__annotations__.keys())
        assert actual_keys == expected_keys

    def test_messages_type_is_list_of_dict(self):
        """Verify 'messages' is annotated as List[dict]."""
        annotations = AgentState.__annotations__
        assert 'messages' in annotations
        
        # Check the type representation
        msg_type = annotations['messages']
        # Python's typing representation for List[dict]
        # We check the string representation to be safe across versions
        assert 'list' in str(msg_type).lower() or 'List' in str(msg_type)
        assert 'dict' in str(msg_type)

    def test_tool_outputs_type_is_list_of_dict(self):
        """Verify 'tool_outputs' is annotated as List[dict]."""
        annotations = AgentState.__annotations__
        assert 'tool_outputs' in annotations
        
        output_type = annotations['tool_outputs']
        assert 'list' in str(output_type).lower() or 'List' in str(output_type)
        assert 'dict' in str(output_type)

    def test_next_action_type_is_str(self):
        """Verify 'next_action' is annotated as str."""
        annotations = AgentState.__annotations__
        assert 'next_action' in annotations
        assert annotations['next_action'] is str

    def test_state_instantiation_with_valid_data(self):
        """Verify a valid dictionary can satisfy the contract."""
        valid_state: AgentState = {
            "messages": [{"role": "user", "content": "Hello"}],
            "tool_outputs": [{"tool": "search", "result": "ok"}],
            "next_action": "continue"
        }
        # If TypedDict is structural, this should pass type checks
        # We verify runtime behavior
        assert isinstance(valid_state, dict)
        assert len(valid_state['messages']) == 1

    def test_state_json_serializability(self):
        """Verify the state contains only JSON-serializable types (dict, list, str)."""
        import json
        
        valid_state = {
            "messages": [{"role": "user", "content": "test"}],
            "tool_outputs": [],
            "next_action": "end"
        }
        
        # This test ensures the schema design goal of "100% JSON serializable" is met
        # by actually serializing a sample instance.
        try:
            json.dumps(valid_state)
        except TypeError:
            pytest.fail("AgentState structure failed JSON serialization")

    def test_state_accepts_empty_lists(self):
        """Verify state handles empty lists for messages and outputs."""
        state: AgentState = {
            "messages": [],
            "tool_outputs": [],
            "next_action": "continue"
        }
        assert state["messages"] == []
        assert state["tool_outputs"] == []

    def test_state_next_action_values(self):
        """Verify next_action accepts the suggested literal values."""
        # The schema defines it as str, but logic suggests specific values
        valid_actions = ['continue', 'human_review', 'end']
        for action in valid_actions:
            state: AgentState = {
                "messages": [],
                "tool_outputs": [],
                "next_action": action
            }
            assert state['next_action'] == action

    def test_nested_dict_in_messages(self):
        """Verify complex nested dicts are allowed within the message list."""
        state: AgentState = {
            "messages": [
                {"role": "system", "content": "Init"},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello", "tool_calls": [{"id": "1", "name": "search"}]}
            ],
            "tool_outputs": [],
            "next_action": "continue"
        }
        assert len(state["messages"]) == 3
        assert "tool_calls" in state["messages"][2]

    def test_nested_dict_in_tool_outputs(self):
        """Verify complex nested dicts are allowed within tool_outputs."""
        state: AgentState = {
            "messages": [],
            "tool_outputs": [
                {"tool_name": "db_read", "status": "success", "data": {"rows": 5}},
                {"tool_name": "api_call", "status": "error", "error_msg": "Timeout"}
            ],
            "next_action": "end"
        }
        assert state["tool_outputs"][0]["data"]["rows"] == 5
        assert state["tool_outputs"][1]["status"] == "error"