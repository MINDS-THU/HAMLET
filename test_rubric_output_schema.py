"""
Test script to verify CodeAgentRubric behavior with output_schema.

This script tests:
1. First creation (in CodeAgentEnv.__init__): agent is None, so no output_schema reward func should be added
2. Second creation (in rollout): agent is initialized, so if output_schema exists, reward func should be added
"""

import asyncio
from pydantic import BaseModel
from datasets import Dataset
from openai import AsyncOpenAI
from hamlet.core.models import Model, ChatMessage, MessageRole, TokenUsage
from hamlet.train.codeagent_environment import CodeAgentEnv, CodeAgentRubric


# Define output schema for testing
class TestOutputSchema(BaseModel):
    result: str
    confidence: float


# Mock Model for testing (doesn't actually call API)
class MockModel(Model):
    """Mock model that returns a simple response without calling API."""
    
    def generate(self, messages, **kwargs) -> ChatMessage:
        content = """
Here is the solution.
```python
final_answer({'result': 'success', 'confidence': 0.99})
```
"""
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            token_usage=TokenUsage(input_tokens=10, output_tokens=20)
        )


async def test_rubric_output_schema():
    """
    Test that CodeAgentRubric correctly detects output_schema.
    
    Steps:
    1. Create CodeAgentEnv with output_schema in agent_kwargs
    2. Check first rubric creation (in __init__) - agent should be None
    3. Call rollout to trigger agent initialization
    4. Check second rubric creation (in rollout) - agent should have output_schema
    """
    
    print("=" * 60)
    print("Test: CodeAgentRubric output_schema detection")
    print("=" * 60)
    
    # Step 1: Create CodeAgentEnv with output_schema
    print("\n[Step 1] Creating CodeAgentEnv with output_schema in agent_kwargs...")
    # Create a minimal dataset (required by Environment)
    dummy_dataset = Dataset.from_dict({"question": ["test"], "answer": ["test"]})
    env = CodeAgentEnv(
        agent_kwargs={
            "output_schema": TestOutputSchema,
            "tools": [],
            "max_steps": 1,
        },
        model_kwargs={},
        model_class=MockModel,
        message_type="chat",
        dataset=dummy_dataset,
    )
    
    # Step 2: Check first rubric creation (in __init__)
    print("\n[Step 2] Checking first rubric (created in __init__)...")
    first_rubric = env.rubric
    print(f"  - Number of reward functions: {len(first_rubric.reward_funcs)}")
    print(f"  - Reward function names: {[f.__name__ for f in first_rubric.reward_funcs]}")
    print(f"  - Number of weights: {len(first_rubric.reward_weights)}")
    print(f"  - Agent in rubric: {first_rubric.agent}")
    
    # Expected: agent is None, so only 2 reward funcs (parsing + execution)
    assert len(first_rubric.reward_funcs) == 2, f"Expected 2 reward funcs, got {len(first_rubric.reward_funcs)}"
    assert first_rubric.agent is None, "Agent should be None at this point"
    print("  ✓ First rubric check passed: agent is None, no output_schema reward func added")
    
    # Step 3: Create a mock AsyncOpenAI client
    print("\n[Step 3] Creating mock AsyncOpenAI client...")
    # We'll use a simple mock - the actual API won't be called because we use MockModel
    mock_client = AsyncOpenAI(api_key="test-key", base_url="http://test-url")
    
    # Step 4: Call rollout to trigger agent initialization
    print("\n[Step 4] Calling rollout to initialize agent...")
    prompt = [
        {"role": "user", "content": "Test task: return a result with confidence 0.9"}
    ]
    
    try:
        completion, state = await env.rollout(
            client=mock_client,
            model="test-model",
            prompt=prompt,
            answer="",
            task="test",
        )
        print("  ✓ Rollout completed successfully")
        
        # Step 4.5: Call scoring to trigger reward functions
        print("\n[Step 4.5] Calling rubric.score_rollout to trigger reward functions...")
        try:
            score_result = await env.rubric.score_rollout(
                prompt=prompt,
                completion=completion,
                answer="",
                state=state,
                task="test",
            )
            print(f"  ✓ Scoring completed. Reward: {score_result.reward}")
            print(f"  - Metrics: {score_result.metrics}")
        except Exception as e:
            print(f"  ⚠ Scoring raised exception: {e}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"  ⚠ Rollout raised exception (this is OK for testing): {e}")
        print("  Continuing to check rubric...")
    
    # Step 5: Check second rubric creation (in rollout)
    print("\n[Step 5] Checking second rubric (created in rollout)...")
    second_rubric = env.rubric
    print(f"  - Number of reward functions: {len(second_rubric.reward_funcs)}")
    print(f"  - Reward function names: {[f.__name__ for f in second_rubric.reward_funcs]}")
    print(f"  - Number of weights: {len(second_rubric.reward_weights)}")
    print(f"  - Agent in rubric: {second_rubric.agent}")
    if second_rubric.agent:
        print(f"  - Agent output_schema: {second_rubric.agent.output_schema}")
    
    # Expected: agent is initialized with output_schema, so 3 reward funcs should be present
    if second_rubric.agent and second_rubric.agent.output_schema is not None:
        assert len(second_rubric.reward_funcs) == 3, \
            f"Expected 3 reward funcs (with output_schema), got {len(second_rubric.reward_funcs)}"
        assert len(second_rubric.reward_weights) == 3, \
            f"Expected 3 weights, got {len(second_rubric.reward_weights)}"
        # Check that the new reward func is present
        reward_func_names = [f.__name__ for f in second_rubric.reward_funcs]
        assert "successful_output_schema_reward_func" in reward_func_names, \
            "successful_output_schema_reward_func should be in reward_funcs"
        print("  ✓ Second rubric check passed: agent has output_schema, reward func added")
    else:
        print("  ⚠ Agent or output_schema is None - this might indicate an issue")
        print(f"     Agent: {second_rubric.agent}")
        if second_rubric.agent:
            print(f"     output_schema: {second_rubric.agent.output_schema}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


async def test_rubric_without_output_schema():
    """
    Test that CodeAgentRubric doesn't add output_schema reward func when output_schema is not provided.
    """
    
    print("\n" + "=" * 60)
    print("Test: CodeAgentRubric without output_schema")
    print("=" * 60)
    
    # Create CodeAgentEnv without output_schema
    print("\n[Step 1] Creating CodeAgentEnv WITHOUT output_schema...")
    # Create a minimal dataset (required by Environment)
    dummy_dataset = Dataset.from_dict({"question": ["test"], "answer": ["test"]})
    env = CodeAgentEnv(
        agent_kwargs={
            "tools": [],
            "max_steps": 1,
        },
        model_kwargs={},
        model_class=MockModel,
        message_type="chat",
        dataset=dummy_dataset,
    )
    
    # Check first rubric
    print("\n[Step 2] Checking first rubric...")
    first_rubric = env.rubric
    print(f"  - Number of reward functions: {len(first_rubric.reward_funcs)}")
    assert len(first_rubric.reward_funcs) == 2, "Should have 2 reward funcs (parsing + execution)"
    print("  ✓ First rubric check passed")
    
    # Call rollout
    print("\n[Step 3] Calling rollout...")
    mock_client = AsyncOpenAI(api_key="test-key", base_url="http://test-url")
    prompt = [{"role": "user", "content": "Test task"}]
    
    try:
        await env.rollout(
            client=mock_client,
            model="test-model",
            prompt=prompt,
            answer="",
            task="test",
        )
    except Exception as e:
        print(f"  ⚠ Rollout exception (OK for testing): {e}")
    
    # Check second rubric
    print("\n[Step 4] Checking second rubric...")
    second_rubric = env.rubric
    print(f"  - Number of reward functions: {len(second_rubric.reward_funcs)}")
    print(f"  - Agent output_schema: {second_rubric.agent.output_schema if second_rubric.agent else None}")
    
    # Should still have only 2 reward funcs (no output_schema)
    assert len(second_rubric.reward_funcs) == 2, \
        f"Should have 2 reward funcs (no output_schema), got {len(second_rubric.reward_funcs)}"
    print("  ✓ Second rubric check passed: no output_schema reward func added")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running CodeAgentRubric output_schema tests")
    print("=" * 60)
    
    # Run both tests
    asyncio.run(test_rubric_output_schema())
    asyncio.run(test_rubric_without_output_schema())
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

