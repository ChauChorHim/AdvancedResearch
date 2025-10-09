#!/usr/bin/env python3
"""
Unit tests for main.py - Advanced Research System
Tests all functions and methods individually without using pytest or unittest
"""

import os
import tempfile
from unittest.mock import Mock, patch
import sys
from loguru import logger


from advanced_research.main import (
    generate_id,
    create_json_file,
    summarization_agent,
    run_agent,
    execute_worker_search_agents,
    create_director_agent,
    AdvancedResearch,
    AdvancedResearchAdditionalConfig,
    schema,
    model_name,
    director_model_name,
    max_tokens,
    exa_search_num_results,
    exa_search_max_characters,
)


def test_generate_id():
    """Test the generate_id function"""
    logger.info("Testing generate_id...")

    # Test that it generates a unique ID
    id1 = generate_id()
    id2 = generate_id()

    assert id1 != id2, "Generated IDs should be unique"
    assert isinstance(id1, str), "Generated ID should be a string"
    assert id1.startswith(
        "AdvancedResearch-"
    ), "ID should start with 'AdvancedResearch-'"
    assert "-time-" in id1, "ID should contain '-time-'"

    # Test that it contains a timestamp
    timestamp_part = id1.split("-time-")[1]
    assert (
        len(timestamp_part) == 14
    ), "Timestamp should be 14 characters (YYYYMMDDHHMMSS)"

    logger.success("✓ generate_id test passed")


def test_create_json_file():
    """Test the create_json_file function"""
    logger.info("Testing create_json_file...")

    # Test creating a new file
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".json"
    ) as temp_file:
        temp_path = temp_file.name

    try:
        test_data = {"test": "data", "number": 42}
        create_json_file(test_data, temp_path)

        # Verify file was created and contains correct data
        with open(temp_path, "rb") as f:
            import orjson

            loaded_data = orjson.loads(f.read())

        assert (
            loaded_data == test_data
        ), "File should contain the correct data"

        # Test updating existing file
        update_data = {"additional": "info", "test": "updated"}
        create_json_file(update_data, temp_path)

        with open(temp_path, "rb") as f:
            updated_data = orjson.loads(f.read())

        expected_data = {
            "test": "updated",
            "number": 42,
            "additional": "info",
        }
        assert (
            updated_data == expected_data
        ), "File should be updated correctly"

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    logger.success("✓ create_json_file test passed")


@patch("advanced_research.main.Agent")
def test_summarization_agent(mock_agent_class):
    """Test the summarization_agent function"""
    logger.info("Testing summarization_agent...")

    # Mock the agent instance
    mock_agent_instance = Mock()
    mock_agent_instance.run.return_value = "Test summary"
    mock_agent_class.return_value = mock_agent_instance

    # Test basic functionality
    result = summarization_agent(
        model_name="test-model", task="Test task", max_tokens=1000
    )

    # Verify agent was created with correct parameters
    mock_agent_class.assert_called_once()
    call_args = mock_agent_class.call_args
    assert call_args[1]["agent_name"] == "Report-Generator-Agent"
    assert call_args[1]["model_name"] == "test-model"
    assert call_args[1]["max_loops"] == 1
    assert call_args[1]["max_tokens"] == 1000

    # Verify agent was run
    mock_agent_instance.run.assert_called_once_with(
        task="Test task", img=None
    )

    # Verify return value
    assert result == "Test summary"

    logger.success("✓ summarization_agent test passed")


@patch("advanced_research.main.Agent")
def test_run_agent(mock_agent_class):
    """Test the run_agent function"""
    logger.info("Testing run_agent...")

    # Mock the agent instance
    mock_agent_instance = Mock()
    mock_agent_instance.run.return_value = "Agent output"
    mock_agent_class.return_value = mock_agent_instance

    # Test basic functionality
    result = run_agent(1, "Test query")

    # Verify agent was created with correct parameters
    mock_agent_class.assert_called_once()
    call_args = mock_agent_class.call_args
    assert call_args[1]["agent_name"] == "Worker-Search-Agent-1"
    assert call_args[1]["model_name"] == schema.worker_model_name
    assert call_args[1]["max_loops"] == 1
    assert call_args[1]["max_tokens"] == schema.worker_max_tokens
    assert call_args[1]["tool_call_summary"]

    # Verify agent was run
    mock_agent_instance.run.assert_called_once_with(task="Test query")

    # Verify return value
    assert result == "Agent output"

    logger.success("✓ run_agent test passed")


@patch("advanced_research.main.run_agent")
def test_execute_worker_search_agents(mock_run_agent):
    """Test the execute_worker_search_agents function"""
    logger.info("Testing execute_worker_search_agents...")

    # Mock the run_agent function to return different outputs
    mock_run_agent.side_effect = ["Output 1", "Output 2", "Output 3"]

    queries = ["Query 1", "Query 2", "Query 3"]
    result = execute_worker_search_agents(queries)

    # Verify run_agent was called for each query
    assert mock_run_agent.call_count == 3

    # Verify the calls were made with correct parameters
    expected_calls = [(0, "Query 1"), (1, "Query 2"), (2, "Query 3")]
    actual_calls = [call[0] for call in mock_run_agent.call_args_list]
    assert actual_calls == expected_calls

    # Verify result is concatenated outputs
    assert result == "Output 1 Output 2 Output 3"

    logger.success("✓ execute_worker_search_agents test passed")


@patch("advanced_research.main.Agent")
def test_create_director_agent(mock_agent_class):
    """Test the create_director_agent function"""
    logger.info("Testing create_director_agent...")

    # Mock the agent instance
    mock_agent_instance = Mock()
    mock_agent_instance.run.return_value = "Director output"
    mock_agent_class.return_value = mock_agent_instance

    # Test basic functionality
    result = create_director_agent(
        agent_name="Test-Director",
        model_name="test-model",
        task="Test task",
        max_tokens=4000,
        max_loops=2,
    )

    # Verify agent was created with correct parameters
    mock_agent_class.assert_called_once()
    call_args = mock_agent_class.call_args
    assert call_args[1]["agent_name"] == "Test-Director"
    assert call_args[1]["model_name"] == "test-model"
    assert call_args[1]["max_loops"] == 2
    assert call_args[1]["max_tokens"] == 4000
    assert call_args[1]["tool_call_summary"]

    # Verify agent was run
    mock_agent_instance.run.assert_called_once_with(
        task="Test task", img=None
    )

    # Verify return value
    assert result == "Director output"

    logger.success("✓ create_director_agent test passed")


def test_advanced_research_initialization():
    """Test AdvancedResearch class initialization"""
    logger.info("Testing AdvancedResearch initialization...")

    # Test with default parameters
    research = AdvancedResearch()

    assert research.name == "Advanced Research"
    assert research.description == "Advanced Research"
    assert research.worker_model_name == model_name
    assert research.director_agent_name == "Director-Agent"
    assert research.director_model_name == director_model_name
    assert research.director_max_tokens == 8000
    assert research.output_type == "final"
    assert research.max_loops == 1
    assert not research.export_on
    assert research.director_max_loops == 1
    assert research.id.startswith("AdvancedResearch-")
    assert research.conversation is not None

    # Test with custom parameters
    custom_research = AdvancedResearch(
        id="custom-id",
        name="Custom Research",
        description="Custom Description",
        worker_model_name="custom-worker",
        director_agent_name="Custom-Director",
        director_model_name="custom-director",
        director_max_tokens=4000,
        output_type="all",
        max_loops=3,
        export_on=True,
        director_max_loops=2,
    )

    assert custom_research.id == "custom-id"
    assert custom_research.name == "Custom Research"
    assert custom_research.description == "Custom Description"
    assert custom_research.worker_model_name == "custom-worker"
    assert custom_research.director_agent_name == "Custom-Director"
    assert custom_research.director_model_name == "custom-director"
    assert custom_research.director_max_tokens == 4000
    assert custom_research.output_type == "all"
    assert custom_research.max_loops == 3
    assert custom_research.export_on
    assert custom_research.director_max_loops == 2

    logger.success("✓ AdvancedResearch initialization test passed")


@patch("advanced_research.main.create_director_agent")
def test_advanced_research_step(mock_create_director):
    """Test AdvancedResearch.step method"""
    logger.info("Testing AdvancedResearch.step...")

    mock_create_director.return_value = "Step output"

    research = AdvancedResearch()
    result = research.step("Test task")

    # Verify create_director_agent was called with correct parameters
    mock_create_director.assert_called_once_with(
        agent_name=research.director_agent_name,
        model_name=research.director_model_name,
        task="Test task",
        max_tokens=research.director_max_tokens,
        img=None,
    )

    # Verify result
    assert result == "Step output"

    # Verify conversation was updated
    assert len(research.conversation.get_history()) == 1
    last_message = research.conversation.get_final_message()
    assert last_message == "Step output"

    logger.success("✓ AdvancedResearch.step test passed")


@patch("advanced_research.main.create_director_agent")
def test_advanced_research_run(mock_create_director):
    """Test AdvancedResearch.run method"""
    logger.info("Testing AdvancedResearch.run...")

    mock_create_director.return_value = "Research output"

    research = AdvancedResearch(max_loops=2)
    result = research.run("Test research task")

    # Verify create_director_agent was called twice (for 2 loops)
    assert mock_create_director.call_count == 2

    # Verify conversation was updated
    assert (
        len(research.conversation.get_history()) == 3
    )  # human + 2 agent responses

    # Verify result is formatted
    assert result is not None

    # Test with None task (should raise ValueError)
    research2 = AdvancedResearch()
    try:
        research2.run(None)
        assert False, "Should have raised ValueError for None task"
    except ValueError as e:
        assert "task argument is required" in str(e)

    logger.success("✓ AdvancedResearch.run test passed")


@patch("advanced_research.main.create_director_agent")
def test_advanced_research_batched_run(mock_create_director):
    """Test AdvancedResearch.batched_run method"""
    logger.info("Testing AdvancedResearch.batched_run...")

    mock_create_director.return_value = "Batch output"

    research = AdvancedResearch()
    tasks = ["Task 1", "Task 2", "Task 3"]

    # This should not raise an error
    research.batched_run(tasks)

    # Verify create_director_agent was called for each task
    assert mock_create_director.call_count == 3

    logger.success("✓ AdvancedResearch.batched_run test passed")


@patch("advanced_research.main.create_json_file")
@patch("advanced_research.main.os.makedirs")
def test_advanced_research_export_conversation(
    mock_makedirs, mock_create_json_file
):
    """Test AdvancedResearch._export_conversation method"""
    logger.info("Testing AdvancedResearch._export_conversation...")

    # Test with export_on = True
    research = AdvancedResearch(export_on=True)
    research.conversation.add("test", "test message")

    research._export_conversation()

    # Verify makedirs was called
    mock_makedirs.assert_called_once()

    # Verify create_json_file was called
    mock_create_json_file.assert_called_once()
    call_args = mock_create_json_file.call_args[0]
    assert call_args[1].endswith(f"{research.id}.json")

    # Test with export_on = False
    research2 = AdvancedResearch(export_on=False)
    research2._export_conversation()

    # Should not call create_json_file
    assert (
        mock_create_json_file.call_count == 1
    )  # Only from first call

    logger.success(
        "✓ AdvancedResearch._export_conversation test passed"
    )


def test_advanced_research_get_output_methods():
    """Test AdvancedResearch.get_output_methods method"""
    logger.info("Testing AdvancedResearch.get_output_methods...")

    research = AdvancedResearch()
    output_methods = research.get_output_methods()

    # Verify it returns a list
    assert isinstance(output_methods, list)

    # Verify it contains expected output types
    expected_types = ["final", "all", "last"]
    for output_type in expected_types:
        assert output_type in output_methods

    logger.success(
        "✓ AdvancedResearch.get_output_methods test passed"
    )


def test_advanced_research_additional_config():
    """Test AdvancedResearchAdditionalConfig schema"""
    logger.info("Testing AdvancedResearchAdditionalConfig...")

    # Test default values
    config = AdvancedResearchAdditionalConfig()
    assert config.worker_model_name == model_name
    assert config.worker_max_tokens == max_tokens
    assert config.exa_search_num_results == exa_search_num_results
    assert (
        config.exa_search_max_characters == exa_search_max_characters
    )

    # Test custom values
    custom_config = AdvancedResearchAdditionalConfig(
        worker_model_name="custom-model",
        worker_max_tokens=4000,
        exa_search_num_results=5,
        exa_search_max_characters=200,
    )

    assert custom_config.worker_model_name == "custom-model"
    assert custom_config.worker_max_tokens == 4000
    assert custom_config.exa_search_num_results == 5
    assert custom_config.exa_search_max_characters == 200

    logger.success("✓ AdvancedResearchAdditionalConfig test passed")


def run_all_tests():
    """Run all unit tests"""
    logger.info("=" * 60)
    logger.info("Running Advanced Research System Unit Tests")
    logger.info("=" * 60)

    test_functions = [
        test_generate_id,
        test_create_json_file,
        test_summarization_agent,
        test_run_agent,
        test_execute_worker_search_agents,
        test_create_director_agent,
        test_advanced_research_initialization,
        test_advanced_research_step,
        test_advanced_research_run,
        test_advanced_research_batched_run,
        test_advanced_research_export_conversation,
        test_advanced_research_get_output_methods,
        test_advanced_research_additional_config,
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            logger.error(f"✗ {test_func.__name__} failed: {str(e)}")
            failed += 1

    logger.info("=" * 60)
    logger.info(f"Test Results: {passed} passed, {failed} failed")
    logger.info("=" * 60)

    if failed == 0:
        logger.success("🎉 All tests passed!")
    else:
        logger.error(f"❌ {failed} tests failed!")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
