from unittest.mock import MagicMock, patch

import pytest
import requests
from omegaconf import DictConfig

from habitat_llm.llm.rlm_lock import RemoteLanguageModel


# Mock configuration for the RemoteLanguageModel
@pytest.fixture
def mock_conf():
    return DictConfig(
        {"host": "localhost", "port": 8000, "generation_params": {"engine": "gpt-oss"}}
    )


# Test case for a single tagged prompt
def test_batch_chat_generate_single_tagged_prompt(mock_conf):
    model = RemoteLanguageModel(mock_conf, chat=True)
    prompts = [
        "<|start|>system<|message|>\nYou are a helpful assistant.<|end|><|start|>user<|message|>\nHello!<|end|>"
    ]

    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hi there!"}}]
        }
        mock_post.return_value = mock_response

        results = model.batch_chat_generate(prompts)

        assert len(results) == 1
        assert results[0]["generation"] == "Hi there!"

        mock_post.assert_called_once()
        call_args = mock_post.call_args[1]["json"]
        assert call_args["messages"] == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]


# Test case for a single untagged prompt
def test_batch_chat_generate_single_untagged_prompt(mock_conf):
    model = RemoteLanguageModel(mock_conf, chat=True)
    prompts = ["Just a simple prompt."]

    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response to simple prompt."}}]
        }
        mock_post.return_value = mock_response

        results = model.batch_chat_generate(prompts)

        assert len(results) == 1
        assert results[0]["generation"] == "Response to simple prompt."

        mock_post.assert_called_once()
        call_args = mock_post.call_args[1]["json"]
        assert call_args["messages"] == [
            {"role": "system", "content": "You are an expert at task planning."},
            {"role": "user", "content": "Just a simple prompt."},
        ]


# Test case for multiple prompts (tagged and untagged)
def test_batch_chat_generate_multiple_prompts(mock_conf):
    model = RemoteLanguageModel(mock_conf, chat=True)
    prompts = [
        "<|start|>user<|message|>\nFirst prompt<|end|>",
        "Second prompt, untagged.",
    ]

    mock_responses = [
        {"choices": [{"message": {"content": "Response to first."}}]},
        {"choices": [{"message": {"content": "Response to second."}}]},
    ]

    def side_effect(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.status_code = 200
        if "First prompt" in kwargs["json"]["messages"][-1]["content"]:
            mock_response.json.return_value = mock_responses[0]
        else:
            mock_response.json.return_value = mock_responses[1]
        return mock_response

    with patch("requests.post", side_effect=side_effect) as mock_post:
        results = model.batch_chat_generate(prompts)

        assert len(results) == 2
        assert results[0]["generation"] == "Response to first."
        assert results[1]["generation"] == "Response to second."
        assert mock_post.call_count == 2


# Test for malformed tagged prompt
def test_batch_chat_generate_malformed_prompt(mock_conf):
    model = RemoteLanguageModel(mock_conf, chat=True)
    prompts = [
        "<|start|>system<|message|>\nValid part<|end|><|invalid_tag|>\nInvalid part<|end|>"
    ]

    with pytest.raises(ValueError, match="Malformed tagged chat prompt"):
        model.batch_chat_generate(prompts)


# Test for empty prompt
def test_batch_chat_generate_empty_prompt(mock_conf):
    model = RemoteLanguageModel(mock_conf, chat=True)
    prompts = [""]

    with pytest.raises(ValueError, match="Prompt not formatted correctly for chat"):
        model.batch_chat_generate(prompts)


# Test for request exception
def test_batch_chat_generate_request_exception(mock_conf):
    model = RemoteLanguageModel(mock_conf, chat=True)
    prompts = ["A prompt that will fail."]

    with patch("requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.RequestException("Test error")

        results = model.batch_chat_generate(prompts)

        assert len(results) == 1
        assert "Error calling remote model" in results[0]["generation"]


# --- reasoning_effort tests ---


# Config with reasoning_effort set
@pytest.fixture
def mock_conf_with_reasoning():
    return DictConfig(
        {
            "host": "localhost",
            "port": 8000,
            "generation_params": {
                "engine": "gpt-oss",
                "chat": True,
                "reasoning_effort": "medium",
            },
        }
    )


# Config with reasoning_effort = null (None)
@pytest.fixture
def mock_conf_reasoning_null():
    return DictConfig(
        {
            "host": "localhost",
            "port": 8000,
            "generation_params": {
                "engine": "gpt-oss",
                "chat": True,
                "reasoning_effort": None,
            },
        }
    )


def test_reasoning_effort_included_in_payload(mock_conf_with_reasoning):
    model = RemoteLanguageModel(mock_conf_with_reasoning)
    assert model.reasoning_effort == "medium"
    prompts = ["Hello"]

    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hi"}}]}
        mock_post.return_value = mock_response

        model.batch_chat_generate(prompts)

        call_args = mock_post.call_args[1]["json"]
        assert "reasoning_effort" in call_args
        assert call_args["reasoning_effort"] == "medium"


def test_reasoning_effort_omitted_when_null(mock_conf_reasoning_null):
    model = RemoteLanguageModel(mock_conf_reasoning_null)
    assert model.reasoning_effort is None
    prompts = ["Hello"]

    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hi"}}]}
        mock_post.return_value = mock_response

        model.batch_chat_generate(prompts)

        call_args = mock_post.call_args[1]["json"]
        assert "reasoning_effort" not in call_args


def test_reasoning_effort_omitted_when_absent(mock_conf):
    """When reasoning_effort is not in config at all, it should be None."""
    model = RemoteLanguageModel(mock_conf, chat=True)
    assert model.reasoning_effort is None


def test_reasoning_effort_invalid_value_raises():
    conf = DictConfig(
        {
            "host": "localhost",
            "port": 8000,
            "generation_params": {"engine": "gpt-oss", "reasoning_effort": "extreme"},
        }
    )
    with pytest.raises(ValueError, match="reasoning_effort must be"):
        RemoteLanguageModel(conf)


def test_reasoning_effort_all_valid_values():
    for level in ("low", "medium", "high"):
        conf = DictConfig(
            {
                "host": "localhost",
                "port": 8000,
                "generation_params": {"engine": "gpt-oss", "reasoning_effort": level},
            }
        )
        model = RemoteLanguageModel(conf)
        assert model.reasoning_effort == level
