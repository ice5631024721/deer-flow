# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import pytest
from src.llms import llm


class DummyChatOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def invoke(self, msg):
        return f"Echo: {msg}"


@pytest.fixture(autouse=True)
def patch_chat_openai(monkeypatch):
    monkeypatch.setattr(llm, "ChatOpenAI", DummyChatOpenAI)


@pytest.fixture
def dummy_conf():
    return {
        "BASIC_MODEL": {"api_key": "test_key", "base_url": "http://test"},
        "REASONING_MODEL": {"api_key": "reason_key"},
        "VISION_MODEL": {"api_key": "vision_key"},
    }


def test_get_env_llm_conf(monkeypatch):
    # Clear any existing environment variables that might interfere
    monkeypatch.delenv("BASIC_MODEL__API_KEY", raising=False)
    monkeypatch.delenv("BASIC_MODEL__BASE_URL", raising=False)
    monkeypatch.delenv("BASIC_MODEL__MODEL", raising=False)

    monkeypatch.setenv("BASIC_MODEL__API_KEY", "env_key")
    monkeypatch.setenv("BASIC_MODEL__BASE_URL", "http://env")
    conf = llm._get_env_llm_conf("basic")
    assert conf["api_key"] == "env_key"
    assert conf["base_url"] == "http://env"


def test_create_llm_use_conf_merges_env(monkeypatch, dummy_conf):
    # Clear any existing environment variables that might interfere
    monkeypatch.delenv("BASIC_MODEL__BASE_URL", raising=False)
    monkeypatch.delenv("BASIC_MODEL__MODEL", raising=False)
    monkeypatch.setenv("BASIC_MODEL__API_KEY", "env_key")
    result = llm._create_llm_use_conf("basic", dummy_conf)
    assert isinstance(result, DummyChatOpenAI)
    assert result.kwargs["api_key"] == "env_key"
    assert result.kwargs["base_url"] == "http://test"


def test_create_llm_use_conf_invalid_type(monkeypatch, dummy_conf):
    # Clear any existing environment variables that might interfere
    monkeypatch.delenv("BASIC_MODEL__API_KEY", raising=False)
    monkeypatch.delenv("BASIC_MODEL__BASE_URL", raising=False)
    monkeypatch.delenv("BASIC_MODEL__MODEL", raising=False)

    with pytest.raises(ValueError):
        llm._create_llm_use_conf("unknown", dummy_conf)


def test_create_llm_use_conf_empty_conf(monkeypatch):
    # Clear any existing environment variables that might interfere
    monkeypatch.delenv("BASIC_MODEL__API_KEY", raising=False)
    monkeypatch.delenv("BASIC_MODEL__BASE_URL", raising=False)
    monkeypatch.delenv("BASIC_MODEL__MODEL", raising=False)

    with pytest.raises(ValueError):
        llm._create_llm_use_conf("basic", {})


def test_get_llm_by_type_caches(monkeypatch, dummy_conf):
    called = {}

    def fake_load_yaml_config(path):
        called["called"] = True
        return dummy_conf

    monkeypatch.setattr(llm, "load_yaml_config", fake_load_yaml_config)
    llm._llm_cache.clear()
    inst1 = llm.get_llm_by_type("basic")
    inst2 = llm.get_llm_by_type("basic")
    assert inst1 is inst2
    assert called["called"]


def test_create_llm_with_max_tokens(monkeypatch, dummy_conf):
    """Test that max_tokens parameter is properly handled."""
    # Clear any existing environment variables that might interfere
    monkeypatch.delenv("BASIC_MODEL__API_KEY", raising=False)
    monkeypatch.delenv("BASIC_MODEL__BASE_URL", raising=False)
    monkeypatch.delenv("BASIC_MODEL__MODEL", raising=False)
    monkeypatch.delenv("BASIC_MODEL__MAX_TOKENS", raising=False)
    
    # Test with max_tokens in config
    conf_with_max_tokens = {
        "BASIC_MODEL": {
            "api_key": "test_key", 
            "base_url": "http://test",
            "max_tokens": 4096
        }
    }
    result = llm._create_llm_use_conf("basic", conf_with_max_tokens)
    assert isinstance(result, DummyChatOpenAI)
    assert result.kwargs["max_tokens"] == 4096


def test_create_llm_with_max_tokens_from_env(monkeypatch, dummy_conf):
    """Test that max_tokens parameter from environment variables is properly handled."""
    # Clear any existing environment variables that might interfere
    monkeypatch.delenv("BASIC_MODEL__BASE_URL", raising=False)
    monkeypatch.delenv("BASIC_MODEL__MODEL", raising=False)
    
    # Set max_tokens via environment variable
    monkeypatch.setenv("BASIC_MODEL__API_KEY", "env_key")
    monkeypatch.setenv("BASIC_MODEL__MAX_TOKENS", "2048")
    
    result = llm._create_llm_use_conf("basic", dummy_conf)
    assert isinstance(result, DummyChatOpenAI)
    assert result.kwargs["max_tokens"] == 2048


def test_create_llm_with_invalid_max_tokens(monkeypatch, dummy_conf, capsys):
    """Test that invalid max_tokens values are handled gracefully."""
    # Clear any existing environment variables that might interfere
    monkeypatch.delenv("BASIC_MODEL__API_KEY", raising=False)
    monkeypatch.delenv("BASIC_MODEL__BASE_URL", raising=False)
    monkeypatch.delenv("BASIC_MODEL__MODEL", raising=False)
    monkeypatch.delenv("BASIC_MODEL__MAX_TOKENS", raising=False)
    
    # Test with invalid max_tokens in config
    conf_with_invalid_max_tokens = {
        "BASIC_MODEL": {
            "api_key": "test_key", 
            "base_url": "http://test",
            "max_tokens": "invalid_value"
        }
    }
    result = llm._create_llm_use_conf("basic", conf_with_invalid_max_tokens)
    assert isinstance(result, DummyChatOpenAI)
    # max_tokens should not be set when invalid
    assert "max_tokens" not in result.kwargs
    
    # Check that warning was printed
    captured = capsys.readouterr()
    assert "Warning: Invalid max_tokens value 'invalid_value' for basic, ignoring." in captured.out


def test_create_llm_without_max_tokens(monkeypatch, dummy_conf):
    """Test that LLM creation works normally without max_tokens parameter."""
    # Clear any existing environment variables that might interfere
    monkeypatch.delenv("BASIC_MODEL__API_KEY", raising=False)
    monkeypatch.delenv("BASIC_MODEL__BASE_URL", raising=False)
    monkeypatch.delenv("BASIC_MODEL__MODEL", raising=False)
    monkeypatch.delenv("BASIC_MODEL__MAX_TOKENS", raising=False)
    
    result = llm._create_llm_use_conf("basic", dummy_conf)
    assert isinstance(result, DummyChatOpenAI)
    # max_tokens should not be set when not provided
    assert "max_tokens" not in result.kwargs
