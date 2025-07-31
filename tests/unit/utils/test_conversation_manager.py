# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.utils.conversation_manager import ConversationManager, truncate_content


class TestTruncateContent:
    """Test the truncate_content function."""
    
    def test_short_content_unchanged(self):
        """Test that short content is not truncated."""
        content = "This is a short message."
        result = truncate_content(content, max_length=100)
        assert result == content
    
    def test_long_content_truncated(self):
        """Test that long content is properly truncated."""
        content = "A" * 1000
        result = truncate_content(content, max_length=100)
        
        assert len(result) <= 150  # Should be around max_length + truncation notice
        assert "Content truncated" in result
        assert result.startswith("A" * 50)  # First half
        assert result.endswith("A" * 50)   # Second half
    
    def test_custom_max_length(self):
        """Test truncation with custom max length."""
        content = "B" * 200
        result = truncate_content(content, max_length=50)
        
        assert "Content truncated" in result
        assert result.startswith("B" * 25)
        assert result.endswith("B" * 25)


class TestConversationManager:
    """Test the ConversationManager class."""
    
    def test_initialization(self):
        """Test ConversationManager initialization."""
        manager = ConversationManager(max_messages=10, max_content_length=1000)
        
        assert manager.max_messages == 10
        assert manager.max_content_length == 1000
        assert len(manager.messages) == 0
        assert manager.summary is None
    
    def test_add_short_message(self):
        """Test adding a short message."""
        manager = ConversationManager()
        message = HumanMessage(content="Hello, world!")
        
        manager.add_message(message)
        
        assert len(manager.messages) == 1
        assert manager.messages[0].content == "Hello, world!"
    
    def test_add_long_message_truncated(self):
        """Test that long messages are truncated when added."""
        manager = ConversationManager(max_content_length=100)
        long_content = "X" * 200
        message = HumanMessage(content=long_content)
        
        manager.add_message(message)
        
        assert len(manager.messages) == 1
        assert "Content truncated" in manager.messages[0].content
        assert len(manager.messages[0].content) < 200
    
    def test_automatic_summarization(self):
        """Test that messages are automatically summarized when limit is reached."""
        manager = ConversationManager(max_messages=3, preserve_recent=2)
        
        # Add messages that exceed the limit
        for i in range(5):
            message = HumanMessage(content=f"Message {i}")
            manager.add_message(message)
        
        # Should have only recent messages + summary
        assert len(manager.messages) == 2  # preserve_recent
        assert manager.summary is not None
        assert "Message 3" in manager.messages[0].content  # Recent messages preserved
        assert "Message 4" in manager.messages[1].content
    
    def test_get_messages_normal_mode(self):
        """Test getting messages in normal mode."""
        manager = ConversationManager()
        message1 = HumanMessage(content="First message")
        message2 = AIMessage(content="Second message")
        
        manager.add_message(message1)
        manager.add_message(message2)
        
        messages = manager.get_messages(summary_mode=False)
        
        assert len(messages) == 2
        assert messages[0].content == "First message"
        assert messages[1].content == "Second message"
    
    def test_get_messages_summary_mode(self):
        """Test getting messages in summary mode."""
        manager = ConversationManager(max_messages=2, preserve_recent=1)
        
        # Add messages to trigger summarization
        for i in range(3):
            message = HumanMessage(content=f"Message {i}")
            manager.add_message(message)
        
        messages = manager.get_messages(summary_mode=True)
        
        # Should have summary message + recent messages
        assert len(messages) >= 2
        assert any("Conversation Summary" in msg.content for msg in messages if hasattr(msg, 'content'))
    
    def test_get_recent_messages(self):
        """Test getting recent messages."""
        manager = ConversationManager()
        
        for i in range(5):
            message = HumanMessage(content=f"Message {i}")
            manager.add_message(message)
        
        recent = manager.get_recent_messages(3)
        
        assert len(recent) == 3
        assert recent[0].content == "Message 2"
        assert recent[2].content == "Message 4"
    
    def test_clear_conversation(self):
        """Test clearing conversation history."""
        manager = ConversationManager()
        
        manager.add_message(HumanMessage(content="Test message"))
        manager.summary = "Test summary"
        
        manager.clear()
        
        assert len(manager.messages) == 0
        assert manager.summary is None
    
    def test_conversation_stats(self):
        """Test getting conversation statistics."""
        manager = ConversationManager(max_messages=5, preserve_recent=2)
        
        manager.add_message(HumanMessage(content="Hello"))
        manager.add_message(AIMessage(content="Hi there"))
        
        stats = manager.get_conversation_stats()
        
        assert stats["total_messages"] == 2
        assert stats["has_summary"] is False
        assert stats["max_messages_threshold"] == 5
        assert stats["preserve_recent_count"] == 2
        assert stats["total_content_length"] > 0
    
    def test_create_summary_for_content(self):
        """Test the content summarization method."""
        manager = ConversationManager()
        
        # Test short content (should remain unchanged)
        short_content = "Short content"
        result = manager._create_summary_for_content(short_content)
        assert result == short_content
        
        # Test long content (should be summarized)
        long_content = "A" * 2000
        result = manager._create_summary_for_content(long_content)
        
        assert len(result) < len(long_content)
        assert "Content summarized" in result
        assert result.startswith("A" * 400)  # Beginning preserved
        assert result.endswith("A" * 400)    # End preserved


class TestConversationManagerIntegration:
    """Integration tests for ConversationManager."""
    
    def test_mixed_message_types(self):
        """Test handling different message types."""
        manager = ConversationManager(max_messages=4, preserve_recent=2)
        
        # Add various message types
        manager.add_message(SystemMessage(content="System initialization"))
        manager.add_message(HumanMessage(content="User question"))
        manager.add_message(AIMessage(content="AI response", name="assistant"))
        manager.add_message(HumanMessage(content="Follow-up question"))
        manager.add_message(AIMessage(content="Final response", name="researcher"))
        
        # Should trigger summarization
        assert len(manager.messages) == 2  # preserve_recent
        assert manager.summary is not None
        
        # Get messages in summary mode
        messages = manager.get_messages(summary_mode=True)
        assert len(messages) >= 3  # summary + recent messages
    
    def test_realistic_conversation_flow(self):
        """Test a realistic conversation flow with various content lengths."""
        manager = ConversationManager(max_messages=6, max_content_length=500, preserve_recent=3)
        
        # Simulate a research conversation
        manager.add_message(HumanMessage(content="Research topic: AI in healthcare"))
        manager.add_message(AIMessage(content="I'll help you research AI applications in healthcare. Let me start by gathering information.", name="researcher"))
        
        # Add a long research result
        long_research = "AI in healthcare has numerous applications: " + "detailed analysis " * 100
        manager.add_message(AIMessage(content=long_research, name="researcher"))
        
        # Add more conversation
        manager.add_message(HumanMessage(content="Can you focus on diagnostic applications?"))
        manager.add_message(AIMessage(content="Certainly! AI diagnostic applications include medical imaging analysis, symptom assessment, and predictive diagnostics.", name="researcher"))
        
        # Add more messages to trigger summarization
        for i in range(3):
            manager.add_message(HumanMessage(content=f"Additional question {i}"))
        
        # Verify the conversation is properly managed
        stats = manager.get_conversation_stats()
        assert stats["total_messages"] == 3  # preserve_recent
        assert stats["has_summary"] is True
        
        # Verify summary mode works
        summary_messages = manager.get_messages(summary_mode=True)
        assert len(summary_messages) >= 4  # summary + recent messages