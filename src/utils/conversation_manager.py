# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Conversation History Manager

This module provides conversation history management capabilities inspired by DeepResearchAgent,
including content truncation and summary mode for managing long conversations.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# Inspired by DeepResearchAgent: https://github.com/SkyworkAI/DeepResearchAgent
# Default configuration constants
# Balanced limits to prevent OpenAI token limit errors while maintaining functionality
MAX_CONTENT_LENGTH = 900000  # Reasonable limit for individual message content
MAX_MESSAGES_BEFORE_SUMMARY = 30  # Allow more messages before summarization
SUMMARY_PRESERVE_RECENT = 5  # Keep more recent messages for better context


def truncate_content(content: str, max_length: int = MAX_CONTENT_LENGTH) -> str:
    """
    Truncate content to stay within specified length limits.
    
    Inspired by DeepResearchAgent's truncate_content function, this preserves
    both the beginning and end of the content when truncating.
    
    Args:
        content: The content to truncate
        max_length: Maximum allowed length
        
    Returns:
        Truncated content with indication of truncation
    """
    if len(content) <= max_length:
        return content
    
    # Calculate half length for beginning and end
    half_length = max_length // 2
    truncation_notice = "\n...Content truncated...\n"
    
    return (
        content[:half_length] + 
        truncation_notice + 
        content[-half_length:]
    )


class ConversationManager:
    """
    Manages conversation history with automatic summarization and truncation.
    
    This class provides functionality to:
    - Track conversation messages
    - Automatically summarize old messages when conversation gets too long
    - Truncate individual message content
    - Provide summary mode for memory-efficient processing
    """
    
    def __init__(
        self,
        max_messages: int = MAX_MESSAGES_BEFORE_SUMMARY,
        max_content_length: int = MAX_CONTENT_LENGTH,
        preserve_recent: int = SUMMARY_PRESERVE_RECENT
    ):
        self.max_messages = max_messages
        self.max_content_length = max_content_length
        self.preserve_recent = preserve_recent
        self.messages: List[BaseMessage] = []
        self.summary: Optional[str] = None
        
    def add_message(self, message: BaseMessage) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            message: The message to add
        """
        # Truncate message content if too long
        if hasattr(message, 'content') and isinstance(message.content, str):
            if len(message.content) > self.max_content_length:
                truncated_content = self._create_summary_for_content(message.content)
                # Create new message with truncated content
                if isinstance(message, AIMessage):
                    message = AIMessage(content=truncated_content, name=getattr(message, 'name', None))
                elif isinstance(message, HumanMessage):
                    message = HumanMessage(content=truncated_content, name=getattr(message, 'name', None))
                elif isinstance(message, SystemMessage):
                    message = SystemMessage(content=truncated_content)
        
        # Check if we need to summarize before adding the new message
        if len(self.messages) >= self.max_messages:
            self._create_summary()
        
        self.messages.append(message)
        
        # Check again after adding the message in case we still exceed the limit
        if len(self.messages) > self.preserve_recent:
            self._create_summary()
    
    def _create_summary(self) -> None:
        """
        Create a summary of older messages and keep only recent ones.
        
        This method summarizes messages beyond the preserve_recent threshold
        and removes them from the active message list.
        """
        if len(self.messages) <= self.preserve_recent:
            return
            
        # Messages to summarize (all except the most recent ones)
        messages_to_summarize = self.messages[:-self.preserve_recent]
        recent_messages = self.messages[-self.preserve_recent:]
        
        # Create a simple summary of the conversation
        summary_parts = []
        if self.summary:
            summary_parts.append(f"Previous conversation summary: {self.summary}")
            
        # Add key information from messages being summarized
        for msg in messages_to_summarize:
            if isinstance(msg, SystemMessage):
                continue  # Skip system messages in summary
            elif isinstance(msg, HumanMessage):
                content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                summary_parts.append(f"User: {content}")
            elif isinstance(msg, AIMessage):
                content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                agent_name = getattr(msg, 'name', 'Assistant')
                summary_parts.append(f"{agent_name}: {content}")
        
        # Update summary and keep only recent messages
        self.summary = "\n".join(summary_parts) if summary_parts else None
        self.messages = recent_messages
        
        logger.info(f"Conversation summarized. Keeping {len(recent_messages)} recent messages.")
    
    def _create_summary_for_content(self, content: str) -> str:
        """
        Create a summary for long content that exceeds reasonable limits.
        
        Args:
            content: The content to summarize
            
        Returns:
            Summarized content
        """
        if len(content) <= 500:  # Ultra-conservative threshold
            return content
            
        # Extract key information from the beginning and end with smaller chunks
        beginning = content[:200]  # Further reduced for safety
        ending = content[-200:]    # Further reduced for safety
        
        # Create a summary indicating truncation
        summary = f"{beginning}\n\n...Content summarized...\n\n{ending}"
        
        return summary
    
    def get_messages(self, summary_mode: bool = False) -> List[BaseMessage]:
        """
        Get conversation messages, optionally in summary mode.
        
        Args:
            summary_mode: If True, returns a condensed version suitable for context windows
            
        Returns:
            List of messages, potentially with summary prepended
        """
        if not summary_mode or not self.summary:
            return self.messages.copy()

        # In summary mode, prepend summary as a system message
        summary_message = SystemMessage(content=f"Conversation Summary:\n{self.summary}")
        return [summary_message] + self.messages.copy()
    
    def get_recent_messages(self, count: int) -> List[BaseMessage]:
        """
        Get the most recent N messages.
        
        Args:
            count: Number of recent messages to return
            
        Returns:
            List of recent messages
        """
        return self.messages[-count:] if count > 0 else []
    
    def clear(self) -> None:
        """
        Clear all conversation history and summary.
        """
        self.messages.clear()
        self.summary = None
        logger.info("Conversation history cleared.")
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current conversation.
        
        Returns:
            Dictionary with conversation statistics
        """
        total_content_length = sum(
            len(msg.content) if hasattr(msg, 'content') and isinstance(msg.content, str) else 0
            for msg in self.messages
        )
        
        return {
            "total_messages": len(self.messages),
            "has_summary": self.summary is not None,
            "summary_length": len(self.summary) if self.summary else 0,
            "total_content_length": total_content_length,
            "max_messages_threshold": self.max_messages,
            "preserve_recent_count": self.preserve_recent
        }


def create_conversation_manager(
    max_messages: int = MAX_MESSAGES_BEFORE_SUMMARY,
    max_content_length: int = MAX_CONTENT_LENGTH,
    preserve_recent: int = SUMMARY_PRESERVE_RECENT
) -> ConversationManager:
    """
    Factory function to create a ConversationManager instance.
    
    Args:
        max_messages: Maximum messages before summarization
        max_content_length: Maximum length for individual message content
        preserve_recent: Number of recent messages to preserve during summarization
        
    Returns:
        ConversationManager instance
    """
    return ConversationManager(
        max_messages=max_messages,
        max_content_length=max_content_length,
        preserve_recent=preserve_recent
    )
