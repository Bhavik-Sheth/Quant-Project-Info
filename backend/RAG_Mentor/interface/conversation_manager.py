"""
Conversation Manager

Manages conversation context and history for the Trading Mentor chatbot.
"""

from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation state and history.
    """
    
    def __init__(self, max_history: int = 50):
        """
        Initialize conversation manager.
        
        Args:
            max_history: Maximum messages to keep in history
        """
        self.max_history = max_history
        self.conversation_history: List[Dict[str, str]] = []
        self.session_context: Dict[str, Any] = {}
        self.session_start = datetime.now()
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to conversation history.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
        """
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def set_context(self, key: str, value: Any) -> None:
        """
        Set a session context variable.
        
        Args:
            key: Context key (e.g., 'current_strategy', 'time_period')
            value: Context value
        """
        self.session_context[key] = value
        logger.debug(f"Context set: {key} = {value}")
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """
        Get a session context variable.
        
        Args:
            key: Context key
            default: Default value if key not found
            
        Returns:
            Context value
        """
        return self.session_context.get(key, default)
    
    def get_recent_history(self, n: int = 5) -> List[Dict[str, str]]:
        """
        Get recent conversation history.
        
        Args:
            n: Number of recent messages
            
        Returns:
            List of recent messages
        """
        return self.conversation_history[-n:]
    
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def reset_session(self) -> None:
        """Reset entire session including context"""
        self.conversation_history = []
        self.session_context = {}
        self.session_start = datetime.now()
        logger.info("Session reset")
    
    def get_conversation_summary(self) -> str:
        """
        Generate a summary of the conversation.
        
        Returns:
            Summary string
        """
        summary = f"Session started: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n"
        summary += f"Messages exchanged: {len(self.conversation_history)}\n"
        
        if self.session_context:
            summary += "\nActive Context:\n"
            for key, value in self.session_context.items():
                summary += f"  - {key}: {value}\n"
        
        return summary
    
    def save_conversation(self, filepath: str) -> None:
        """
        Save conversation to file.
        
        Args:
            filepath: Path to save conversation
        """
        data = {
            "session_start": self.session_start.isoformat(),
            "history": self.conversation_history,
            "context": self.session_context
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Conversation saved to {filepath}")
    
    def load_conversation(self, filepath: str) -> None:
        """
        Load conversation from file.
        
        Args:
            filepath: Path to load conversation from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.session_start = datetime.fromisoformat(data['session_start'])
        self.conversation_history = data['history']
        self.session_context = data['context']
        
        logger.info(f"Conversation loaded from {filepath}")
    
    def classify_intent(self, query: str) -> str:
        """
        Classify user query intent.
        
        Args:
            query: User query
            
        Returns:
            Intent category: 'show_trades', 'explain', 'compare', 'improve', 'general'
        """
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['show', 'display', 'list', 'trades', 'during']):
            return 'show_trades'
        elif any(word in query_lower for word in ['why', 'explain', 'reason', 'cause']):
            return 'explain'
        elif any(word in query_lower for word in ['compare', 'vs', 'versus', 'benchmark']):
            return 'compare'
        elif any(word in query_lower for word in ['improve', 'suggest', 'recommend', 'fix', 'better']):
            return 'improve'
        else:
            return 'general'
