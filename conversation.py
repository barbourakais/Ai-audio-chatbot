"""
Conversation memory management for the AI Audio Agent
"""

import os
import json
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from config import get_config
from utils import format_timestamp, colored_print

@dataclass
class Message:
    """Represents a single message in the conversation"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    audio_duration: Optional[float] = None
    confidence: Optional[float] = None

@dataclass
class Conversation:
    """Represents a conversation session"""
    session_id: str
    start_time: str
    messages: List[Message]
    metadata: Dict

class ConversationManager:
    """Manages conversation history and context"""
    
    def __init__(self):
        self.config = get_config("conversation")
        self.current_conversation: Optional[Conversation] = None
        self.conversation_history: List[Conversation] = []
        self._load_conversations()
    
    def start_new_conversation(self) -> str:
        """Start a new conversation session"""
        session_id = f"session_{int(time.time())}"
        self.current_conversation = Conversation(
            session_id=session_id,
            start_time=format_timestamp(),
            messages=[],
            metadata={
                "model": get_config("ollama")["model"],
                "whisper_model": get_config("whisper")["model_size"],
                "tts_engine": get_config("tts")["engine"],
            }
        )
        colored_print(f"Started new conversation: {session_id}", "green")
        return session_id
    
    def add_user_message(self, content: str, audio_duration: Optional[float] = None, 
                        confidence: Optional[float] = None) -> None:
        """Add a user message to the current conversation"""
        if not self.current_conversation:
            self.start_new_conversation()
        
        message = Message(
            role="user",
            content=content,
            timestamp=format_timestamp(),
            audio_duration=audio_duration,
            confidence=confidence
        )
        
        self.current_conversation.messages.append(message)
        colored_print(f"🎤 User: {content}", "green")
    
    def add_assistant_message(self, content: str, audio_duration: Optional[float] = None) -> None:
        """Add an assistant message to the current conversation"""
        if not self.current_conversation:
            self.start_new_conversation()
        
        message = Message(
            role="assistant",
            content=content,
            timestamp=format_timestamp(),
            audio_duration=audio_duration
        )
        
        self.current_conversation.messages.append(message)
        colored_print(f"🤖 Assistant: {content}", "blue")
    
    def get_conversation_context(self, max_tokens: Optional[int] = None) -> str:
        """Get conversation context for the LLM"""
        if not self.current_conversation or not self.current_conversation.messages:
            return ""
        
        if max_tokens is None:
            max_tokens = self.config["context_window"]
        
        # Convert messages to conversation format
        context_parts = []
        current_tokens = 0
        
        # Get recent messages (last 10 exchanges = 20 messages max)
        recent_messages = self.current_conversation.messages[-20:]
        
        for message in recent_messages:
            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            estimated_tokens = len(message.content) // 4
            
            if current_tokens + estimated_tokens > max_tokens:
                break
            
            role = "User" if message.role == "user" else "Assistant"
            context_parts.append(f"{role}: {message.content}")
            current_tokens += estimated_tokens
        
        context = "\n".join(context_parts)
        
        # Debug: Show context info
        if context:
            colored_print(f"📝 Context: {len(self.current_conversation.messages)} messages, {len(context)} chars", "blue")
        
        return context
    
    def get_current_conversation(self) -> Optional[Conversation]:
        """Get the current conversation"""
        return self.current_conversation
    
    def get_recent_messages(self, count: int = 5) -> List[Message]:
        """Get the most recent messages from the current conversation"""
        if not self.current_conversation:
            return []
        
        return self.current_conversation.messages[-count:]
    
    def get_conversation_summary(self) -> Dict:
        """Get a summary of the current conversation"""
        if not self.current_conversation:
            return {}
        
        messages = self.current_conversation.messages
        user_messages = [m for m in messages if m.role == "user"]
        assistant_messages = [m for m in messages if m.role == "assistant"]
        
        total_duration = sum(
            m.audio_duration or 0 for m in messages if m.audio_duration
        )
        
        return {
            "session_id": self.current_conversation.session_id,
            "start_time": self.current_conversation.start_time,
            "total_messages": len(messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "total_audio_duration": total_duration,
            "last_message_time": messages[-1].timestamp if messages else None,
        }
    
    def clear_current_conversation(self) -> None:
        """Clear the current conversation"""
        if self.current_conversation:
            # Save the conversation before clearing
            self._save_conversation(self.current_conversation)
            self.current_conversation = None
        
        colored_print("Conversation history cleared", "yellow")
    
    def save_current_conversation(self) -> None:
        """Save the current conversation to disk"""
        if self.current_conversation:
            self._save_conversation(self.current_conversation)
    
    def _save_conversation(self, conversation: Conversation) -> None:
        """Save a conversation to disk"""
        if not self.config["save_conversations"]:
            return
        
        try:
            os.makedirs(self.config["conversation_dir"], exist_ok=True)
            
            # Convert conversation to dict
            conv_dict = {
                "session_id": conversation.session_id,
                "start_time": conversation.start_time,
                "messages": [asdict(msg) for msg in conversation.messages],
                "metadata": conversation.metadata
            }
            
            filename = f"{conversation.session_id}.json"
            filepath = os.path.join(self.config["conversation_dir"], filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conv_dict, f, indent=2, ensure_ascii=False)
            
            colored_print(f"Conversation saved: {filename}", "cyan")
            
        except Exception as e:
            colored_print(f"Error saving conversation: {e}", "red")
    
    def _load_conversations(self) -> None:
        """Load saved conversations from disk"""
        if not self.config["save_conversations"]:
            return
        
        try:
            if not os.path.exists(self.config["conversation_dir"]):
                return
            
            for filename in os.listdir(self.config["conversation_dir"]):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.config["conversation_dir"], filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            conv_data = json.load(f)
                        
                        # Convert dict back to Conversation object
                        messages = [
                            Message(**msg_data) for msg_data in conv_data["messages"]
                        ]
                        
                        conversation = Conversation(
                            session_id=conv_data["session_id"],
                            start_time=conv_data["start_time"],
                            messages=messages,
                            metadata=conv_data["metadata"]
                        )
                        
                        self.conversation_history.append(conversation)
                        
                    except Exception as e:
                        colored_print(f"Error loading conversation {filename}: {e}", "red")
            
            colored_print(f"Loaded {len(self.conversation_history)} saved conversations", "cyan")
            
        except Exception as e:
            colored_print(f"Error loading conversations: {e}", "red")
    
    def get_all_conversations(self) -> List[Dict]:
        """Get all conversations for API response"""
        conversations = []
        
        # Add current conversation if it exists
        if self.current_conversation:
            conversations.append({
                "id": self.current_conversation.session_id,
                "messages": [
                    {
                        "id": f"{self.current_conversation.session_id}_{i}",
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                        "confidence": msg.confidence,
                        "audioDuration": msg.audio_duration,
                    }
                    for i, msg in enumerate(self.current_conversation.messages)
                ],
                "createdAt": self.current_conversation.start_time,
                "updatedAt": self.current_conversation.messages[-1].timestamp if self.current_conversation.messages else self.current_conversation.start_time,
            })
        
        # Add saved conversations
        for conv in self.conversation_history:
            conversations.append({
                "id": conv.session_id,
                "messages": [
                    {
                        "id": f"{conv.session_id}_{i}",
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                        "confidence": msg.confidence,
                        "audioDuration": msg.audio_duration,
                    }
                    for i, msg in enumerate(conv.messages)
                ],
                "createdAt": conv.start_time,
                "updatedAt": conv.messages[-1].timestamp if conv.messages else conv.start_time,
            })
        
        return conversations
    
    def get_conversation_history(self) -> List[Dict]:
        """Get summary of all saved conversations"""
        summaries = []
        for conv in self.conversation_history:
            summary = self.get_conversation_summary()
            summary["session_id"] = conv.session_id
            summary["start_time"] = conv.start_time
            summaries.append(summary)
        return summaries
    
    def print_conversation_stats(self) -> None:
        """Print statistics about the current conversation"""
        if not self.current_conversation:
            colored_print("No active conversation", "yellow")
            return
        
        summary = self.get_conversation_summary()
        
        colored_print("\n=== Conversation Statistics ===", "cyan", "bright")
        colored_print(f"Session ID: {summary['session_id']}", "blue")
        colored_print(f"Start Time: {summary['start_time']}", "blue")
        colored_print(f"Total Messages: {summary['total_messages']}", "blue")
        colored_print(f"User Messages: {summary['user_messages']}", "green")
        colored_print(f"Assistant Messages: {summary['assistant_messages']}", "blue")
        colored_print(f"Total Audio Duration: {summary['total_audio_duration']:.1f}s", "blue")
        
        if summary['last_message_time']:
            colored_print(f"Last Message: {summary['last_message_time']}", "blue")
        
        print()
    
    def export_conversation(self, format: str = "json") -> Optional[str]:
        """Export the current conversation in the specified format"""
        if not self.current_conversation:
            return None
        
        if format.lower() == "json":
            conv_dict = {
                "session_id": self.current_conversation.session_id,
                "start_time": self.current_conversation.start_time,
                "messages": [asdict(msg) for msg in self.current_conversation.messages],
                "metadata": self.current_conversation.metadata
            }
            
            filename = f"export_{self.current_conversation.session_id}.json"
            filepath = os.path.join(self.config["conversation_dir"], filename)
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(conv_dict, f, indent=2, ensure_ascii=False)
                
                colored_print(f"Conversation exported: {filename}", "green")
                return filepath
                
            except Exception as e:
                colored_print(f"Error exporting conversation: {e}", "red")
                return None
        
        elif format.lower() == "txt":
            filename = f"export_{self.current_conversation.session_id}.txt"
            filepath = os.path.join(self.config["conversation_dir"], filename)
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"Conversation Session: {self.current_conversation.session_id}\n")
                    f.write(f"Start Time: {self.current_conversation.start_time}\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for message in self.current_conversation.messages:
                        role = "User" if message.role == "user" else "Assistant"
                        f.write(f"{role} ({message.timestamp}): {message.content}\n\n")
                
                colored_print(f"Conversation exported: {filename}", "green")
                return filepath
                
            except Exception as e:
                colored_print(f"Error exporting conversation: {e}", "red")
                return None
        
        else:
            colored_print(f"Unsupported export format: {format}", "red")
            return None 