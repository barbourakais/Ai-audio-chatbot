"""
Configuration settings for the AI Audio Agent
"""

import os
import json
from typing import Dict, Any


# Audio Configuration
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "channels": 1,
    "chunk_size": 1024,
    "format": "int16",
    "silence_threshold": 0.01,
    "silence_duration": 1.0,  # seconds
    "max_recording_duration": 30,  # seconds
}

# Whisper Configuration
WHISPER_CONFIG = {
    "model_size": "base",  # Options: tiny, base, small, medium, large
    "language": "en",  # Force English language detection
    "task": "transcribe",
    "temperature": 0.0,
    "best_of": 1,
    "beam_size": 5,
    "patience": 1.0,
    "length_penalty": 1.0,
    "suppress_tokens": [-1],
    "initial_prompt": None,
    "condition_on_previous_text": True,
    "no_speech_threshold": 0.6,

}

# Ollama Configuration
OLLAMA_CONFIG = {
    "base_url": "http://localhost:11434",
    "model": "llama3.2:3b",  # Using your installed llama3.2:3b model
    "temperature": 0.1,  # Reduced for more consistent responses
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "max_tokens": 300,  # Increased for more complete responses
    "system_prompt": None,  # Will be set dynamically from JSON content
}

# TTS Configuration
TTS_CONFIG = {
    "engine": "pyttsx3",
    "voice_rate": 140,  # Speed of speech (even slower for better clarity)
    "voice_volume": 0.9,  # Volume (0.0 to 1.0)
    "voice_id": None,  # None for default voice
    "language": "en",
    "voice_dir": "voices",  # Directory for voice files
}

# Conversation Configuration
CONVERSATION_CONFIG = {
    "max_history": 10,  # Number of exchanges to remember
    "context_window": 2000,  # Max tokens for context
    "include_timestamps": True,
    "save_conversations": True,
    "conversation_dir": "conversations",
}

# UI Configuration
UI_CONFIG = {
    "show_audio_levels": True,
    "show_transcription": True,
    "show_response": True,
    "colors": {
        "user": "green",
        "assistant": "blue",
        "system": "red",
        "warning": "yellow",
        "error": "red",
    },
    "prompt_symbol": "🎤",
    "response_symbol": "🤖",
}

# Development Configuration
DEV_CONFIG = {
    "debug": False,
    "log_level": "INFO",
    "save_audio_files": False,
    "audio_dir": "audio_samples",
    "test_mode": False,
}

# Combine all configurations
CONFIG = {
    "audio": AUDIO_CONFIG,
    "whisper": WHISPER_CONFIG,
    "ollama": OLLAMA_CONFIG,
    "tts": TTS_CONFIG,
    "conversation": CONVERSATION_CONFIG,
    "ui": UI_CONFIG,
    "dev": DEV_CONFIG,
}

def get_config(section: str = None) -> Dict[str, Any]:
    """Get configuration for a specific section or all configs"""
    if section:
        return CONFIG.get(section, {})
    return CONFIG

def update_config(section: str, key: str, value: Any) -> None:
    """Update a specific configuration value"""
    if section in CONFIG and key in CONFIG[section]:
        CONFIG[section][key] = value

def load_content_from_json() -> str:
    """Load content from Ox4labs.json file"""
    try:
        print("🔍 Loading Ox4labs.json...")
        with open('Ox4labs.json', 'r', encoding='utf-8') as f:
            content = json.load(f)
        print(f"✅ Successfully loaded JSON content ({len(json.dumps(content))} characters)")
        return json.dumps(content, indent=2)
    except FileNotFoundError:
        print("❌ Ox4labs.json file not found!")
        return "{}"
    except Exception as e:
        print(f"❌ Error loading Ox4labs.json: {e}")
        return "{}"

def get_system_prompt() -> str:
    """Get the system prompt with content from JSON file"""
    print("🎯 Generating system prompt...")
    content = load_content_from_json()
    prompt = f"""You are an AI assistant that answers strictly based on the provided content.
If the answer cannot be found in the content, respond with: "I don't have information about that."
Always respond in English only. If the user speaks in another language, politely ask them to speak in English.

CRITICAL RULES:
1. reply to the first question with Hey there!
2. NEVER repeat the user's question in your response
3. NEVER start with "Your process is..." or "Your services are..." or similar phrases
4. Give DIRECT answers only - no question repetition
5. When asked about "services" or "process" - list ONLY the titles/names, NO descriptions and if i ask about services dont mention the process and vice versa
6. When asked "what do you offer ..." - provide ONLY the offerings
7. When asked "how do you do..." or "what is..." or "describe..." - provide the description exactly how it is written in the json file
8. Be consistent - give the same answer every time
9. Keep responses concise and direct
10. Do not add commentary, suggestions, or advice
11. Do not offer additional information unless specifically asked
12. If the user asks about the phone number, provide the phone number from the json file
13. If the user asks about the email, provide the email from the json file
14. If the user asks about the website, provide the website from the json file

DIRECT ANSWERS (use these exact responses):
- "Where are you located?" → "Our Company is located in The Dot, lac 2, Tunisia."
- "What services do you offer?" → "For our services, we offer AI Consulting & Strategy, AI Prototyping & Development, and AI Training & Business Workshops."
- "What is your process?" → "Our process includes Align AI with Business Goals, Co-Creation & Rapid Prototyping, and Optimize, Scale & Govern."
- "How do you do AI consulting?" → Provide the description
- "What is your company name?" → "Our company name is ox4labs."
- "How can I contact you?" → Provide contact information including each title and the value
- "what are your offerings for AI Training & Business Workshops?" → for AI Training & Business Workshops we offer Provide the offerings

CRITICAL: Never repeat the question. Give direct answers only.

CONTENT: {content}"""
    print(f"✅ System prompt generated ({len(prompt)} characters)")
    return prompt

def load_env_config() -> None:
    """Load configuration from environment variables"""
    # Ollama settings
    if os.getenv("OLLAMA_MODEL"):
        CONFIG["ollama"]["model"] = os.getenv("OLLAMA_MODEL")
    if os.getenv("OLLAMA_BASE_URL"):
        CONFIG["ollama"]["base_url"] = os.getenv("OLLAMA_BASE_URL")
    
    # Whisper settings
    if os.getenv("WHISPER_MODEL_SIZE"):
        CONFIG["whisper"]["model_size"] = os.getenv("WHISPER_MODEL_SIZE")
    
    # TTS settings
    if os.getenv("TTS_MODEL_NAME"):
        CONFIG["tts"]["model_name"] = os.getenv("TTS_MODEL_NAME")
    
    # Debug settings
    if os.getenv("DEBUG"):
        CONFIG["dev"]["debug"] = os.getenv("DEBUG").lower() == "true"

# Load environment configuration on import
load_env_config() 