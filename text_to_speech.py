"""
Text-to-Speech functionality using pyttsx3
"""

import pyttsx3
from typing import Optional, List, Dict, Any
from config import get_config
from utils import colored_print, setup_logging

logger = setup_logging(__name__)

class Pyttsx3TTS:
    """Text-to-Speech using pyttsx3"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize TTS with configuration"""
        self.config = config or get_config("tts")
        colored_print("✓ TTS system ready (fresh engine per speech)", "green")
    
    def _load_engine(self):
        """Load the TTS engine"""
        try:
            self.engine = pyttsx3.init()
            
            # Configure engine settings
            self.engine.setProperty('rate', self.config.get("voice_rate", 150))
            self.engine.setProperty('volume', self.config.get("voice_volume", 0.9))
            
            # Set voice if specified
            voice_id = self.config.get("voice_id")
            if voice_id:
                voices = self.engine.getProperty('voices')
                if voice_id < len(voices):
                    self.engine.setProperty('voice', voices[voice_id].id)
            
            colored_print("✓ Loaded pyttsx3 TTS engine", "green")
        except Exception as e:
            colored_print(f"✗ Failed to load TTS engine: {e}", "red")
            raise
    
    def _clean_text_for_speech(self, text: str) -> str:
        """Clean text to make it more natural for speech"""
        import re
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Remove code
        
        # Remove bullet points and numbers
        text = re.sub(r'^\s*[-*•]\s*', '', text, flags=re.MULTILINE)  # Remove bullet points
        text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)  # Remove numbered lists
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove extra punctuation
        text = re.sub(r'\.{2,}', '.', text)  # Multiple dots to single dot
        text = re.sub(r'!{2,}', '!', text)   # Multiple exclamation to single
        text = re.sub(r'\?{2,}', '?', text)  # Multiple question to single
        
        # Clean up spacing around punctuation
        text = re.sub(r'\s+([.!?])', r'\1', text)
        
        return text.strip()
    
    def speak_text(self, text: str) -> None:
        """Speak text directly through speakers"""
        try:
            if not text.strip():
                return
            
            # Clean the text for better speech
            cleaned_text = self._clean_text_for_speech(text)
            
            colored_print(f"🔊 Speaking: {cleaned_text[:50]}{'...' if len(cleaned_text) > 50 else ''}", "cyan")
            
            # Create a fresh engine for each speech request to avoid run loop conflicts
            engine = None
            try:
                engine = pyttsx3.init()
                
                # Configure engine settings
                engine.setProperty('rate', self.config.get("voice_rate", 150))
                engine.setProperty('volume', self.config.get("voice_volume", 0.9))
                
                # Set voice if specified
                voice_id = self.config.get("voice_id")
                if voice_id:
                    voices = engine.getProperty('voices')
                    if voice_id < len(voices):
                        engine.setProperty('voice', voices[voice_id].id)
                
                # Speak the text
                engine.say(cleaned_text)
                engine.runAndWait()
                colored_print("✓ Speech playback completed", "green")
                
            except Exception as e:
                colored_print(f"✗ Speech playback failed: {e}", "red")
                raise
            finally:
                # Clean up the engine
                if engine:
                    try:
                        engine.stop()
                    except:
                        pass
        except Exception as e:
            colored_print(f"✗ Speech playback failed: {e}", "red")
            raise
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices"""
        try:
            # Create a temporary engine to get voices
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            voice_info = []
            for i, voice in enumerate(voices):
                voice_info.append({
                    "id": i,
                    "name": voice.name,
                    "languages": voice.languages,
                    "gender": voice.gender,
                    "age": voice.age
                })
            # Clean up
            try:
                engine.stop()
            except:
                pass
            return voice_info
        except Exception as e:
            logger.error(f"Failed to get available voices: {e}")
            return []
    
    def switch_voice(self, voice_id: int) -> bool:
        """Switch to a different voice"""
        try:
            # Create a temporary engine to check voices
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            if 0 <= voice_id < len(voices):
                self.config["voice_id"] = voice_id
                colored_print(f"✓ Switched to voice ID: {voice_id}", "green")
                # Clean up
                try:
                    engine.stop()
                except:
                    pass
                return True
            # Clean up
            try:
                engine.stop()
            except:
                pass
            return False
        except Exception as e:
            colored_print(f"✗ Failed to switch voice: {e}", "red")
            return False
    
    def set_rate(self, rate: int) -> None:
        """Set speech rate"""
        self.config["voice_rate"] = rate
        colored_print(f"✓ Speech rate set to: {rate}", "green")
    
    def set_volume(self, volume: float) -> None:
        """Set speech volume"""
        volume = max(0.0, min(1.0, volume))
        self.config["voice_volume"] = volume
        colored_print(f"✓ Speech volume set to: {volume}", "green")
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about current engine"""
        try:
            return {
                "rate": self.config.get("voice_rate", 150),
                "volume": self.config.get("voice_volume", 0.9),
                "voice_id": self.config.get("voice_id"),
                "available_voices": len(self.get_available_voices())
            }
        except Exception as e:
            logger.error(f"Failed to get engine info: {e}")
            return {}
    
    def test_synthesis(self, text: str = "Hello, this is a test of the text-to-speech system.") -> bool:
        """Test the TTS system"""
        try:
            colored_print("Testing TTS system...", "cyan")
            self.speak_text(text)
            return True
        except Exception as e:
            colored_print(f"✗ TTS test failed: {e}", "red")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the TTS model (compatibility method)"""
        return self.get_engine_info() 