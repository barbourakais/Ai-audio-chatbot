"""
Main Audio Agent class that orchestrates all components
"""

import time
import threading
import numpy as np
import sounddevice as sd
from typing import Optional, Callable
from config import get_config
from utils import (
    colored_print, setup_logging, create_directories, 
    detect_silence, calculate_audio_level, format_duration
)
from speech_to_text import WhisperSTT
from text_to_speech import Pyttsx3TTS
from llm_client import OllamaClient
from conversation import ConversationManager

logger = setup_logging()

class AudioAgent:
    """Main AI Audio Agent that handles voice interaction"""
    
    def __init__(self):
        self.config = get_config()
        self.audio_config = self.config["audio"]
        self.dev_config = self.config["dev"]
        
        # Initialize components
        self.stt = None
        self.tts = None
        self.llm = None
        self.conversation = None
        
        # Audio recording state
        self.is_recording = False
        self.is_processing = False
        self.recording_thread = None
        self.audio_buffer = []
        self.silence_start = None
        
        # Callbacks
        self.on_audio_level = None
        self.on_transcription = None
        self.on_response = None
        
        # Initialize the agent
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all components of the audio agent"""
        try:
            colored_print("Initializing AI Audio Agent...", "cyan", "bright")
            
            # Create necessary directories
            create_directories()
            
            # Initialize conversation manager
            colored_print("Loading conversation manager...", "cyan")
            self.conversation = ConversationManager()
            
            # Initialize speech-to-text
            colored_print("Loading speech-to-text...", "cyan")
            self.stt = WhisperSTT()
            
            # Initialize text-to-speech
            colored_print("Loading text-to-speech...", "cyan")
            self.tts = Pyttsx3TTS()
            
            # Initialize LLM client
            colored_print("Connecting to LLM...", "cyan")
            self.llm = OllamaClient()
            
            colored_print("✓ AI Audio Agent initialized successfully!", "green", "bright")
            
        except Exception as e:
            colored_print(f"✗ Error initializing Audio Agent: {e}", "red")
            raise
    
    def start_listening(self) -> None:
        """Start listening for voice input"""
        if self.is_recording:
            colored_print("Already listening...", "yellow")
            return
        
        try:
            self.is_recording = True
            self.audio_buffer = []
            self.silence_start = None
            
            colored_print("🎤 Listening for voice input...", "green", "bright")
            colored_print("(Press 'q' to quit, 'h' for help)", "cyan")
            
            # Start recording in a separate thread
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
        except Exception as e:
            colored_print(f"✗ Error starting audio recording: {e}", "red")
            self.is_recording = False
    
    def stop_listening(self) -> None:
        """Stop listening for voice input"""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=1.0)
        colored_print("Stopped listening", "yellow")
    
    def _record_audio(self) -> None:
        """Record audio from microphone"""
        try:
            # Use a simpler approach without callback
            colored_print("🎤 Recording started...", "cyan")
            
            # Create stream without context manager to keep it open
            stream = sd.InputStream(
                channels=self.audio_config["channels"],
                samplerate=self.audio_config["sample_rate"],
                dtype=np.float32,
                blocksize=self.audio_config["chunk_size"]
            )
            stream.start()
            
            try:
                while self.is_recording:
                    try:
                        # Read audio data
                        audio_chunk, overflowed = stream.read(self.audio_config["chunk_size"])
                        if overflowed:
                            logger.warning("Audio buffer overflow")
                        
                        # Convert to float and add to buffer
                        audio_chunk = audio_chunk[:, 0].astype(np.float32)
                        self.audio_buffer.append(audio_chunk)
                        
                        # Calculate audio level
                        level = calculate_audio_level(audio_chunk)
                        
                        # Check for silence
                        if detect_silence(audio_chunk, self.audio_config["silence_threshold"]):
                            if self.silence_start is None:
                                self.silence_start = time.time()
                        else:
                            self.silence_start = None
                        
                        # Call audio level callback
                        if self.on_audio_level:
                            self.on_audio_level(level)
                        
                        # Check if we should stop recording
                        silence_duration = self.audio_config["silence_duration"]
                        max_duration = self.audio_config["max_recording_duration"]
                        
                        # Only process if we have significant audio (not just silence)
                        if (self.silence_start and 
                            time.time() - self.silence_start > silence_duration and
                            len(self.audio_buffer) > 0):
                            
                            # Check if we have meaningful audio content
                            total_audio = np.concatenate(self.audio_buffer)
                            audio_level = calculate_audio_level(total_audio)
                            
                            if audio_level > 0.02:  # Only process if audio level is above threshold
                                colored_print("Speech detected, processing audio...", "cyan")
                                self._process_audio()
                            else:
                                colored_print("Silence detected, ignoring...", "yellow")
                                self._reset_recording()
                            # Continue listening after processing
                            continue
                        
                        elif (len(self.audio_buffer) * self.audio_config["chunk_size"] / self.audio_config["sample_rate"] > max_duration):
                            colored_print("Maximum recording duration reached", "yellow")
                            self._process_audio()
                            # Continue listening after processing
                            continue
                        
                        time.sleep(0.01)  # Small delay to prevent CPU overuse
                        
                    except Exception as e:
                        logger.error(f"Error reading audio: {e}")
                        # Don't break, just continue listening
                        continue
            finally:
                # Clean up stream when done
                stream.stop()
                stream.close()
                        
        except Exception as e:
            logger.error(f"Error in audio recording: {e}")
            colored_print(f"✗ Audio recording error: {e}", "red")
            self.is_recording = False
    
    def _process_audio(self) -> None:
        """Process recorded audio through the pipeline"""
        if self.is_processing or not self.audio_buffer:
            return
        
        self.is_processing = True
        
        try:
            # Combine audio chunks
            audio_data = np.concatenate(self.audio_buffer)
            recording_duration = len(audio_data) / self.audio_config["sample_rate"]
            
            colored_print(f"Processing {format_duration(recording_duration)} of audio...", "cyan")
            
            # Check if audio is too short
            if recording_duration < 1.0:  # Increased minimum duration
                colored_print("Audio too short, ignoring", "yellow")
                self._reset_recording()
                return
            
            # Check if audio level is sufficient for speech
            audio_level = calculate_audio_level(audio_data)
            if audio_level < 0.01:  # Very low audio level
                colored_print("Audio level too low, ignoring", "yellow")
                self._reset_recording()
                return
            
            # Transcribe audio
            transcription_result = self.stt.transcribe_audio(
                audio_data, 
                self.audio_config["sample_rate"]
            )
            
            transcription = transcription_result["text"]
            confidence = transcription_result["confidence"]
            
            if not transcription:
                colored_print("No speech detected", "yellow")
                self._reset_recording()
                return
            
            # Call transcription callback
            if self.on_transcription:
                self.on_transcription(transcription, confidence)
            
            # Add to conversation
            self.conversation.add_user_message(
                transcription, 
                audio_duration=recording_duration,
                confidence=confidence
            )
            
            # Generate response
            self._generate_and_speak_response(transcription)
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            colored_print(f"✗ Error processing audio: {e}", "red")
        finally:
            # Reset for next recording
            self._reset_recording()
    
    def _reset_recording(self) -> None:
        """Reset recording state for next input"""
        self.audio_buffer = []
        self.silence_start = None
        self.is_processing = False
        
        colored_print("🎤 Ready for your next question...", "green")
    
    def _generate_and_speak_response(self, user_input: str) -> None:
        """Generate LLM response and speak it"""
        try:
            # Get conversation context
            context = self.conversation.get_conversation_context()
            
            # Debug: Show conversation context
            if context:
                colored_print(f"📝 Context ({len(context)} chars): {context[:100]}...", "blue")
            else:
                colored_print("📝 No conversation context", "blue")
            
            # Generate response
            response = self.llm.generate_response(user_input, context)
            
            if not response:
                colored_print("No response generated", "yellow")
                return
            
            # Add response to conversation
            self.conversation.add_assistant_message(response)
            
            # Call response callback
            if self.on_response:
                self.on_response(response)
            
            # Speak the response and wait for it to complete
            try:
                colored_print("🔊 Speaking response...", "cyan")
                self.tts.speak_text(response)
                colored_print("✓ Speech completed", "green")
            except Exception as e:
                colored_print(f"TTS error: {e}", "red")
            
        except Exception as e:
            logger.error(f"Error generating/speaking response: {e}")
            colored_print(f"✗ Error generating response: {e}", "red")
    

    
    
    
    def process_text_input(self, text: str) -> None:
        """Process text input directly (for testing or non-voice input)"""
        try:
            if not text.strip():
                return
            
            colored_print(f"Processing text input: {text}", "cyan")
            
            # Add to conversation
            self.conversation.add_user_message(text)
            
            # Generate and speak response
            self._generate_and_speak_response(text)
            
        except Exception as e:
            logger.error(f"Error processing text input: {e}")
            colored_print(f"Error processing text input: {e}", "red")
    
    def clear_conversation(self) -> None:
        """Clear the current conversation"""
        self.conversation.clear_current_conversation()
    
    def save_conversation(self) -> None:
        """Save the current conversation"""
        self.conversation.save_current_conversation()
    
    def get_conversation_stats(self) -> dict:
        """Get conversation statistics"""
        return self.conversation.get_conversation_summary()
    
    def print_conversation_stats(self) -> None:
        """Print conversation statistics"""
        self.conversation.print_conversation_stats()
    
    def set_audio_level_callback(self, callback: Callable[[float], None]) -> None:
        """Set callback for audio level updates"""
        self.on_audio_level = callback
    
    def set_transcription_callback(self, callback: Callable[[str, float], None]) -> None:
        """Set callback for transcription updates"""
        self.on_transcription = callback
    
    def set_response_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for response updates"""
        self.on_response = callback
    
    def test_components(self) -> dict:
        """Test all components and return status"""
        results = {
            "stt": False,
            "tts": False,
            "llm": False,
            "audio": False,
        }
        
        try:
            # Test STT
            colored_print("Testing Speech-to-Text...", "cyan")
            test_audio = np.random.randn(16000) * 0.1  # 1 second of noise
            result = self.stt.transcribe_audio(test_audio)
            results["stt"] = True
            colored_print("✓ STT test passed", "green")
            
        except Exception as e:
            colored_print(f"✗ STT test failed: {e}", "red")
        
        try:
            # Test TTS
            colored_print("Testing Text-to-Speech...", "cyan")
            results["tts"] = self.tts.test_synthesis("Test")
            
        except Exception as e:
            colored_print(f"✗ TTS test failed: {e}", "red")
        
        try:
            # Test LLM
            colored_print("Testing LLM...", "cyan")
            response = self.llm.generate_response("Hello")
            results["llm"] = bool(response)
            
        except Exception as e:
            colored_print(f"✗ LLM test failed: {e}", "red")
        
        try:
            # Test audio devices
            colored_print("Testing audio devices...", "cyan")
            from utils import get_audio_devices
            input_devices, output_devices = get_audio_devices()
            results["audio"] = len(input_devices) > 0 and len(output_devices) > 0
            
        except Exception as e:
            colored_print(f"✗ Audio test failed: {e}", "red")
        
        return results
    
    def get_system_info(self) -> dict:
        """Get system information"""
        return {
            "stt_model": self.stt.get_model_info() if self.stt else {},
            "tts_model": self.tts.get_model_info() if self.tts else {},
            "llm_model": self.llm.get_model_info() if self.llm else {},
            "conversation": self.conversation.get_conversation_summary() if self.conversation else {},
            "audio_config": self.audio_config,
        }
    
    def shutdown(self) -> None:
        """Shutdown the audio agent"""
        try:
            colored_print("Shutting down Audio Agent...", "cyan")
            
            # Stop recording
            self.stop_listening()
            
            # Save conversation
            if self.conversation:
                self.conversation.save_current_conversation()
            
            colored_print("✓ Audio Agent shutdown complete", "green")
            
        except Exception as e:
            colored_print(f"✗ Error during shutdown: {e}", "red") 