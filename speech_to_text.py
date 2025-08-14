"""
Speech-to-Text module using OpenAI Whisper
"""

import os
import tempfile
import numpy as np
import whisper
import torch
from typing import Optional, Tuple, Dict, Any
from config import get_config
from utils import colored_print, setup_logging, normalize_audio

logger = setup_logging()

class WhisperSTT:
    """Speech-to-Text using OpenAI Whisper"""
    
    def __init__(self):
        self.config = get_config("whisper")
        self.model_size = self.config["model_size"]
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the Whisper model"""
        try:
            colored_print(f"Loading Whisper model: {self.model_size}", "cyan")
            colored_print(f"Device: {self.device}", "cyan")
            
            self.model = whisper.load_model(self.model_size, device=self.device)
            
            colored_print(f"✓ Whisper model loaded successfully", "green")
            
        except Exception as e:
            colored_print(f"✗ Error loading Whisper model: {e}", "red")
            raise
    
    def transcribe_audio(self, audio_data: np.ndarray, 
                        sample_rate: int = 16000) -> Dict[str, Any]:
        """Transcribe audio data to text"""
        try:
            if self.model is None:
                raise ValueError("Whisper model not loaded")
            
            # Normalize audio data
            audio_data = normalize_audio(audio_data)
            
            # Ensure audio is in the right format for Whisper
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Ensure audio is mono (single channel)
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
            
            # Transcribe using Whisper directly with audio data
            colored_print("🎤 Transcribing audio...", "cyan")
            
            result = self.model.transcribe(
                audio_data,
                language=self.config["language"],
                task=self.config["task"],
                temperature=self.config["temperature"],
                best_of=self.config["best_of"],
                beam_size=self.config["beam_size"],
                patience=self.config["patience"],
                length_penalty=self.config["length_penalty"],
                suppress_tokens=self.config["suppress_tokens"],
                initial_prompt=self.config["initial_prompt"],
                condition_on_previous_text=self.config["condition_on_previous_text"],
                no_speech_threshold=self.config["no_speech_threshold"],
            )
            
            # Extract results
            transcription = result.get("text", "").strip()
            language = result.get("language", "unknown")
            confidence = self._calculate_confidence(result)
            
            if transcription:
                colored_print(f"✓ Transcription: {transcription}", "green")
                if language != "unknown":
                    colored_print(f"  Language detected: {language}", "blue")
                if confidence > 0:
                    colored_print(f"  Confidence: {confidence:.2f}", "blue")
            else:
                colored_print("⚠ No speech detected", "yellow")
            
            return {
                "text": transcription,
                "language": language,
                "confidence": confidence,
                "segments": result.get("segments", []),
                "no_speech_prob": result.get("no_speech_prob", 0.0),
            }
                
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            colored_print(f"✗ Transcription error: {e}", "red")
            return {
                "text": "",
                "language": "unknown",
                "confidence": 0.0,
                "segments": [],
                "no_speech_prob": 1.0,
            }
    
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score from Whisper result"""
        try:
            segments = result.get("segments", [])
            if not segments:
                return 0.0
            
            # Calculate average confidence from segments
            confidences = []
            for segment in segments:
                if "avg_logprob" in segment:
                    # Convert log probability to confidence (0-1)
                    log_prob = segment["avg_logprob"]
                    confidence = np.exp(log_prob)
                    confidences.append(confidence)
            
            if confidences:
                return np.mean(confidences)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0
    
    def transcribe_file(self, file_path: str) -> Dict[str, Any]:
        """Transcribe audio from file"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            colored_print(f"Transcribing file: {file_path}", "cyan")
            
            result = self.model.transcribe(
                file_path,
                language=self.config["language"],
                task=self.config["task"],
                temperature=self.config["temperature"],
                best_of=self.config["best_of"],
                beam_size=self.config["beam_size"],
                patience=self.config["patience"],
                length_penalty=self.config["length_penalty"],
                suppress_tokens=self.config["suppress_tokens"],
                initial_prompt=self.config["initial_prompt"],
                condition_on_previous_text=self.config["condition_on_previous_text"],
                temperature_increment_on_fallback=self.config["temperature_increment_on_fallback"],
                compression_ratio_threshold=self.config["compression_ratio_threshold"],
                logprob_threshold=self.config["logprob_threshold"],
                no_speech_threshold=self.config["no_speech_threshold"],
            )
            
            transcription = result.get("text", "").strip()
            language = result.get("language", "unknown")
            confidence = self._calculate_confidence(result)
            
            if transcription:
                colored_print(f"✓ Transcription: {transcription}", "green")
            else:
                colored_print("⚠ No speech detected", "yellow")
            
            return {
                "text": transcription,
                "language": language,
                "confidence": confidence,
                "segments": result.get("segments", []),
                "no_speech_prob": result.get("no_speech_prob", 0.0),
            }
            
        except Exception as e:
            logger.error(f"Error transcribing file: {e}")
            colored_print(f"✗ Transcription error: {e}", "red")
            return {
                "text": "",
                "language": "unknown",
                "confidence": 0.0,
                "segments": [],
                "no_speech_prob": 1.0,
            }
    
    def detect_language(self, audio_data: np.ndarray, 
                       sample_rate: int = 16000) -> str:
        """Detect the language of audio data"""
        try:
            if self.model is None:
                return "unknown"
            
            # Normalize audio data
            audio_data = normalize_audio(audio_data)
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Save audio using scipy
                import scipy.io.wavfile as wavfile
                wavfile.write(temp_path, sample_rate, audio_data.astype(np.float32))
                
                # Detect language
                result = self.model.detect_language(temp_path)
                language = result[0] if result else "unknown"
                
                # Clean up temporary file
                os.unlink(temp_path)
                
                return language
                
            except Exception as e:
                # Clean up temporary file on error
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise e
                
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return "unknown"
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages"""
        return whisper.LANGUAGES.keys()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        info = {
            "model_size": self.model_size,
            "device": self.device,
        }
        
        # Safely get model attributes that might not exist
        try:
            info["multilingual"] = getattr(self.model, 'is_multilingual', False)
        except:
            info["multilingual"] = False
            
        try:
            info["dim"] = getattr(self.model, 'dims', {})
        except:
            info["dim"] = {}
            
        # Audio model attributes
        try:
            info["n_audio_ctx"] = getattr(self.model, 'n_audio_ctx', None)
        except:
            info["n_audio_ctx"] = None
            
        try:
            info["n_audio_state"] = getattr(self.model, 'n_audio_state', None)
        except:
            info["n_audio_state"] = None
            
        try:
            info["n_audio_head"] = getattr(self.model, 'n_audio_head', None)
        except:
            info["n_audio_head"] = None
            
        try:
            info["n_audio_layer"] = getattr(self.model, 'n_audio_layer', None)
        except:
            info["n_audio_layer"] = None
            
        # Text model attributes
        try:
            info["n_text_ctx"] = getattr(self.model, 'n_text_ctx', None)
        except:
            info["n_text_ctx"] = None
            
        try:
            info["n_text_state"] = getattr(self.model, 'n_text_state', None)
        except:
            info["n_text_state"] = None
            
        try:
            info["n_text_head"] = getattr(self.model, 'n_text_head', None)
        except:
            info["n_text_head"] = None
            
        try:
            info["n_text_layer"] = getattr(self.model, 'n_text_layer', None)
        except:
            info["n_text_layer"] = None
        
        return info
    
    def switch_model(self, model_size: str) -> bool:
        """Switch to a different Whisper model size"""
        try:
            colored_print(f"Switching to Whisper model: {model_size}", "cyan")
            
            # Load new model
            new_model = whisper.load_model(model_size, device=self.device)
            
            # Update configuration
            self.model_size = model_size
            self.model = new_model
            
            colored_print(f"✓ Switched to model: {model_size}", "green")
            return True
            
        except Exception as e:
            colored_print(f"✗ Error switching model: {e}", "red")
            return False 