"""
Utility functions for the AI Audio Agent
"""

import os
import time
import logging
import numpy as np
import sounddevice as sd
from typing import Optional, Tuple, List
from colorama import Fore, Style, init
from config import get_config

# Initialize colorama for cross-platform colored output
init(autoreset=True)

def setup_logging(name: str = None, level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('audio_agent.log')
        ]
    )
    return logging.getLogger(name or __name__)

def colored_print(text: str, color: str = "white", style: str = "normal") -> None:
    """Print colored text to console"""
    color_map = {
        "red": Fore.RED,
        "green": Fore.GREEN,
        "blue": Fore.BLUE,
        "yellow": Fore.YELLOW,
        "magenta": Fore.MAGENTA,
        "cyan": Fore.CYAN,
        "white": Fore.WHITE,
    }
    
    style_map = {
        "normal": Style.NORMAL,
        "bright": Style.BRIGHT,
        "dim": Style.DIM,
    }
    
    color_code = color_map.get(color.lower(), Fore.WHITE)
    style_code = style_map.get(style.lower(), Style.NORMAL)
    
    print(f"{color_code}{style_code}{text}{Style.RESET_ALL}")

def create_directories() -> None:
    """Create necessary directories for the application"""
    dirs = [
        get_config("conversation")["conversation_dir"],
        get_config("tts")["voice_dir"],
        get_config("dev")["audio_dir"],
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

def get_audio_devices() -> Tuple[List[dict], List[dict]]:
    """Get available audio input and output devices"""
    try:
        devices = sd.query_devices()
        input_devices = []
        output_devices = []
        
        for i, device in enumerate(devices):
            # Check if device has input capabilities
            if 'max_inputs' in device and device['max_inputs'] > 0:
                input_devices.append({
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_inputs'],
                    'sample_rate': device.get('default_samplerate', 44100)
                })
            
            # Check if device has output capabilities
            if 'max_outputs' in device and device['max_outputs'] > 0:
                output_devices.append({
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_outputs'],
                    'sample_rate': device.get('default_samplerate', 44100)
                })
        
        return input_devices, output_devices
    except Exception as e:
        logging.error(f"Error getting audio devices: {e}")
        return [], []

def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """Normalize audio data to prevent clipping"""
    if len(audio_data) == 0:
        return audio_data
    
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        return audio_data / max_val * 0.95
    return audio_data

def detect_silence(audio_data: np.ndarray, threshold: float = 0.01) -> bool:
    """Detect if audio data contains mostly silence"""
    if len(audio_data) == 0:
        return True
    
    rms = np.sqrt(np.mean(audio_data**2))
    return rms < threshold

def calculate_audio_level(audio_data: np.ndarray) -> float:
    """Calculate the RMS level of audio data"""
    if len(audio_data) == 0:
        return 0.0
    
    return np.sqrt(np.mean(audio_data**2))

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"

def format_timestamp() -> str:
    """Get current timestamp in a readable format"""
    return time.strftime("%Y-%m-%d %H:%M:%S")

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def is_valid_audio_file(file_path: str) -> bool:
    """Check if file is a valid audio file"""
    valid_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    return any(file_path.lower().endswith(ext) for ext in valid_extensions)

def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes"""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except OSError:
        return 0.0

def check_system_requirements() -> dict:
    """Check if system meets requirements"""
    requirements = {
        "python_version": "3.8+",
        "ffmpeg": False,
        "ollama": False,
        "audio_devices": False,
    }
    
    # Check Python version
    import sys
    python_version = sys.version_info
    requirements["python_ok"] = python_version.major == 3 and python_version.minor >= 8
    
    # Check FFmpeg
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True)
        requirements["ffmpeg"] = result.returncode == 0
    except FileNotFoundError:
        requirements["ffmpeg"] = False
    
    # Check Ollama
    try:
        import requests
        response = requests.get(get_config("ollama")["base_url"] + "/api/tags", 
                              timeout=5)
        requirements["ollama"] = response.status_code == 200
    except:
        requirements["ollama"] = False
    
    # Check audio devices
    input_devices, output_devices = get_audio_devices()
    requirements["audio_devices"] = len(input_devices) > 0 and len(output_devices) > 0
    requirements["input_devices"] = input_devices
    requirements["output_devices"] = output_devices
    
    return requirements

def print_system_status() -> None:
    """Print system status and requirements"""
    requirements = check_system_requirements()
    
    colored_print("\n=== System Requirements Check ===", "cyan", "bright")
    
    # Python version
    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    status = "✓" if requirements["python_ok"] else "✗"
    colored_print(f"{status} Python {python_version} (3.8+ required)", 
                  "green" if requirements["python_ok"] else "red")
    
    # FFmpeg
    status = "✓" if requirements["ffmpeg"] else "✗"
    colored_print(f"{status} FFmpeg installed", 
                  "green" if requirements["ffmpeg"] else "red")
    
    # Ollama
    status = "✓" if requirements["ollama"] else "✗"
    colored_print(f"{status} Ollama running", 
                  "green" if requirements["ollama"] else "red")
    
    # Audio devices
    status = "✓" if requirements["audio_devices"] else "✗"
    colored_print(f"{status} Audio devices available", 
                  "green" if requirements["audio_devices"] else "red")
    
    if requirements["input_devices"]:
        colored_print(f"  Input devices: {len(requirements['input_devices'])}", "blue")
        for device in requirements["input_devices"][:3]:  # Show first 3
            colored_print(f"    - {device['name']}", "blue")
    
    if requirements["output_devices"]:
        colored_print(f"  Output devices: {len(requirements['output_devices'])}", "blue")
        for device in requirements["output_devices"][:3]:  # Show first 3
            colored_print(f"    - {device['name']}", "blue")
    
    print()

def show_help() -> None:
    """Show help information"""
    colored_print("\n=== AI Audio Agent Help ===", "cyan", "bright")
    colored_print("Controls:", "yellow", "bright")
    colored_print("  🎤 Speak into your microphone to ask questions")
    colored_print("  🤖 The AI will respond with voice and text")
    colored_print("  'q' - Quit the application")
    colored_print("  'c' - Clear conversation history")
    colored_print("  'h' - Show this help")
    colored_print("  's' - Search knowledge base")
    colored_print("  'y' - Show system status")
    colored_print("  'd' - Toggle debug mode")
    colored_print("  'v' - View conversation history")
    colored_print("  'i' - Show system information")
    colored_print("  'x' - Save conversation")
    colored_print("  'u' - Update knowledge base")
    colored_print("  'k' - Show knowledge base stats")
    print()
    colored_print("Tips:", "yellow", "bright")
    colored_print("  - Speak clearly and at a normal volume")
    colored_print("  - Wait for the AI to finish speaking before asking another question")
    colored_print("  - The system remembers conversation context")
    colored_print("  - Use 'u' to update knowledge base when JSON content changes")
    colored_print("  - Use 's' to search the knowledge base directly")
    colored_print("  - Check the README.md for troubleshooting")
    print() 