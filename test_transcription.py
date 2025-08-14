"""
Simple test script to verify transcription works
"""

import numpy as np
import os
from speech_to_text import WhisperSTT

def test_transcription():
    """Test transcription with a simple audio file"""
    try:
        print("Testing transcription...")
        
        # Create a simple test audio file
        sample_rate = 16000
        duration = 3  # 3 seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create a simple sine wave (this won't be speech, but will test the file handling)
        frequency = 440  # A4 note
        audio_data = 0.1 * np.sin(2 * np.pi * frequency * t)
        
        # Save test audio file
        import scipy.io.wavfile as wavfile
        test_file = "test_audio.wav"
        wavfile.write(test_file, sample_rate, audio_data.astype(np.float32))
        
        print(f"Created test audio file: {test_file}")
        print(f"File exists: {os.path.exists(test_file)}")
        print(f"File size: {os.path.getsize(test_file)} bytes")
        
        # Test transcription
        stt = WhisperSTT()
        result = stt.transcribe_audio(audio_data, sample_rate)
        
        print(f"Transcription result: {result}")
        
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)
        if os.path.exists("temp_audio.wav"):
            os.unlink("temp_audio.wav")
            
    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_transcription() 