#!/usr/bin/env python3
"""
Script to set a female voice for the TTS system
"""

import sys
from text_to_speech import Pyttsx3TTS
from utils import colored_print, setup_logging

logger = setup_logging()

def list_available_voices():
    """List all available voices and identify female ones"""
    try:
        colored_print("=== Available Voices ===", "cyan", "bright")
        
        tts = Pyttsx3TTS()
        voices = tts.get_available_voices()
        
        if not voices:
            colored_print("No voices found", "red")
            return
        
        female_voices = []
        
        for voice in voices:
            voice_id = voice["id"]
            name = voice["name"]
            gender = voice.get("gender", "Unknown")
            languages = voice.get("languages", [])
            
            # Determine if it's likely a female voice
            is_female = (
                gender == "VoiceGenderFemale" or 
                "female" in name.lower() or 
                "woman" in name.lower() or
                "girl" in name.lower() or
                "sarah" in name.lower() or
                "zira" in name.lower() or
                "hazel" in name.lower()
            )
            
            status = "♀ FEMALE" if is_female else "♂ MALE"
            color = "green" if is_female else "blue"
            
            colored_print(f"{voice_id:2d}. {status} - {name}", color)
            
            if languages:
                lang_str = ", ".join(languages) if languages else "Unknown"
                colored_print(f"    Languages: {lang_str}", "white")
            
            if is_female:
                female_voices.append(voice_id)
        
        print()
        colored_print(f"Found {len(female_voices)} female voices", "green")
        
        if female_voices:
            colored_print("Female voice IDs:", "green")
            for vid in female_voices:
                colored_print(f"  - {vid}", "green")
        
        return female_voices
        
    except Exception as e:
        colored_print(f"Error listing voices: {e}", "red")
        return []

def test_voice(voice_id: int):
    """Test a specific voice"""
    try:
        colored_print(f"\n=== Testing Voice ID {voice_id} ===", "cyan", "bright")
        
        tts = Pyttsx3TTS()
        
        # Get voice info
        voices = tts.get_available_voices()
        if voice_id >= len(voices):
            colored_print(f"Voice ID {voice_id} not found", "red")
            return False
        
        voice_info = voices[voice_id]
        colored_print(f"Testing voice: {voice_info['name']}", "blue")
        
        # Switch to the voice
        if tts.switch_voice(voice_id):
            # Test the voice
            test_text = "Hello! I'm a female voice assistant for ox4labs. How can I help you today?"
            colored_print("Speaking test message...", "cyan")
            tts.speak_text(test_text)
            return True
        else:
            colored_print("Failed to switch to voice", "red")
            return False
            
    except Exception as e:
        colored_print(f"Error testing voice: {e}", "red")
        return False

def set_female_voice():
    """Set a female voice as default"""
    try:
        colored_print("=== Setting Female Voice ===", "cyan", "bright")
        
        # List available voices
        female_voices = list_available_voices()
        
        if not female_voices:
            colored_print("No female voices found", "red")
            return False
        
        # Try the first female voice
        voice_id = female_voices[0]
        colored_print(f"\nTrying first female voice (ID: {voice_id})...", "cyan")
        
        if test_voice(voice_id):
            # Update the config
            from config import update_config
            update_config("tts", "voice_id", voice_id)
            
            colored_print(f"\n✓ Successfully set female voice (ID: {voice_id}) as default!", "green", "bright")
            colored_print("The voice will be used in future sessions.", "green")
            return True
        else:
            colored_print("Failed to set female voice", "red")
            return False
            
    except Exception as e:
        colored_print(f"Error setting female voice: {e}", "red")
        return False

def main():
    """Main function"""
    try:
        colored_print("Female Voice Setup for ox4labs AI Assistant", "cyan", "bright")
        print()
        
        # Check if user wants to list voices or set female voice
        if len(sys.argv) > 1:
            if sys.argv[1] == "list":
                list_available_voices()
                return 0
            elif sys.argv[1] == "test" and len(sys.argv) > 2:
                voice_id = int(sys.argv[2])
                test_voice(voice_id)
                return 0
        
        # Default: set female voice
        success = set_female_voice()
        
        if success:
            colored_print("\n🎉 Female voice setup completed successfully!", "green", "bright")
            colored_print("You can now run the main application to hear the new voice.", "green")
            return 0
        else:
            colored_print("\n❌ Failed to set female voice", "red")
            return 1
            
    except Exception as e:
        colored_print(f"Fatal error: {e}", "red")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 