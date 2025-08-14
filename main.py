#!/usr/bin/env python3
"""
Main entry point for the AI Audio Agent
"""

import sys
import signal
import threading
import time
from typing import Optional
from config import get_config
from utils import (
    colored_print, setup_logging, create_directories, 
    print_system_status, show_help, check_system_requirements
)
from audio_agent import AudioAgent

logger = setup_logging()

class AudioAgentApp:
    """Main application class for the AI Audio Agent"""
    
    def __init__(self):
        self.agent: Optional[AudioAgent] = None
        self.running = False
        self.debug_mode = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        colored_print(f"\nReceived signal {signum}, shutting down...", "yellow")
        self.shutdown()
        sys.exit(0)
    
    def initialize(self) -> bool:
        """Initialize the application"""
        try:
            colored_print("=== AI Audio Agent ===", "cyan", "bright")
            colored_print("Offline, Real-time AI Assistant", "cyan")
            colored_print("No paid APIs required!", "green")
            print()
            
            # Check system requirements
            requirements = check_system_requirements()
            
            if not all([
                requirements["python_ok"],
                requirements["ffmpeg"],
                requirements["ollama"],
                requirements["audio_devices"]
            ]):
                colored_print("⚠ Some system requirements are not met.", "yellow")
                colored_print("The application may not work properly.", "yellow")
                print()
                
                response = input("Continue anyway? (y/N): ").lower().strip()
                if response != 'y':
                    return False
            
            # Create directories
            create_directories()
            
            # Initialize audio agent
            colored_print("Initializing components...", "cyan")
            self.agent = AudioAgent()
            
            # Set up callbacks
            self._setup_callbacks()
            
            colored_print("✓ Application initialized successfully!", "green", "bright")
            return True
            
        except Exception as e:
            colored_print(f"✗ Failed to initialize application: {e}", "red")
            return False
    
    def _setup_callbacks(self):
        """Set up callbacks for the audio agent"""
        if not self.agent:
            return
        
        # Audio level callback
        def on_audio_level(level: float):
            if self.debug_mode and level > 0.01:
                bars = int(level * 50)
                print(f"\r🎤 Audio: {'█' * bars}{' ' * (50-bars)} {level:.3f}", end='', flush=True)
        
        # Transcription callback
        def on_transcription(text: str, confidence: float):
            if self.debug_mode:
                print(f"\n🎤 Transcribed: {text} (confidence: {confidence:.2f})")
        
        # Response callback
        def on_response(response: str):
            if self.debug_mode:
                print(f"\n🤖 Response: {response}")
        
        self.agent.set_audio_level_callback(on_audio_level)
        self.agent.set_transcription_callback(on_transcription)
        self.agent.set_response_callback(on_response)
    
    def run(self):
        """Run the main application loop"""
        if not self.agent:
            colored_print("Application not initialized", "red")
            return
        
        self.running = True
        
        # Show help
        show_help()
        
        # Start listening
        self.agent.start_listening()
        
        # Main input loop
        try:
            while self.running:
                # Check for keyboard input
                if self._check_keyboard_input():
                    continue
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            colored_print("\nInterrupted by user", "yellow")
        except Exception as e:
            colored_print(f"Error in main loop: {e}", "red")
        finally:
            self.shutdown()
    
    def _check_keyboard_input(self) -> bool:
        """Check for keyboard input (non-blocking)"""
        try:
            import msvcrt  # Windows
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').lower()
                return self._handle_key(key)
        except ImportError:
            try:
                import tty
                import termios
                import select
                
                # Check if input is available (non-blocking)
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1).lower()
                    return self._handle_key(key)
            except:
                pass
        
        return False
    
    def _handle_key(self, key: str) -> bool:
        """Handle keyboard input"""
        if key == 'q':
            colored_print("\nQuitting...", "yellow")
            self.running = False
            return True
        
        elif key == 'h':
            show_help()
            return True
        
        elif key == 'c':
            if self.agent:
                self.agent.clear_conversation()
            return True
        
        elif key == 's':
            print_system_status()
            return True
        
        elif key == 'd':
            self.debug_mode = not self.debug_mode
            status = "enabled" if self.debug_mode else "disabled"
            colored_print(f"Debug mode {status}", "cyan")
            return True
        
        elif key == 't':
            if self.agent:
                self._run_tests()
            return True
        
        elif key == 'i':
            if self.agent:
                self._show_info()
            return True
        
        elif key == 'v':  # Show conversation
            if self.agent:
                self._show_conversation()
            return True
        
        elif key == 'x':
            if self.agent:
                self.agent.save_conversation()
            return True
        
        return False
    
    def _run_tests(self):
        """Run component tests"""
        colored_print("\n=== Running Component Tests ===", "cyan", "bright")
        
        if not self.agent:
            colored_print("Agent not available", "red")
            return
        
        results = self.agent.test_components()
        
        colored_print("\nTest Results:", "cyan")
        for component, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            color = "green" if passed else "red"
            colored_print(f"  {component.upper()}: {status}", color)
        
        print()
    
    def _show_info(self):
        """Show system information"""
        colored_print("\n=== System Information ===", "cyan", "bright")
        
        if not self.agent:
            colored_print("Agent not available", "red")
            return
        
        info = self.agent.get_system_info()
        
        # STT Info
        stt_info = info.get("stt_model", {})
        if stt_info:
            colored_print(f"STT Model: {stt_info.get('model_size', 'unknown')}", "blue")
            colored_print(f"Device: {stt_info.get('device', 'unknown')}", "blue")
        
        # TTS Info
        tts_info = info.get("tts_model", {})
        if tts_info:
            colored_print(f"TTS Engine: {tts_info.get('rate', 'unknown')} WPM", "blue")
            colored_print(f"Volume: {tts_info.get('volume', 'unknown')}", "blue")
            colored_print(f"Available Voices: {tts_info.get('available_voices', 'unknown')}", "blue")
        
        # LLM Info
        llm_info = info.get("llm_model", {})
        if llm_info:
            colored_print(f"LLM Model: {llm_info.get('name', 'unknown')}", "blue")
        
        # Conversation Info
        conv_info = info.get("conversation", {})
        if conv_info:
            colored_print(f"Messages: {conv_info.get('total_messages', 0)}", "blue")
            colored_print(f"Duration: {conv_info.get('total_audio_duration', 0):.1f}s", "blue")
        
        print()
    
    def _show_conversation(self):
        """Show current conversation"""
        colored_print("\n=== Current Conversation ===", "cyan", "bright")
        
        if not self.agent:
            colored_print("Agent not available", "red")
            return
        
        conversation = self.agent.conversation.get_current_conversation()
        if not conversation or not conversation.messages:
            colored_print("No conversation history", "yellow")
            return
        
        for i, message in enumerate(conversation.messages):
            role_icon = "🎤" if message.role == "user" else "🤖"
            role_name = "You" if message.role == "user" else "Assistant"
            colored_print(f"{i+1}. {role_icon} {role_name}: {message.content}", "white")
        
        print()
    
    def shutdown(self):
        """Shutdown the application"""
        try:
            colored_print("Shutting down...", "cyan")
            
            if self.agent:
                self.agent.shutdown()
            
            colored_print("✓ Shutdown complete", "green")
            
        except Exception as e:
            colored_print(f"✗ Error during shutdown: {e}", "red")

def main():
    """Main entry point"""
    try:
        app = AudioAgentApp()
        
        if not app.initialize():
            colored_print("Failed to initialize application", "red")
            sys.exit(1)
        
        app.run()
        
    except Exception as e:
        colored_print(f"Fatal error: {e}", "red")
        sys.exit(1)

if __name__ == "__main__":
    main() 