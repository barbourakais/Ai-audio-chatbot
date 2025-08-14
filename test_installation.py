#!/usr/bin/env python3
"""
Test script to verify AI Audio Agent installation
"""

import sys
import os
from utils import colored_print, check_system_requirements, print_system_status

def test_imports():
    """Test if all required modules can be imported"""
    colored_print("Testing imports...", "cyan")
    
    modules = [
        ("numpy", "numpy"),
        ("sounddevice", "sounddevice"),
        ("scipy", "scipy"),
        ("requests", "requests"),
        ("colorama", "colorama"),
        ("torch", "torch"),
        ("torchaudio", "torchaudio"),
    ]
    
    failed_imports = []
    
    for module_name, import_name in modules:
        try:
            __import__(import_name)
            colored_print(f"✓ {module_name}", "green")
        except ImportError as e:
            colored_print(f"✗ {module_name}: {e}", "red")
            failed_imports.append(module_name)
    
    # Test optional modules
    optional_modules = [
        ("whisper", "openai-whisper"),
        ("TTS", "TTS"),
    ]
    
    for module_name, import_name in optional_modules:
        try:
            __import__(import_name)
            colored_print(f"✓ {module_name} (optional)", "green")
        except ImportError:
            colored_print(f"⚠ {module_name} not installed (will be downloaded on first use)", "yellow")
    
    return len(failed_imports) == 0

def test_system_requirements():
    """Test system requirements"""
    colored_print("\nTesting system requirements...", "cyan")
    
    requirements = check_system_requirements()
    
    all_passed = True
    
    # Python version
    if requirements["python_ok"]:
        colored_print("✓ Python version", "green")
    else:
        colored_print("✗ Python version (3.8+ required)", "red")
        all_passed = False
    
    # FFmpeg
    if requirements["ffmpeg"]:
        colored_print("✓ FFmpeg", "green")
    else:
        colored_print("✗ FFmpeg (required for Whisper)", "red")
        all_passed = False
    
    # Ollama
    if requirements["ollama"]:
        colored_print("✓ Ollama", "green")
    else:
        colored_print("✗ Ollama (required for LLM)", "red")
        all_passed = False
    
    # Audio devices
    if requirements["audio_devices"]:
        colored_print("✓ Audio devices", "green")
        input_count = len(requirements.get("input_devices", []))
        output_count = len(requirements.get("output_devices", []))
        colored_print(f"  Input devices: {input_count}", "blue")
        colored_print(f"  Output devices: {output_count}", "blue")
    else:
        colored_print("✗ Audio devices", "red")
        all_passed = False
    
    return all_passed

def test_config():
    """Test configuration loading"""
    colored_print("\nTesting configuration...", "cyan")
    
    try:
        from config import get_config
        
        config = get_config()
        required_sections = ["audio", "whisper", "ollama", "tts", "conversation"]
        
        for section in required_sections:
            if section in config:
                colored_print(f"✓ {section} config", "green")
            else:
                colored_print(f"✗ {section} config missing", "red")
                return False
        
        return True
        
    except Exception as e:
        colored_print(f"✗ Configuration error: {e}", "red")
        return False

def test_audio_components():
    """Test audio-related components"""
    colored_print("\nTesting audio components...", "cyan")
    
    try:
        import sounddevice as sd
        
        # Test audio devices
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_inputs'] > 0]
        output_devices = [d for d in devices if d['max_outputs'] > 0]
        
        if input_devices and output_devices:
            colored_print("✓ Audio devices detected", "green")
            colored_print(f"  Input: {len(input_devices)} devices", "blue")
            colored_print(f"  Output: {len(output_devices)} devices", "blue")
            return True
        else:
            colored_print("✗ No audio devices found", "red")
            return False
            
    except Exception as e:
        colored_print(f"✗ Audio test failed: {e}", "red")
        return False

def test_ollama_connection():
    """Test Ollama connection"""
    colored_print("\nTesting Ollama connection...", "cyan")
    
    try:
        import requests
        from config import get_config
        
        config = get_config("ollama")
        base_url = config["base_url"]
        
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            
            colored_print("✓ Ollama connection successful", "green")
            colored_print(f"  Available models: {model_names}", "blue")
            
            # Check if configured model is available
            configured_model = config["model"]
            if configured_model in model_names:
                colored_print(f"✓ Configured model '{configured_model}' available", "green")
                return True
            else:
                colored_print(f"⚠ Configured model '{configured_model}' not found", "yellow")
                colored_print(f"  Available: {model_names}", "blue")
                return False
        else:
            colored_print(f"✗ Ollama connection failed: {response.status_code}", "red")
            return False
            
    except Exception as e:
        colored_print(f"✗ Ollama test failed: {e}", "red")
        return False

def main():
    """Main test function"""
    colored_print("=== AI Audio Agent Installation Test ===", "cyan", "bright")
    print()
    
    tests = [
        ("Import Test", test_imports),
        ("System Requirements", test_system_requirements),
        ("Configuration", test_config),
        ("Audio Components", test_audio_components),
        ("Ollama Connection", test_ollama_connection),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            colored_print(f"✗ {test_name} failed with exception: {e}", "red")
            results.append((test_name, False))
    
    # Summary
    print()
    colored_print("=== Test Summary ===", "cyan", "bright")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        color = "green" if result else "red"
        colored_print(f"{status} {test_name}", color)
        if result:
            passed += 1
    
    print()
    colored_print(f"Results: {passed}/{total} tests passed", "cyan")
    
    if passed == total:
        colored_print("🎉 All tests passed! The AI Audio Agent is ready to use.", "green", "bright")
        colored_print("Run 'python main.py' to start the application.", "green")
    else:
        colored_print("⚠ Some tests failed. Please check the requirements and try again.", "yellow")
        colored_print("See README.md for troubleshooting information.", "yellow")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 