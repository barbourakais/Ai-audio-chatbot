# AI Audio Agent - Spike Document

## Project Overview

The AI Audio Agent is an offline, real-time voice assistant that processes speech input, generates intelligent responses using local LLMs, and provides audio output—all without requiring paid APIs or internet connectivity.

## Core Architecture

### System Components

1. **Audio Capture & Processing**
   - Real-time microphone input via `sounddevice`
   - Audio buffering and silence detection
   - Audio level monitoring and visualization
   - Configurable recording parameters (sample rate, chunk size, silence thresholds)

2. **Speech-to-Text (STT)**
   - OpenAI Whisper integration for offline transcription
   - Support for multiple model sizes (tiny, base, small, medium, large)
   - Language auto-detection
   - Confidence scoring and noise filtering
   - GPU acceleration when available

3. **Language Model (LLM)**
   - Ollama integration for local LLM inference
   - Configurable model selection (llama2, llama3, mistral, etc.)
   - Conversation context management
   - System prompt customization for Light Speed Tech branding
   - Response generation with temperature and sampling controls

4. **Text-to-Speech (TTS)**
   - Pyttsx3 engine for offline speech synthesis
   - Configurable voice parameters (rate, volume, voice selection)
   - Multi-language support
   - Voice file management

5. **Conversation Management**
   - Conversation history tracking
   - Context window management
   - Timestamp recording
   - Conversation persistence and export
   - Memory optimization for long conversations

6. **Configuration System**
   - Centralized configuration management
   - Environment variable support
   - Runtime configuration updates
   - Component-specific settings

## Component Selection Rationale

### Why These Specific Components?

#### 1. **Audio Capture: sounddevice**
**Why sounddevice over alternatives:**
- **Cross-platform compatibility**: Works seamlessly on Windows, macOS, and Linux
- **Real-time performance**: Low-latency audio capture with minimal buffering
- **Python-native**: Pure Python implementation with no external dependencies
- **Active maintenance**: Regularly updated with bug fixes and improvements
- **Simple API**: Easy to use for both simple and complex audio applications
- **Alternative considered**: PyAudio was rejected due to complex installation requirements and platform-specific issues

#### 2. **Speech-to-Text: OpenAI Whisper**
**Why Whisper over alternatives:**
- **Offline capability**: No internet connection required after model download
- **High accuracy**: State-of-the-art transcription quality (85-95% accuracy)
- **Multi-language support**: Automatic language detection and transcription
- **Multiple model sizes**: From tiny (39MB) to large (1.5GB) for different use cases
- **GPU acceleration**: CUDA support for faster processing
- **Open source**: No licensing fees or usage limits
- **Alternatives considered**: 
  - Google Speech-to-Text: Requires internet and API keys
  - Mozilla DeepSpeech: Lower accuracy and limited language support
  - Vosk: Good but less accurate than Whisper

#### 3. **Language Model: Ollama**
**Why Ollama over alternatives:**
- **Local deployment**: Runs entirely on user's machine
- **No API costs**: No per-request charges or usage limits
- **Model variety**: Support for llama2, llama3, mistral, and other open models
- **Easy setup**: Simple installation and model management
- **HTTP API**: Standard REST interface for easy integration
- **Resource efficient**: Optimized for consumer hardware
- **Alternatives considered**:
  - OpenAI GPT: Requires internet and paid API
  - Hugging Face Transformers: More complex setup and higher resource usage
  - LocalAI: Less mature and fewer supported models

#### 4. **Text-to-Speech: Pyttsx3**
**Why Pyttsx3 over alternatives:**
- **Offline operation**: No internet connection required
- **System integration**: Uses native system TTS engines
- **Cross-platform**: Works on Windows (SAPI), macOS (NSSpeechSynthesizer), Linux (espeak)
- **Lightweight**: Minimal resource usage compared to neural TTS
- **Fast synthesis**: Real-time speech generation
- **No model downloads**: Uses pre-installed system voices
- **Alternatives considered**:
  - Coqui TTS: Requires large model downloads and more resources
  - gTTS (Google): Requires internet connection
  - Azure Speech: Requires internet and API keys

#### 5. **Audio Processing: NumPy + SciPy**
**Why NumPy/SciPy over alternatives:**
- **Industry standard**: De facto standard for scientific computing in Python
- **Performance**: Optimized C implementations for fast array operations
- **Ecosystem**: Extensive library support and community
- **Audio processing**: Built-in functions for FFT, filtering, and signal processing
- **Memory efficiency**: Optimized for large audio data arrays
- **Alternatives considered**:
  - PyTorch: Overkill for simple audio processing
  - TensorFlow: Too heavy for this use case
  - Pure Python: Too slow for real-time processing

#### 6. **HTTP Client: Requests**
**Why Requests over alternatives:**
- **Python standard**: Most popular HTTP library for Python
- **Simple API**: Easy to use and understand
- **Session management**: Built-in connection pooling and reuse
- **Error handling**: Comprehensive exception handling
- **Mature**: Stable and well-tested
- **Alternatives considered**:
  - urllib: More verbose and complex API
  - httpx: Good but newer and less mature
  - aiohttp: Overkill for simple HTTP requests

#### 7. **Configuration: Custom Config System**
**Why custom config over alternatives:**
- **Lightweight**: No external dependencies
- **Flexible**: Easy to extend and modify
- **Environment support**: Built-in environment variable integration
- **Type safety**: Structured configuration with validation
- **Runtime updates**: Ability to modify settings without restart
- **Alternatives considered**:
  - Pydantic: Overkill for simple configuration
  - ConfigParser: Limited to INI format
  - YAML/JSON: Requires external parsing libraries

## Technical Specifications

### Dependencies
```
Core Dependencies:
- openai-whisper==20231117 (Speech-to-Text)
- numpy>=1.21.6,<1.28.0 (Audio processing)
- sounddevice==0.4.6 (Audio capture)
- scipy>=1.11.2 (Signal processing)
- ollama==0.1.7 (LLM client)
- pyttsx3==2.90 (Text-to-Speech)
- requests==2.31.0 (HTTP client)
- colorama==0.4.6 (Terminal colors)

Development Dependencies:
- flask==3.0.0 (API server)
- flask-cors==4.0.0 (CORS support)
- python-dotenv==1.0.0 (Environment management)
```

### System Requirements
- **Python**: 3.8+
- **Ollama**: Local LLM server with compatible model
- **FFmpeg**: Audio processing (required for Whisper)
- **Audio Devices**: Microphone input and speaker output
- **Memory**: 4GB+ RAM (8GB+ recommended for larger models)
- **Storage**: 2GB+ free space for models

### Performance Characteristics
- **Latency**: 1-3 seconds end-to-end response time
- **Accuracy**: 85-95% transcription accuracy (Whisper base model)
- **Memory Usage**: 2-4GB RAM depending on model sizes
- **CPU Usage**: Moderate (varies with model complexity)

## Component Analysis

### AudioAgent Class (`audio_agent.py`)
**Purpose**: Main orchestrator class that coordinates all components

**Key Responsibilities**:
- Component initialization and lifecycle management
- Audio recording and processing pipeline
- Real-time audio level monitoring
- Silence detection and automatic processing triggers
- Callback management for UI integration
- Error handling and recovery

**Critical Methods**:
- `_initialize_components()`: Sets up STT, TTS, LLM, and conversation manager
- `_record_audio()`: Continuous audio capture with silence detection
- `_process_audio()`: Audio processing pipeline execution
- `_generate_and_speak_response()`: LLM response generation and TTS output

**Threading Model**:
- Main thread: User interface and control
- Recording thread: Continuous audio capture
- Processing thread: Audio transcription and response generation

### WhisperSTT Class (`speech_to_text.py`)
**Purpose**: Speech-to-text conversion using OpenAI Whisper

**Key Features**:
- Model loading and caching
- Audio preprocessing and normalization
- Multi-language support
- Confidence scoring
- GPU acceleration support

**Configuration Options**:
- Model size selection (tiny/base/small/medium/large)
- Language specification or auto-detection
- Transcription parameters (temperature, beam size, etc.)
- Noise suppression thresholds

### OllamaClient Class (`llm_client.py`)
**Purpose**: Local LLM interaction via Ollama API

**Key Features**:
- HTTP client for Ollama server communication
- Model validation and connection testing
- Context-aware prompt construction
- Response cleaning and formatting
- Error handling and retry logic

**Configuration Options**:
- Model selection (llama2, llama3, mistral, etc.)
- Generation parameters (temperature, top_p, top_k)
- System prompt customization
- Response length limits

### Pyttsx3TTS Class (`text_to_speech.py`)
**Purpose**: Text-to-speech synthesis using Pyttsx3

**Key Features**:
- Voice selection and management
- Speech rate and volume control
- Multi-language support
- Voice file caching

### ConversationManager Class (`conversation.py`)
**Purpose**: Conversation history and context management

**Key Features**:
- Message history tracking
- Context window management
- Conversation persistence
- Statistics and analytics
- Export functionality

## Configuration Architecture

### Configuration Sections
1. **Audio Configuration**: Recording parameters, silence detection
2. **Whisper Configuration**: STT model settings and parameters
3. **Ollama Configuration**: LLM model and generation settings
4. **TTS Configuration**: Speech synthesis parameters
5. **Conversation Configuration**: Memory and context management
6. **Development Configuration**: Debug and testing settings

### Environment Variable Support
- `OLLAMA_MODEL`: Override default LLM model
- `OLLAMA_BASE_URL`: Custom Ollama server URL
- `WHISPER_MODEL_SIZE`: Whisper model size selection
- `DEBUG`: Enable debug mode

## Error Handling & Resilience

### Component-Level Error Handling
- **Audio Capture**: Graceful handling of device failures
- **STT Processing**: Fallback for transcription failures
- **LLM Generation**: Timeout and connection error handling
- **TTS Synthesis**: Voice engine error recovery

### System-Level Resilience
- Automatic component reinitialization
- Graceful degradation on partial failures
- User notification of system status
- Recovery mechanisms for common failure modes

## Performance Optimization

### Audio Processing
- Efficient audio buffering and chunking
- Silence detection to minimize processing overhead
- Audio level monitoring for user feedback
- Memory-efficient audio data handling

### Model Management
- Lazy loading of heavy models
- Model caching to avoid repeated downloads
- GPU utilization when available
- Memory management for large models

### Conversation Management
- Context window limiting to prevent memory bloat
- Efficient message storage and retrieval
- Conversation summarization for long histories
- Automatic cleanup of old conversations

## Security Considerations

### Local Processing
- All processing occurs locally (no data sent to external services)
- No API keys or credentials required
- Audio data not persisted unless explicitly configured
- Conversation data stored locally with user control

### Privacy Protection
- No cloud-based processing
- User controls over data retention
- Optional conversation encryption
- Clear data deletion mechanisms

## Testing Strategy

### Component Testing
- Unit tests for individual components
- Integration tests for component interactions
- Performance benchmarks for critical paths
- Error condition testing

### System Testing
- End-to-end voice interaction testing
- Stress testing with continuous usage
- Memory leak detection
- Cross-platform compatibility testing

### User Experience Testing
- Latency measurement and optimization
- Accuracy assessment across different speakers
- Usability testing for various use cases
- Accessibility considerations

## Deployment Considerations

### Platform Support
- **Windows**: Full support with audio device detection
- **macOS**: Native audio integration
- **Linux**: ALSA/PulseAudio compatibility

### Installation Requirements
- Python environment setup
- System dependencies (FFmpeg, audio drivers)
- Ollama server installation and model download
- Voice synthesis engine setup

### Resource Requirements
- Minimum 4GB RAM for basic operation
- 8GB+ RAM recommended for larger models
- 2GB+ storage for models and audio files
- Multi-core CPU for optimal performance

## Future Enhancements

### Planned Features
- Multi-language conversation support
- Custom voice training and cloning
- Advanced conversation analytics
- Integration with external APIs (optional)
- Mobile application support

### Technical Improvements
- WebRTC integration for web-based access
- Real-time streaming capabilities
- Advanced audio preprocessing
- Model quantization for reduced memory usage
- Distributed processing support

## Risk Assessment

### Technical Risks
- **Model Performance**: Accuracy and latency variations
- **Resource Usage**: Memory and CPU requirements
- **Platform Compatibility**: Audio device differences
- **Dependency Stability**: Third-party library updates

### Mitigation Strategies
- Comprehensive testing across platforms
- Performance monitoring and optimization
- Fallback mechanisms for critical components
- Regular dependency updates and compatibility checks

## Conclusion

The AI Audio Agent provides a robust, offline-capable voice assistant solution with strong technical foundations. The modular architecture allows for easy customization and extension, while the local processing approach ensures privacy and reliability. The system is well-suited for deployment in environments where internet connectivity is limited or privacy is paramount. 