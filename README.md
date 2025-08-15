
# AI Audio Agent

An offline, real-time AI assistant that can listen to your voice, understand your questions, and respond verbally—without relying on any paid APIs. The system uses open-source models and runs entirely locally.

## 🚀 Features

- **Voice Input**: Real-time microphone capture
- **Speech-to-Text**: OpenAI Whisper (offline)
- **AI Reasoning**: Local LLM (Ollama) with semantic search
- **Text-to-Speech**: pyttsx3 (offline)
- **Context Memory**: Keeps track of conversation history
- **Vector Database**: ChromaDB with semantic search for accurate responses
- **No Paid APIs**: Entirely open-source and locally run

## 🛠️ Prerequisites

Before running the AI Audio Agent, make sure you have the following installed:

### 1. Python 3.8+
```bash
python --version
```

### 2. FFmpeg
```bash
# Windows (using winget)
winget install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

### 3. Ollama
```bash
# Download from: https://ollama.ai/
# Then install a model:
ollama pull llama3.2:3b
```

## 📦 Installation

1. **Clone or download this project**

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python test_installation.py
```

4. **Test vector database functionality:**
```bash
python test_vector_db.py
```

## 🎯 Usage

Start the AI Audio Agent:

```bash
python main.py
```

### How to Use

1. **Start the application** - Run `python main.py`
2. **Press Enter** - The agent will start listening for your voice
3. **First interaction** - The AI will greet you with "Hi, how can I help you?"
4. **Speak your question** - Ask anything about ox4labs services
5. **Listen to the response** - The AI will respond verbally using semantic search
6. **Ask follow-up questions** - The conversation continues naturally
7. **Press 'q' to quit** - When you're done

### Voice Commands

- **Press Enter**: Start listening for voice input
- **Press 'q'**: Quit the application
- **Press 'h'**: Show help
- **Press 'i'**: Show system information
- **Press 'c'**: Clear conversation history
- **Press 'v'**: View conversation history
- **Press 'x'**: Save current conversation
- **Press 's'**: Search knowledge base directly
- **Press 'u'**: Update knowledge base (when JSON content changes)
- **Press 'k'**: Show knowledge base statistics
- **Press 'y'**: Show system status

## 🔧 Configuration

All settings can be modified in `config.py`:

- **Audio settings**: Sample rate, channels, silence detection
- **Whisper settings**: Model size, language, confidence thresholds
- **LLM settings**: Model, temperature, system prompt
- **TTS settings**: Voice rate, volume, voice selection
- **Vector Database**: ChromaDB settings and embedding model configuration

## 🧠 Vector Database & Semantic Search

The system now uses a sophisticated vector database (ChromaDB) with semantic search capabilities:

### How It Works

1. **Content Chunking**: The JSON content is parsed into semantically meaningful chunks
2. **Embedding Generation**: Each chunk is converted to vector embeddings using sentence-transformers
3. **Vector Storage**: Embeddings are stored in ChromaDB for fast retrieval
4. **Semantic Search**: User queries are embedded and matched against stored vectors
5. **Context-Aware Responses**: The most relevant chunks are provided to the LLM for accurate responses

### Benefits

- **More Accurate**: Semantic search finds relevant information even with different wordings
- **Faster**: Vector similarity search is much faster than text matching
- **Scalable**: Easy to add new content without changing the system
- **Flexible**: Can handle variations in how users ask questions

### Updating Content

When you modify the `Ox4labs.json` file:

1. **Press 'u'** in the application to update the knowledge base
2. **Or restart** the application (it will auto-update if empty)
3. **The system** will automatically re-embed and store the new content

## 🏢 ox4labs Integration

The AI is configured to act as a representative of ox4labs, providing information about:

- **AI Consulting & Strategy**: AI roadmaps, opportunity mapping, strategic frameworks
- **AI Prototyping & Development**: AI-powered products, model training, AI agents
- **AI Training & Business Workshops**: Executive workshops, workforce upskilling, government training
- **Process**: Align AI with Business Goals, Co-Creation & Rapid Prototyping, Optimize, Scale & Govern

The system uses semantic search to provide accurate, context-aware responses based on the company's actual information.

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'whisper'"**
   - Run: `pip install -r requirements.txt`

2. **"FFmpeg not found"**
   - Install FFmpeg: `winget install ffmpeg`

3. **"Ollama connection failed"**
   - Start Ollama: `ollama serve`
   - Check if model is installed: `ollama list`

4. **"No audio devices found"**
   - Check microphone permissions
   - Verify audio drivers are installed

5. **"TTS run loop already started"**
   - This is a known pyttsx3 issue, the system will continue working

### Audio Device Issues

If you have audio device problems:

```bash
# Test audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### Performance Optimization

- **Use smaller Whisper model**: Change `model` in config to `"tiny"` or `"base"`
- **Reduce audio quality**: Lower sample rate in audio config
- **Use faster LLM**: Try smaller models like `llama3.2:1b`

## 📁 Project Structure

```
├── main.py                 # Main application entry point
├── audio_agent.py          # Core AI agent orchestration
├── speech_to_text.py       # Whisper STT implementation
├── text_to_speech.py       # pyttsx3 TTS implementation
├── llm_client.py           # Ollama LLM client
├── conversation.py         # Conversation memory management
├── config.py              # Configuration settings
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the MIT License.

---

**ox4labs AI Audio Agent** - Your intelligent voice assistant for all things ox4labs! 🚀 