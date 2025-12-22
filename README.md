# 🎤 Live Voice Assistant

A real-time, low-latency voice assistant with live streaming capabilities for natural conversation flow.

## ✨ Features

### 🚀 Live Streaming Pipeline
- **Real-time STT**: Word-level transcription with minimal latency
- **Live LLM Streaming**: Immediate response generation as you speak
- **Instant TTS**: Audio playback starts as soon as text is generated
- **Performance Monitoring**: Real-time latency metrics for each component

### 🎯 Low Latency Optimizations
- **100ms Audio Chunks**: Reduced from 400ms for faster processing
- **Optimized Audio Settings**: 48kHz sample rate, single channel
- **WebRTC-style Streaming**: Minimal buffering and delays
- **Parallel Processing**: STT, LLM, and TTS run concurrently

### 📊 Performance Dashboard
- **Live Metrics**: STT, LLM, and TTS latency tracking
- **Audio Visualization**: Real-time audio level display
- **Performance Status**: Excellent/Good/Slow indicators
- **Word-level Display**: Live transcription with typing effects

## 🛠️ Technology Stack

### Backend
- **FastAPI**: High-performance async web framework
- **Deepgram**: Real-time speech-to-text with Nova-2 model
- **Google Gemini**: Streaming LLM responses
- **WebSocket**: Real-time bidirectional communication

### Frontend
- **React**: Modern UI with real-time updates
- **Web Audio API**: Low-latency audio processing
- **Tailwind CSS**: Beautiful, responsive design
- **Canvas API**: Real-time audio visualization

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.8+
- Deepgram API key
- Google Gemini API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd voice_agent
   ```

2. **Install dependencies**
   ```bash
   # Frontend dependencies
   npm install
   
   # Backend dependencies
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   cp .env.example .env
   
   # Add your API keys
   DEEPGRAM_API_KEY=your_deepgram_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ```

4. **Start the application**
   ```bash
   chmod +x start.sh
   ./start.sh
   ```

5. **Open your browser**
   - Frontend: http://localhost:5173
   - Backend: http://localhost:8000

## 🎤 Usage

### Basic Conversation
1. **Click the microphone button** to start recording
2. **Speak naturally** - see live transcription appear
3. **Watch real-time responses** as the assistant processes
4. **Interrupt anytime** by clicking the button again

### Performance Monitoring
- **Green indicators**: Excellent performance (< 1s total latency)
- **Yellow indicators**: Good performance (1-2s total latency)
- **Red indicators**: Slow performance (> 2s total latency)

### Audio Visualization
- **Blue bars**: Recording mode
- **Yellow/Orange bars**: Processing mode
- **Bar height**: Audio level intensity

## 📊 Performance Metrics

### Target Latencies
- **STT**: < 500ms for word recognition
- **LLM**: < 1000ms for first response chunk
- **TTS**: < 500ms for audio generation
- **Total**: < 2000ms end-to-end

### Optimizations
- **Audio Chunk Size**: 100ms (vs 400ms standard)
- **Sample Rate**: 48kHz for high quality
- **Channels**: Mono for faster processing
- **Interim Results**: 100ms intervals
- **Utterance End**: 500ms detection

## 🔧 Configuration

### Backend Settings
```python
# Deepgram Options
sample_rate=48000
channels=1
interim_results_interval_ms=100
utterance_end_ms=500

# TTS Settings
sample_rate=24000  # Lower for faster processing
channels=1
```

### Frontend Settings
```javascript
// Audio Recording
audioBitsPerSecond: 128000
mimeType: 'audio/webm;codecs=opus'

// Audio Playback
sampleRate: 24000  // Match backend
```

## 🏗️ Architecture

### Real-time Pipeline
```
User Speech → Audio Chunks → Deepgram STT → Interim Results
                                    ↓
                              Final Transcript → Gemini LLM
                                    ↓
                              Text Chunks → Deepgram TTS
                                    ↓
                              Audio Chunks → Web Audio API
```

### Performance Flow
1. **Audio Capture**: 100ms chunks sent immediately
2. **STT Processing**: Word-level streaming with interim results
3. **LLM Generation**: Streaming responses as tokens arrive
4. **TTS Conversion**: Immediate audio generation per text chunk
5. **Audio Playback**: Low-latency Web Audio API rendering

## 🐛 Troubleshooting

### Common Issues

**High Latency**
- Check internet connection speed
- Verify API key validity
- Monitor browser console for errors
- Try refreshing the page

**Audio Issues**
- Ensure microphone permissions
- Check browser audio settings
- Verify Web Audio API support
- Try different browsers (Chrome recommended)

**Connection Errors**
- Verify backend is running on port 8000
- Check firewall settings
- Ensure WebSocket support
- Review browser console logs

### Debug Mode
```bash
# Backend with verbose logging
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level debug

# Frontend with development tools
npm run dev -- --debug
```

## 📈 Performance Tips

### For Best Results
1. **Use Chrome/Edge**: Best Web Audio API support
2. **Stable Internet**: Low latency connection
3. **Quiet Environment**: Reduce audio noise
4. **Clear Speech**: Speak naturally but clearly
5. **Close Other Tabs**: Reduce browser resource usage

### Monitoring
- Watch the performance dashboard
- Monitor latency indicators
- Check audio visualization
- Review browser console for errors

## 🔮 Future Enhancements

### Planned Features
- **Multi-language Support**: Real-time language detection
- **Voice Cloning**: Custom voice synthesis
- **Emotion Detection**: Sentiment-aware responses
- **Context Memory**: Conversation history
- **Custom Models**: Fine-tuned LLM integration

### Performance Improvements
- **WebRTC Integration**: Direct peer-to-peer audio
- **WebAssembly**: Native audio processing
- **Edge Computing**: Distributed processing
- **Caching**: Response optimization
- **Compression**: Reduced bandwidth usage

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review browser console logs
- Verify API key configurations
- Test with different browsers

---

**Built with ❤️ for real-time voice interaction** 