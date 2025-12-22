import React, { useState, useEffect, useRef, useCallback } from 'react';
import MicButton from './components/MicButton';
import ResponseBubble from './components/ResponseBubble';
import ChatHistory from './components/ChatHistory';
import { v4 as uuidv4 } from 'uuid';

// Define the backend URL in a constant for easy management
// const BACKEND_URL = "http://localhost:8000";
const WS_URL = "ws://localhost:8000";

// Audio configuration for PCM playback
const AUDIO_SAMPLE_RATE = 24000;
const AUDIO_CHANNELS = 1;

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [interimTranscript, setInterimTranscript] = useState('');
  const [finalTranscript, setFinalTranscript] = useState('');
  const [streamingResponse, setStreamingResponse] = useState('');
  const [response, setResponse] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isAssistantSpeaking, setIsAssistantSpeaking] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('initializing');

  // Accumulate LLM tokens but only display when speaking
  const pendingResponseRef = useRef('');

  const websocketRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const sessionIdRef = useRef(null);
  const audioContextRef = useRef(null);
  const audioQueueRef = useRef([]);
  const isPlayingRef = useRef(false);
  const nextPlayTimeRef = useRef(0);

  useEffect(() => {
    sessionIdRef.current = uuidv4();
    setConnectionStatus('ready');
    console.log('Client session ID created:', sessionIdRef.current);

    // Initialize AudioContext for streaming playback
    audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: AUDIO_SAMPLE_RATE
    });

    return () => {
      cleanup();
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  // Process and play audio queue
  const playNextAudioChunk = useCallback(async () => {
    if (isPlayingRef.current || audioQueueRef.current.length === 0) {
      return;
    }

    isPlayingRef.current = true;

    while (audioQueueRef.current.length > 0) {
      const audioBase64 = audioQueueRef.current.shift();

      try {
        // Decode base64 to ArrayBuffer
        const binaryString = atob(audioBase64);
        const len = binaryString.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }

        // Convert Int16 PCM to Float32 for Web Audio API
        const pcmData = new Int16Array(bytes.buffer);
        const floatData = new Float32Array(pcmData.length);
        for (let i = 0; i < pcmData.length; i++) {
          floatData[i] = pcmData[i] / 32768.0;
        }

        // Create audio buffer
        const audioBuffer = audioContextRef.current.createBuffer(
          AUDIO_CHANNELS,
          floatData.length,
          AUDIO_SAMPLE_RATE
        );
        audioBuffer.getChannelData(0).set(floatData);

        // Schedule playback
        const source = audioContextRef.current.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContextRef.current.destination);

        const currentTime = audioContextRef.current.currentTime;
        const startTime = Math.max(currentTime, nextPlayTimeRef.current);
        source.start(startTime);
        nextPlayTimeRef.current = startTime + audioBuffer.duration;

        console.log(`Playing audio chunk: ${floatData.length} samples, duration: ${audioBuffer.duration.toFixed(2)}s`);
      } catch (error) {
        console.error('Error playing audio chunk:', error);
      }
    }

    isPlayingRef.current = false;
  }, []);

  const initializeWebSocket = () => {
    if (!sessionIdRef.current) {
      console.error("Session ID not initialized.");
      setConnectionStatus('error');
      return;
    }

    if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
      return;
    }

    const wsUrl = `${WS_URL}/ws/audio/${sessionIdRef.current}`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnectionStatus('connected');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'recording_started':
          console.log('Recording started on server');
          break;

        case 'stt_ready':
          console.log('STT stream ready');
          break;

        case 'speech_started':
          console.log('Speech detected');
          break;

        case 'interim_transcript':
          // Update interim transcript display
          setInterimTranscript(data.text);
          if (data.is_final) {
            console.log('Final segment:', data.text);
          }
          break;

        case 'final_transcript':
          // User finished speaking
          setFinalTranscript(data.text);
          setInterimTranscript('');
          setIsProcessing(true);
          setIsStreaming(true);
          setStreamingResponse('');
          console.log('Final transcript:', data.text);
          break;

        case 'llm_token':
          // Show text immediately as it streams
          setStreamingResponse(prev => prev + data.text);
          break;

        case 'llm_complete':
          // LLM response complete
          console.log('LLM response complete');
          break;

        case 'playback_started':
          console.log('Assistant speaking started');
          setIsAssistantSpeaking(true);
          break;

        case 'playback_finished':
          console.log('Assistant speaking finished');
          setIsAssistantSpeaking(false);
          // Finalize response when playback completes
          finalizeResponse();
          break;

        case 'audio_chunk':
          // Queue audio chunk for playback
          audioQueueRef.current.push(data.audio);

          // Resume audio context if suspended (browser autoplay policy)
          if (audioContextRef.current.state === 'suspended') {
            audioContextRef.current.resume();
          }

          // Start playing
          playNextAudioChunk();
          break;

        case 'recording_stopped':
          console.log('Recording stopped');
          break;

        case 'error':
          console.error('Server error:', data.message);
          setIsProcessing(false);
          setIsStreaming(false);
          setConnectionStatus('error');
          break;

        default:
          console.log('Unknown message type:', data.type);
          break;
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionStatus('error');
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      if (connectionStatus !== 'error') {
        setConnectionStatus('ready');
      }

      // Finalize the response when connection closes
      if (streamingResponse || finalTranscript) {
        finalizeResponse();
      }
    };

    websocketRef.current = ws;
  };

  const finalizeResponse = () => {
    setStreamingResponse(prev => {
      if (prev) {
        const newChatItem = {
          id: uuidv4(),
          userMessage: finalTranscript,
          assistantMessage: prev,
          timestamp: new Date().toISOString()
        };

        setChatHistory(history => [...history, newChatItem]);

        setResponse({
          transcription: finalTranscript,
          text: prev
        });
      }
      return '';
    });

    setIsProcessing(false);
    setIsStreaming(false);
    setFinalTranscript('');
  };

  const handleRecordingStart = async () => {
    try {
      if (connectionStatus === 'error') {
        alert('Connection error. Please reconnect.');
        return;
      }

      setIsRecording(true);
      setResponse(null);
      setInterimTranscript('');
      setFinalTranscript('');
      setStreamingResponse('');

      // Reset audio playback state
      audioQueueRef.current = [];
      nextPlayTimeRef.current = 0;
      pendingResponseRef.current = '';  // Clear pending response

      // Generate new session ID for each conversation
      sessionIdRef.current = uuidv4();

      initializeWebSocket();

      await new Promise((resolve, reject) => {
        const checkConnection = () => {
          if (websocketRef.current?.readyState === WebSocket.OPEN) {
            resolve();
          } else if (websocketRef.current?.readyState === WebSocket.CLOSING || websocketRef.current?.readyState === WebSocket.CLOSED) {
            reject(new Error('WebSocket connection failed to open'));
          } else {
            setTimeout(checkConnection, 100);
          }
        };
        checkConnection();
      });

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000,
        }
      });

      const mimeType = 'audio/webm;codecs=opus';
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        alert('Your browser does not support the required audio format.');
        throw new Error('Unsupported mimeType');
      }

      const mediaRecorder = new MediaRecorder(stream, { mimeType });

      mediaRecorder.onstop = () => {
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.ondataavailable = async (event) => {
        if (event.data.size > 0 && websocketRef.current?.readyState === WebSocket.OPEN) {
          try {
            const arrayBuffer = await event.data.arrayBuffer();
            const base64Data = arrayBufferToBase64(arrayBuffer);
            websocketRef.current.send(JSON.stringify({
              type: 'audio_chunk',
              audio_data: base64Data
            }));
          } catch (error) {
            console.error("Error processing audio chunk:", error);
          }
        }
      };

      mediaRecorderRef.current = mediaRecorder;

      websocketRef.current.send(JSON.stringify({
        type: 'start_recording'
      }));

      // Send audio chunks more frequently for lower latency
      mediaRecorder.start(100);

    } catch (error) {
      console.error('Failed to start recording:', error);
      alert('Failed to start recording: ' + error.message);
      setIsRecording(false);
      setConnectionStatus('error');
    }
  };

  const handleRecordingStop = async () => {
    setIsRecording(false);

    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }

    setTimeout(() => {
      if (websocketRef.current?.readyState === WebSocket.OPEN) {
        websocketRef.current.send(JSON.stringify({
          type: 'stop_recording'
        }));
      }
    }, 100);
  };

  const handleReconnect = () => {
    cleanup();
    sessionIdRef.current = uuidv4();
    setConnectionStatus('ready');
    setResponse(null);
    setInterimTranscript('');
    setFinalTranscript('');
    setStreamingResponse('');
    setIsProcessing(false);
    setIsStreaming(false);
    console.log('Reconnected with new session ID:', sessionIdRef.current);
  };

  const cleanup = () => {
    if (websocketRef.current) {
      websocketRef.current.close();
      websocketRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current = null;
    }
    audioQueueRef.current = [];
  };

  const arrayBufferToBase64 = (buffer) => {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return window.btoa(binary);
  };

  const getConnectionStatusText = () => {
    switch (connectionStatus) {
      case 'ready': return 'Ready';
      case 'connected': return 'Connected';
      case 'error': return 'Connection Error';
      default: return 'Initializing...';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-blue-900 flex flex-col items-center justify-center p-4 font-sans">
      <div className="max-w-2xl w-full space-y-6">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-white mb-2">Voice Assistant</h1>
          <p className="text-white/80 mb-2">Click the mic, speak naturally, and get an instant voice response.</p>
          <div className="flex items-center justify-center space-x-2 text-sm text-white/60">
            <span>Status: {getConnectionStatusText()}</span>
            {isStreaming && <span className="text-green-400 animate-pulse">● Streaming</span>}
            {isAssistantSpeaking && <span className="text-orange-400 animate-pulse">🔊 Speaking</span>}
          </div>
        </div>

        <ChatHistory chatHistory={chatHistory} />

        {/* Interim transcript - shows while user is speaking */}
        {isRecording && interimTranscript && (
          <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-4 text-white/90">
            <div className="flex items-start space-x-2">
              <span className="text-blue-400 text-sm">You:</span>
              <span className="italic">{interimTranscript}</span>
              <span className="animate-pulse">▋</span>
            </div>
          </div>
        )}

        {/* Live streaming response - shown during LLM streaming OR TTS playback */}
        {(isStreaming || isAssistantSpeaking) && streamingResponse && (
          <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-4">
            <div className="flex items-start space-x-2">
              <span className="text-purple-400 text-sm">
                {isAssistantSpeaking ? '🔊 Assistant:' : 'Assistant:'}
              </span>
              <span className="text-white">{streamingResponse}</span>
              <span className="animate-pulse text-purple-400">▋</span>
            </div>
          </div>
        )}

        {/* Final response (when not streaming and not speaking) */}
        {response && !isStreaming && !isAssistantSpeaking && <ResponseBubble response={response} />}

        {isProcessing && !streamingResponse && (
          <div className="flex justify-center">
            <div className="bg-white/10 backdrop-blur-sm rounded-full px-6 py-3 text-white">
              <div className="flex items-center space-x-3">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                <span>🧠 Thinking...</span>
              </div>
            </div>
          </div>
        )}

        <div className="flex justify-center pt-4">
          <MicButton
            isRecording={isRecording}
            onRecordingStart={handleRecordingStart}
            onRecordingStop={handleRecordingStop}
            disabled={isProcessing || isAssistantSpeaking || (connectionStatus === 'error' && !isRecording)}
          />
        </div>

        {connectionStatus === 'error' && (
          <div className="text-center">
            <button
              onClick={handleReconnect}
              className="bg-red-500 hover:bg-red-600 text-white font-semibold px-4 py-2 rounded-lg transition-colors"
            >
              Reconnect
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;