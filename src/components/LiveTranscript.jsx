import React, { useEffect, useState } from 'react'

const LiveTranscript = ({ transcript, onTranscriptUpdate }) => {
  const [isListening, setIsListening] = useState(false)

  useEffect(() => {
    if (!transcript) return

    // Simulate real-time transcription updates
    // In a real implementation, this would come from WebRTC audio processing
    const words = transcript.split(' ')
    let currentIndex = 0

    const interval = setInterval(() => {
      if (currentIndex < words.length) {
        const partialTranscript = words.slice(0, currentIndex + 1).join(' ')
        onTranscriptUpdate(partialTranscript)
        currentIndex++
      } else {
        clearInterval(interval)
      }
    }, 100)

    return () => clearInterval(interval)
  }, [transcript, onTranscriptUpdate])

  return (
    <div className="flex justify-center">
      <div className="bg-white/20 backdrop-blur-sm rounded-2xl px-6 py-4 max-w-lg w-full">
        <div className="flex items-center space-x-2 mb-2">
          <div className="flex space-x-1">
            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
          </div>
          <span className="text-white/80 text-sm font-medium">Listening...</span>
        </div>
        
        <div className="text-white text-lg leading-relaxed">
          {transcript || (
            <span className="text-white/60 italic">
              Start speaking...
            </span>
          )}
        </div>
        
        {/* Cursor effect */}
        {transcript && (
          <span className="inline-block w-0.5 h-6 bg-white animate-pulse ml-1"></span>
        )}
      </div>
    </div>
  )
}

export default LiveTranscript 