import React from 'react'

const ChatHistory = ({ chatHistory }) => {
  if (chatHistory.length === 0) {
    return null
  }

  return (
    <div className="space-y-4 max-h-96 overflow-y-auto">
      {chatHistory.map((chat) => (
        <div key={chat.id} className="space-y-3">
          {/* User Message */}
          <div className="flex justify-end">
            <div className="bg-blue-500 text-white rounded-2xl px-4 py-2 max-w-xs">
              <p className="text-sm">{chat.userMessage}</p>
            </div>
          </div>

          {/* Assistant Message */}
          <div className="flex justify-start">
            <div className="bg-white rounded-2xl px-4 py-2 max-w-xs shadow-sm">
              <p className="text-sm text-gray-800">{chat.assistantMessage}</p>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

export default ChatHistory 