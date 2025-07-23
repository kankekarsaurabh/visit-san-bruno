// ChatInterface.jsx
import React, { useState } from 'react';

    export default function ChatInterface() {
    const [messages, setMessages] = useState([
        { role: 'system', content: 'You are a helpful itinerary assistant.' }
    ]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);

    const sendMessage = async () => {
        if (!input.trim()) return;

        const updatedMessages = [...messages, { role: 'user', content: input }];
        setMessages(updatedMessages);
        setInput('');
        setLoading(true);

        try {
        const response = await fetch('http://localhost:8002/chat-refine', {
            method: 'POST',
            headers: {
            'Content-Type': 'application/json',
            },
            body: JSON.stringify({ messages: updatedMessages }),
        });

        const data = await response.json();
        setMessages([...updatedMessages, { role: 'assistant', content: data.reply }]);
        } catch (err) {
        setMessages([...updatedMessages, { role: 'assistant', content: '⚠️ Error contacting Gemma backend.' }]);
        } finally {
        setLoading(false);
        }
    };

    return (
        <div className="p-4 max-w-xl mx-auto">
        <div className="h-[500px] overflow-y-auto border rounded p-4 mb-4 bg-white shadow">
            {messages.filter(msg => msg.role !== 'system').map((msg, idx) => (
            <div
                key={idx}
                className={`mb-2 p-2 rounded ${msg.role === 'user' ? 'bg-blue-100 self-end' : 'bg-gray-100 self-start'}`}
            >
                <strong>{msg.role === 'user' ? 'You' : 'Assistant'}:</strong> {msg.content}
            </div>
            ))}
        </div>
        <div className="flex">
            <input
            type="text"
            className="flex-grow p-2 border rounded-l"
            placeholder="Ask about your itinerary..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
            />
            <button
            onClick={sendMessage}
            className="bg-blue-500 text-white px-4 rounded-r hover:bg-blue-600"
            disabled={loading}
            >
            {loading ? '...' : 'Send'}
            </button>
        </div>
        </div>
    );
}