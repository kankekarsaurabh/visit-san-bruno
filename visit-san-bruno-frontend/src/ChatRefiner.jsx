import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import "./ChatRefiner.css";

// --- Configuration ---
// Best practice: Move hardcoded URLs to a single config spot.
const API_BASE_URL = "https://spandatest-a67a23991faf.herokuapp.com";


// --- Helper Functions ---
// Generates a unique session ID.
const generateSessionId = () => `session_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;

export default function ChatRefiner() {
// --- State Management ---
const [messages, setMessages] = useState([]);
const [input, setInput] = useState("");
// Use a lazy initializer for the session ID so it's only generated once.
const [sessionId, setSessionId] = useState(generateSessionId);
const [isTyping, setIsTyping] = useState(false);
const chatBoxRef = useRef(null);

// --- Effects ---
// Automatically scroll to the bottom of the chat when new messages are added.
useEffect(() => {
if (chatBoxRef.current) {
    chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
}
}, [messages, isTyping]);

// --- Core Functions ---
// Resets the chat to its initial state with a new session ID.
const resetChat = () => {
setMessages([]);
setSessionId(generateSessionId());
};

// Handles sending the user's message to the backend.
const handleSend = async () => {
if (!input.trim() || isTyping) return;

const userMessage = { role: "user", content: input };
setMessages(prevMessages => [...prevMessages, userMessage]);
setInput("");
setIsTyping(true);

try {
    const res = await axios.post(`${API_BASE_URL}/chat`, {
    session_id: sessionId,
    message: input,
    });

    const reply = res.data?.reply || "⚠️ Sorry, I couldn't get a response.";
    const assistantMessage = { role: "assistant", content: reply };
    
    setMessages(prevMessages => [...prevMessages, assistantMessage]);

} catch (err) {
    console.error("❌ API Error:", err);
    const errorMessage = err.response?.data?.detail || "Error connecting to the assistant. Please try again.";
    
    setMessages(prevMessages => [
    ...prevMessages,
    { role: "assistant", content: `⚠️ ${errorMessage}` },
    ]);
} finally {
    setIsTyping(false);
}
};

return (
<div className="chat-wrapper">
    <h2 className="chat-title">Plan Your Itinerary</h2>

    <div className="chat-box" ref={chatBoxRef}>
    {messages.length === 0 && (
        <div className="chat-message chat-system">
        Ask me to plan something, like "sushi then karaoke".
        </div>
    )}
    {messages.map((msg, idx) => (
        <div
        key={idx}
        className={`chat-message ${msg.role === "user" ? "chat-user" : "chat-assistant"}`}
        >
        {/* Using a more robust method to render newlines */}
        {msg.content.split('\n').map((line, i) => <p key={i}>{line}</p>)}
        </div>
    ))}
    {isTyping && (
        <div className="chat-message chat-typing">Assistant is typing...</div>
    )}
    </div>

    <div className="chat-input-wrapper">
    <input
        type="text"
        className="chat-input"
        placeholder="e.g., coffee then a park"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && handleSend()}
        disabled={isTyping} // UX: Disable input while waiting for a response
    />
    {/* ✅ SYNTAX FIX: Corrected the closing tag */}
    <button onClick={handleSend} className="chat-send-btn" disabled={isTyping}>
        Send
    </button>
    <button
        onClick={resetChat}
        className="chat-send-btn"
        style={{ marginLeft: "10px", backgroundColor: "#6c757d" }}
        disabled={isTyping}
    >
        Reset
    </button>
    </div>
</div>
);
}