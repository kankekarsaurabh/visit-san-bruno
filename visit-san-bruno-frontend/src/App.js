import React from 'react';
import './App.css';
import ChatRefiner from './ChatRefiner';

function App() {
    return (
        <div className="App">
        {/* Header */}
        <header className="bg-blue-600 text-white px-6 py-4 shadow-md">
            <h1 className="text-3xl font-bold">Visit San Bruno</h1>
            <p className="text-sm mt-1">Curated itinerary assistant powered by AI</p>
        </header>

        {/* Main Content */}
        <main className="p-6 bg-gray-50 min-h-screen">
            <div className="max-w-3xl mx-auto">
            <ChatRefiner initialPlan="visit museum then coffee then karaoke" />
            </div>
        </main>
        </div>
        
    );
    }

export default App;