const chatContainer = document.getElementById('chat-container');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const historyList = document.getElementById('history-list');
const micBtn = document.getElementById('mic-btn');

let messageHistory = [];
let currentSessionId = null;
let recognition = null;
let isVoiceActive = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadSessionHistory();
    startNewChat(); // Start a new session by default
    setupSpeechRecognition();
});

function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

async function startNewChat(skipAutoSave = false) {
    console.log("startNewChat called. skipAutoSave:", skipAutoSave);
    console.log("Current Session:", currentSessionId);
    console.log("History Length:", messageHistory.length);

    // Auto-Save Previous Session (Background)
    if (!skipAutoSave && currentSessionId && messageHistory.length > 0) {
        const hasUserMessages = messageHistory.some(m => m.role === 'user');
        console.log("Has user messages:", hasUserMessages);

        if (hasUserMessages) {
            console.log("Auto-analyzing session:", currentSessionId);
            // Fire and forget
            fetch('/end_session', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: currentSessionId })
            })
                .then(res => res.json())
                .then(data => console.log("Auto-save response:", data))
                .catch(e => console.error("Auto-save failed:", e));
        } else {
            console.log("Skipping auto-save: No user messages.");
        }
    } else {
        console.log("Skipping auto-save: Condition failed.");
    }

    currentSessionId = generateSessionId();
    messageHistory = [];
    chatContainer.innerHTML = ''; // Start empty
    updateActiveHistoryItem(null);

    try {
        // Check if we need to run onboarding
        // Add timestamp to prevent caching
        const res = await fetch('/onboarding_status?t=' + Date.now());
        const data = await res.json();

        if (data.onboarding_required) {
            // Trigger onboarding greeting immediately
            // We send an empty message to trigger Step 0 in backend
            // Add Loading Indicator
            const loadingId = addLoadingIndicator();

            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    messages: [], // Empty messages triggers step 0
                    sessionId: currentSessionId
                })
            });

            const chatData = await response.json();
            removeLoadingIndicator(loadingId);

            if (chatData.content) {
                addMessageToUI('assistant', chatData.content);
                messageHistory.push({ role: 'assistant', content: chatData.content });
            }
        } else {
            // Normal Welcome - Dynamic Greeting
            // We send an empty message to trigger the greeting logic in app.py

            // Add Loading Indicator
            const loadingId = addLoadingIndicator();

            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    messages: [], // Empty messages triggers greeting
                    sessionId: currentSessionId
                })
            });

            const chatData = await response.json();
            removeLoadingIndicator(loadingId);

            if (chatData.content) {
                addMessageToUI('assistant', chatData.content);
                messageHistory.push({ role: 'assistant', content: chatData.content });
            }
        }
    } catch (e) {
        console.error("Error starting chat:", e);
        // Fallback
        chatContainer.innerHTML = `
            <div class="message ai-message">
                <div class="message-content">
                    Hello! I'm Anchor. How can I help you today?
                </div>
            </div>
        `;
    }
}

async function loadSessionHistory() {
    try {
        const res = await fetch('/sessions');
        const sessions = await res.json();

        historyList.innerHTML = '';
        sessions.forEach(session => {
            const div = document.createElement('div');
            div.className = 'history-item';
            div.textContent = session.summary || 'New Chat';
            div.dataset.id = session.id;
            div.onclick = () => loadSession(session.id);
            historyList.appendChild(div);
        });
    } catch (e) {
        console.error("Failed to load history:", e);
    }
}

async function loadSession(sessionId) {
    try {
        const res = await fetch(`/sessions/${sessionId}`);
        const data = await res.json();

        currentSessionId = sessionId;
        messageHistory = data.messages || [];

        // Render Chat
        chatContainer.innerHTML = '';
        messageHistory.forEach(msg => {
            // Skip system prompt if present in UI rendering usually, but here we just render user/assistant
            if (msg.role === 'system') return;
            addMessageToUI(msg.role, msg.content);
        });

        updateActiveHistoryItem(sessionId);

    } catch (e) {
        console.error("Failed to load session:", e);
    }
}

function updateActiveHistoryItem(sessionId) {
    document.querySelectorAll('.history-item').forEach(el => {
        if (el.dataset.id === sessionId) el.classList.add('active');
        else el.classList.remove('active');
    });
}

function setupSpeechRecognition() {
    if ('webkitSpeechRecognition' in window) {
        recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onstart = () => {
            isVoiceActive = true;
            micBtn.classList.add('listening');
        };

        recognition.onend = () => {
            isVoiceActive = false;
            micBtn.classList.remove('listening');
        };

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            userInput.value = transcript;
            resizeTextarea();
            sendBtn.disabled = false;
            // Optional: Auto-send if desired, but safer to let user confirm
            // sendMessage(true); 
        };

        micBtn.addEventListener('click', () => {
            if (isVoiceActive) recognition.stop();
            else recognition.start();
        });
    } else {
        micBtn.style.display = 'none';
        console.log('Web Speech API not supported');
    }
}

function resizeTextarea() {
    userInput.style.height = 'auto';
    userInput.style.height = (userInput.scrollHeight) + 'px';
}

// Auto-resize textarea
userInput.addEventListener('input', function () {
    resizeTextarea();
    if (this.value.trim().length > 0) {
        sendBtn.disabled = false;
    } else {
        sendBtn.disabled = true;
    }
});

userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);

async function endSession() {
    if (!currentSessionId) return;

    const btn = document.getElementById('finish-btn');
    const originalText = btn.textContent;
    btn.textContent = "Analyzing...";
    btn.disabled = true;

    try {
        const response = await fetch('/end_session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: currentSessionId })
        });

        const data = await response.json();

        if (response.ok) {
            btn.textContent = "Saved";
            setTimeout(() => {
                btn.textContent = originalText;
                btn.disabled = false;
                startNewChat(true); // Skip duplicate auto-save since we just saved
            }, 2000);
        } else {
            alert("Error: " + data.error);
            btn.textContent = originalText;
            btn.disabled = false;
        }
    } catch (e) {
        console.error(e);
        btn.textContent = originalText;
        btn.disabled = false;
    }
}


async function resetMemory() {
    if (!confirm("Are you sure? This will delete all chat history and your user profile permanently.")) return;

    try {
        const response = await fetch('/reset_memory', { method: 'POST' });
        if (response.ok) {
            alert("Memory has been wiped.");
            location.reload();
        } else {
            alert("Failed to reset memory.");
        }
    } catch (e) {
        console.error(e);
        alert("Error resetting memory.");
    }
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    // Add User Message
    addMessageToUI('user', text);
    // Update local history for API call
    messageHistory.push({ role: 'user', content: text });

    userInput.value = '';
    userInput.style.height = 'auto';
    sendBtn.disabled = true;

    // Add Loading Indicator
    const loadingId = addLoadingIndicator();

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                messages: messageHistory,
                sessionId: currentSessionId
            })
        });

        const data = await response.json();

        // Remove Loading Indicator
        removeLoadingIndicator(loadingId);

        // Add AI Response
        addMessageToUI('assistant', data.content);

        // Update local history
        messageHistory.push({ role: 'assistant', content: data.content });

        // Reload history list to show update summary
        loadSessionHistory();

    } catch (error) {
        console.error('Error:', error);
        removeLoadingIndicator(loadingId);
        addMessageToUI('assistant', "I'm having trouble connecting right now. Please try again.");
    }
}

function speakText(text) {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel(); // Stop previous
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        // utterance.voice = ... (Choose a nice voice if available)
        window.speechSynthesis.speak(utterance);
    }
}

function addMessageToUI(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;

    // Parse Markdown for AI messages
    const formattedContent = role === 'assistant' ? marked.parse(content) : content;

    let roleHtml = '';
    if (role === 'assistant') {
        // Add Speaker Button
        // We use a simple onclick with the raw content (stripped of markdown ideally, but TTS handles some ok)
        // Let's strip simple markdown for TTS
        const cleanText = content.replace(/[*#_`]/g, '');
        // Escape quotes for the onclick attribute
        const safeText = cleanText.replace(/"/g, "&quot;").replace(/'/g, "\\'");

        roleHtml = `
            <button class="speaker-btn" onclick="speakText('${safeText}')" title="Read aloud">
                <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                    <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/>
                </svg>
            </button>
        `;
    }

    messageDiv.innerHTML = `
        <div class="message-content">
            ${formattedContent}
            ${roleHtml}
        </div>
    `;

    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addLoadingIndicator() {
    const id = 'loading-' + Date.now();
    const loadingDiv = document.createElement('div');
    loadingDiv.id = id;
    loadingDiv.className = 'message ai-message';
    loadingDiv.innerHTML = `
        <div class="typing">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    chatContainer.appendChild(loadingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return id;
}

function removeLoadingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

/* --- Quick Tools & Sidebar Logic --- */

function toggleSidebar(toolId) {
    const sidebar = document.getElementById('quick-sidebar');
    const title = document.getElementById('sidebar-title');

    // 1. Open Sidebar if closed
    sidebar.classList.add('open');

    // 2. Hide all widgets
    document.querySelectorAll('.widget').forEach(w => w.classList.remove('active'));

    // 3. Show selected widget
    const widget = document.getElementById(`widget-${toolId}`);
    if (widget) {
        widget.classList.add('active');
    }

    // 4. Set Title
    const titles = {
        'breathing': 'Breathing Bubble',
        'grounding': '5-4-3-2-1 Grounding',
        'crisis': 'Crisis Resources'
    };
    title.textContent = titles[toolId];

    // 5. Initialize specific tool logic
    if (toolId === 'breathing') resetBreathing();
    if (toolId === 'grounding') initGrounding();
}

function closeSidebar() {
    document.getElementById('quick-sidebar').classList.remove('open');
    stopBreathing(); // Ensure breathing stops
}

/* --- Breathing Tool --- */
let breathingInterval;
let breathState = 'idle'; // idle, inhale, hold, exhale

function toggleBreathing() {
    const btn = document.getElementById('breath-toggle');
    if (breathState === 'idle') {
        startBreathing();
        btn.textContent = "Stop";
    } else {
        stopBreathing();
        btn.textContent = "Start";
    }
}

function startBreathing() {
    const circle = document.querySelector('.breathing-circle');
    const text = document.getElementById('breath-text');

    breathState = 'inhale';
    runPropagate(circle, text);
}

function runPropagate(circle, text) {
    if (breathState === 'idle') return;

    // Inhale (4s)
    circle.className = 'breathing-circle inhale';
    text.textContent = "Inhale...";

    setTimeout(() => {
        if (breathState === 'idle') return;

        // Hold (7s)
        circle.className = 'breathing-circle hold';
        text.textContent = "Hold...";

        setTimeout(() => {
            if (breathState === 'idle') return;

            // Exhale (8s)
            circle.className = 'breathing-circle exhale';
            text.textContent = "Exhale...";

            setTimeout(() => {
                if (breathState === 'idle') return;
                runPropagate(circle, text); // Loop
            }, 8000);

        }, 7000);

    }, 4000);
}

function stopBreathing() {
    breathState = 'idle';
    resetBreathing();
}

function resetBreathing() {
    const circle = document.querySelector('.breathing-circle');
    const text = document.getElementById('breath-text');
    const btn = document.getElementById('breath-toggle');

    if (circle) circle.className = 'breathing-circle';
    if (text) text.textContent = "Ready";
    if (btn) btn.textContent = "Start";
    breathState = 'idle';
}

/* --- Grounding Tool --- */
let groundingStep = 5;

function initGrounding() {
    groundingStep = 5;
    renderGroundingStep();
}

function renderGroundingStep() {
    const container = document.getElementById('grounding-step-container');
    container.innerHTML = '';

    if (groundingStep === 0) {
        container.innerHTML = `
            <div class="grounding-step">
                <h4>Great job.</h4>
                <p>You are safe. You are here.</p>
            </div>
        `;
        document.querySelector('.grounding-controls').style.display = 'none';
        return;
    }

    document.querySelector('.grounding-controls').style.display = 'block';

    const prompts = {
        5: { text: "List 5 things you see", inputs: 5 },
        4: { text: "List 4 things you feel", inputs: 4 },
        3: { text: "List 3 things you hear", inputs: 3 },
        2: { text: "List 2 things you smell", inputs: 2 },
        1: { text: "List 1 thing you taste", inputs: 1 }
    };

    const current = prompts[groundingStep];

    let html = `
        <div class="grounding-step">
            <h4>${current.text}</h4>
            <div class="grounding-inputs">
    `;

    for (let i = 0; i < current.inputs; i++) {
        html += `<input type="text" placeholder="Item ${i + 1}">`;
    }

    html += `</div></div>`;
    container.innerHTML = html;
}

function nextGroundingStep() {
    groundingStep--;
    renderGroundingStep();
}
