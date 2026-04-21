(() => {
  // ── Storage helpers ────────────────────────────────────────────────────
  const STORAGE_KEY = "anchor_chats";

  function loadChats() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || []; }
    catch { return []; }
  }

  function saveChats(chats) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
  }

  function uuid() {
    if (typeof crypto !== "undefined" && crypto.randomUUID) return crypto.randomUUID();
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, c => {
      const r = Math.random() * 16 | 0;
      return (c === "x" ? r : (r & 0x3 | 0x8)).toString(16);
    });
  }

  function newChat(modelId) {
    return {
      id: uuid(),
      title: "New chat",
      model: modelId,
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
  }

  // ── State ──────────────────────────────────────────────────────────────
  let chats = loadChats();
  let activeChatId = null;
  let isStreaming = false;

  function activeChat() { return chats.find(c => c.id === activeChatId); }

  // ── DOM refs ───────────────────────────────────────────────────────────
  const sidebarEl    = document.getElementById("sidebar");
  const chatListEl   = document.getElementById("chat-list");
  const messagesEl   = document.getElementById("messages");
  const inputEl      = document.getElementById("input");
  const sendBtn      = document.getElementById("send-btn");
  const modelSel     = document.getElementById("model-select");
  const newChatBtn   = document.getElementById("new-chat-btn");
  const toggleBtn    = document.getElementById("sidebar-toggle");

  // ── Sidebar toggle ─────────────────────────────────────────────────────
  const backdropEl = document.getElementById("sidebar-backdrop");

  function closeSidebar() { sidebarEl.classList.add("collapsed"); }
  function toggleSidebar() { sidebarEl.classList.toggle("collapsed"); }

  // Collapse by default on mobile
  if (window.innerWidth <= 600) closeSidebar();

  const closeBtn = document.getElementById("sidebar-close");

  toggleBtn.addEventListener("click", toggleSidebar);
  backdropEl.addEventListener("click", closeSidebar);
  closeBtn.addEventListener("click", closeSidebar);

  // ── Load models ────────────────────────────────────────────────────────
  async function loadModels() {
    sendBtn.disabled = true;
    inputEl.disabled = true;

    try {
      const res = await fetch("/api/models");
      const models = await res.json();
      models.forEach(m => {
        const opt = document.createElement("option");
        opt.value = m.id;
        opt.textContent = m.name;
        modelSel.appendChild(opt);
      });
    } catch {
      const opt = document.createElement("option");
      opt.textContent = "Error loading models";
      opt.disabled = true;
      modelSel.appendChild(opt);
    }

    // Update the active chat's model now that options are loaded
    const chat = activeChat();
    if (chat && !chat.model && modelSel.value) {
      chat.model = modelSel.value;
      saveChats(chats);
    }

    if (!isStreaming) {
      sendBtn.disabled = false;
      inputEl.disabled = false;
    }
  }

  // ── Model change → new chat ────────────────────────────────────────────
  modelSel.addEventListener("change", () => createNewChat());

  // ── New chat ───────────────────────────────────────────────────────────
  newChatBtn.addEventListener("click", () => createNewChat());

  function createNewChat() {
    const chat = newChat(modelSel.value);
    chats.unshift(chat);
    saveChats(chats);
    selectChat(chat.id);
  }

  // ── Select chat ────────────────────────────────────────────────────────
  function selectChat(id) {
    activeChatId = id;
    const chat = activeChat();
    if (!chat) return;

    // Sync model selector
    if (modelSel.value !== chat.model) modelSel.value = chat.model;

    renderMessages();
    renderChatList();
  }

  // ── Render sidebar chat list ───────────────────────────────────────────
  function renderChatList() {
    chatListEl.innerHTML = "";

    const now = Date.now();
    const DAY = 86400000;

    const groups = [
      { label: "Today",     filter: c => now - c.updatedAt < DAY },
      { label: "Yesterday", filter: c => now - c.updatedAt >= DAY && now - c.updatedAt < 2 * DAY },
      { label: "Older",     filter: c => now - c.updatedAt >= 2 * DAY },
    ];

    groups.forEach(({ label, filter }) => {
      const group = chats.filter(filter);
      if (!group.length) return;

      const labelEl = document.createElement("div");
      labelEl.className = "chat-group-label";
      labelEl.textContent = label;
      chatListEl.appendChild(labelEl);

      group.forEach(chat => {
        const item = document.createElement("div");
        item.className = "chat-item" + (chat.id === activeChatId ? " active" : "");
        item.innerHTML = `
          <svg class="chat-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
          </svg>
          <span class="chat-item-title">${escapeHtml(chat.title)}</span>
        `;
        item.addEventListener("click", () => selectChat(chat.id));
        chatListEl.appendChild(item);
      });
    });
  }

  // ── Render messages ────────────────────────────────────────────────────
  function renderMessages() {
    messagesEl.innerHTML = "";
    const chat = activeChat();
    if (!chat || !chat.messages.length) {
      showEmptyState();
      return;
    }
    chat.messages.forEach(m => appendBubble(m.role, m.content));
    scrollToBottom();
  }

  function showEmptyState() {
    const div = document.createElement("div");
    div.className = "empty-state";
    div.innerHTML = `
      <div class="welcome-icon">⚓</div>
      <h2>Hey, I'm Anchor</h2>
      <p class="welcome-sub">We're building a private, on-device mental wellness companion — an AI that lives on your phone and never sends your conversations anywhere. This is an early version we're using to gather feedback before we get there. Try it out and let us know what you think.</p>
    `;
    messagesEl.appendChild(div);
  }

  function appendBubble(role, text) {
    document.querySelector(".empty-state")?.remove();
    const msg = document.createElement("div");
    msg.className = `msg ${role}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    msg.appendChild(bubble);
    messagesEl.appendChild(msg);
    return bubble;
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // ── Send message ───────────────────────────────────────────────────────
  async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text || isStreaming) return;

    const chat = activeChat();
    if (!chat) return;

    // Update title from first user message
    if (!chat.messages.length) {
      chat.title = text.length > 36 ? text.slice(0, 36) + "…" : text;
    }

    chat.messages.push({ role: "user", content: text });
    chat.updatedAt = Date.now();
    saveChats(chats);

    appendBubble("user", text);
    inputEl.value = "";
    inputEl.style.height = "auto";
    renderChatList();
    setStreaming(true);

    const botBubble = appendBubble("bot", "");
    botBubble.classList.add("streaming");
    scrollToBottom();

    let accumulated = "";

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: chat.id,
          model_id: chat.model,
          messages: chat.messages,
        }),
      });

      if (!response.ok) throw new Error(`Server error ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const payload = line.slice(6).trim();
          if (!payload) continue;
          let parsed;
          try { parsed = JSON.parse(payload); } catch { continue; }

          if (parsed.error) { botBubble.textContent = "Something went wrong. Please try again."; break; }
          if (parsed.token) { accumulated += parsed.token; botBubble.textContent = accumulated; scrollToBottom(); }
          if (parsed.done) break;
        }
      }
    } catch (err) {
      botBubble.textContent = "Connection error. Please try again.";
      console.error(err);
    } finally {
      botBubble.classList.remove("streaming");
      if (accumulated) {
        chat.messages.push({ role: "assistant", content: accumulated });
        chat.updatedAt = Date.now();
        saveChats(chats);
        if (!toastShown) {
          messagesSent++;
          if (messagesSent >= 3) {
            toastShown = true;
            feedbackToast.classList.add("show");
            setTimeout(() => feedbackToast.classList.remove("show"), 6000);
          }
        }
      }
      setStreaming(false);
    }
  }

  function setStreaming(val) {
    isStreaming = val;
    sendBtn.disabled = val;
    inputEl.disabled = val;
  }

  // ── Auto-grow textarea ─────────────────────────────────────────────────
  inputEl.addEventListener("input", () => {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
  });

  inputEl.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });

  sendBtn.addEventListener("click", sendMessage);

  // ── Contact modal ──────────────────────────────────────────────────────
  const contactModal = document.getElementById("contact-modal");
  const contactBtn   = document.getElementById("contact-btn");
  const contactClose = document.getElementById("contact-close");

  contactBtn.addEventListener("click", () => contactModal.classList.add("open"));
  contactClose.addEventListener("click", () => contactModal.classList.remove("open"));
  contactModal.addEventListener("click", e => { if (e.target === contactModal) contactModal.classList.remove("open"); });

  // ── Feedback pulse (after 30s) ─────────────────────────────────────────
  const feedbackBtn = document.getElementById("feedback-btn");
  setTimeout(() => {
    feedbackBtn.classList.add("pulsing");
    feedbackBtn.addEventListener("animationend", () => feedbackBtn.classList.remove("pulsing"), { once: true });
  }, 30000);

  // ── Feedback toast (after 3rd message sent) ────────────────────────────
  const feedbackToast = document.getElementById("feedback-toast");
  let messagesSent = 0;
  let toastShown = false;

  // ── Init ───────────────────────────────────────────────────────────────
  createNewChat();   // synchronous — always starts with a fresh chat
  loadModels();      // async — populates model dropdown in background
})();
