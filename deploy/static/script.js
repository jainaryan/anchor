(() => {
  // ── Storage keys ───────────────────────────────────────────────────────
  const CHATS_KEY   = "anchor_chats";
  const PROFILE_KEY = "anchor_profile";

  // ── Storage helpers ────────────────────────────────────────────────────
  function loadChats() {
    try { return JSON.parse(localStorage.getItem(CHATS_KEY)) || []; }
    catch { return []; }
  }
  function saveChats(chats) {
    localStorage.setItem(CHATS_KEY, JSON.stringify(chats));
  }

  function loadProfile() {
    try { return JSON.parse(localStorage.getItem(PROFILE_KEY)) || {}; }
    catch { return {}; }
  }
  function saveProfile(profile) {
    localStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
  }

  function uuid() {
    if (typeof crypto !== "undefined" && crypto.randomUUID) return crypto.randomUUID();
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, c => {
      const r = Math.random() * 16 | 0;
      return (c === "x" ? r : (r & 0x3 | 0x8)).toString(16);
    });
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function newChat(modelId) {
    return { id: uuid(), title: "New chat", model: modelId, messages: [], createdAt: Date.now(), updatedAt: Date.now() };
  }

  // ── State ──────────────────────────────────────────────────────────────
  let chats = loadChats();
  let activeChatId = null;
  let isStreaming = false;
  let activeTab = "chat";

  function activeChat() { return chats.find(c => c.id === activeChatId); }

  // ── DOM refs ───────────────────────────────────────────────────────────
  const sidebarEl  = document.getElementById("sidebar");
  const chatListEl = document.getElementById("chat-list");
  const messagesEl = document.getElementById("messages");
  const inputEl    = document.getElementById("input");
  const sendBtn    = document.getElementById("send-btn");
  const modelSel   = document.getElementById("model-select");
  const newChatBtn = document.getElementById("new-chat-btn");
  const toggleBtn  = document.getElementById("sidebar-toggle");
  const backdropEl = document.getElementById("sidebar-backdrop");
  const closeBtn   = document.getElementById("sidebar-close");

  // ── Tab switching ──────────────────────────────────────────────────────
  function switchTab(name) {
    activeTab = name;

    document.querySelectorAll(".tab-btn").forEach(btn => {
      const isActive = btn.dataset.tab === name;
      btn.classList.toggle("active", isActive);
      btn.setAttribute("aria-selected", isActive);
    });

    document.querySelectorAll(".tab-panel").forEach(panel => {
      panel.classList.toggle("active", panel.id === `panel-${name}`);
    });

    // Sidebar only makes sense on Chat tab
    if (name === "chat") {
      if (window.innerWidth > 600) sidebarEl.classList.remove("collapsed");
    } else {
      sidebarEl.classList.add("collapsed");
    }

    // Header controls relevant only on Chat tab
    const modelWrap = document.getElementById("model-select-wrap");
    modelWrap.style.display = name === "chat" ? "" : "none";

    if (name === "insights") renderInsights();
    if (name === "profile") loadProfileForm();
  }

  // Expose for inline onclick use in empty state
  window.switchTab = switchTab;

  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  // ── Sidebar toggle ─────────────────────────────────────────────────────
  function closeSidebar() { sidebarEl.classList.add("collapsed"); }
  function toggleSidebar() { sidebarEl.classList.toggle("collapsed"); }

  if (window.innerWidth <= 600) closeSidebar();

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
    const chat = activeChat();
    if (chat && !chat.model && modelSel.value) { chat.model = modelSel.value; saveChats(chats); }
    if (!isStreaming) { sendBtn.disabled = false; inputEl.disabled = false; }
  }

  modelSel.addEventListener("change", () => createNewChat());
  newChatBtn.addEventListener("click", () => createNewChat());

  function createNewChat() {
    const chat = newChat(modelSel.value);
    chats.unshift(chat);
    saveChats(chats);
    selectChat(chat.id);
  }

  function selectChat(id) {
    activeChatId = id;
    const chat = activeChat();
    if (!chat) return;
    if (modelSel.value !== chat.model) modelSel.value = chat.model;
    renderMessages();
    renderChatList();
  }

  // ── Sidebar chat list ──────────────────────────────────────────────────
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

  // ── Messages ───────────────────────────────────────────────────────────
  function renderMessages() {
    messagesEl.innerHTML = "";
    const chat = activeChat();
    if (!chat || !chat.messages.length) { showEmptyState(); return; }
    chat.messages.forEach(m => appendBubble(m.role, m.content));
    scrollToBottom();
  }

  function showEmptyState() {
    const profile = loadProfile();
    const hasProfile = !!(profile.name || profile.about);
    const greeting = profile.name ? `Hey, ${profile.name}` : "Hey, I'm Anchor";
    const div = document.createElement("div");
    div.className = "empty-state";
    div.innerHTML = `
      <div class="welcome-icon">⚓</div>
      <h2>${escapeHtml(greeting)}</h2>
      <p class="welcome-sub">We're building a private, on-device mental wellness companion — an AI that lives on your phone and never sends your conversations anywhere. This is an early version we're using to gather feedback. Try it out and let us know what you think.</p>
      ${!hasProfile ? `<button class="profile-nudge" onclick="window.switchTab('profile')">Add your profile for a personal experience →</button>` : ""}
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

  function scrollToBottom() { messagesEl.scrollTop = messagesEl.scrollHeight; }

  // ── Profile context builder ────────────────────────────────────────────
  function buildProfileContext(profile) {
    const parts = [];
    if (profile.name)   parts.push(`My name is ${profile.name}.`);
    if (profile.about)  parts.push(`About me: ${profile.about}`);
    if (profile.people) parts.push(`Key people in my life: ${profile.people}`);
    if (profile.helps)  parts.push(`Things that help me: ${profile.helps}`);
    if (profile.style) {
      const map = {
        vent: "I usually want to vent and be heard — please don't jump to advice unless I ask.",
        help: "I usually want practical strategies and concrete next steps.",
        both: "I want a mix — sometimes I need to be heard, sometimes I want advice. Read the situation.",
      };
      if (map[profile.style]) parts.push(map[profile.style]);
    }
    if (!parts.length) return null;
    return `[Personal context — keep this in mind throughout our conversation, but don't reference it robotically]\n${parts.join("\n")}`;
  }

  // ── Send message ───────────────────────────────────────────────────────
  async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text || isStreaming) return;

    const chat = activeChat();
    if (!chat) return;

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

    // Build API messages — prepend profile context if available
    let apiMessages = [...chat.messages];
    const profileCtx = buildProfileContext(loadProfile());
    if (profileCtx) {
      apiMessages = [
        { role: "user", content: profileCtx },
        { role: "assistant", content: "Got it, I'll keep this in mind." },
        ...apiMessages,
      ];
    }

    let accumulated = "";
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: chat.id, model_id: chat.model, messages: apiMessages }),
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

  inputEl.addEventListener("input", () => {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
  });
  inputEl.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  sendBtn.addEventListener("click", sendMessage);

  // ── Profile form ───────────────────────────────────────────────────────
  function loadProfileForm() {
    const profile = loadProfile();
    document.getElementById("profile-name").value  = profile.name   || "";
    document.getElementById("profile-about").value = profile.about  || "";
    document.getElementById("profile-helps").value = profile.helps  || "";
    document.getElementById("profile-people").value = profile.people || "";
    const style = profile.style || "";
    document.querySelectorAll("input[name='interaction-style']").forEach(radio => {
      radio.checked = radio.value === style;
    });
  }

  document.getElementById("profile-save-btn").addEventListener("click", () => {
    const style = document.querySelector("input[name='interaction-style']:checked")?.value || "";
    const profile = {
      name:   document.getElementById("profile-name").value.trim(),
      about:  document.getElementById("profile-about").value.trim(),
      helps:  document.getElementById("profile-helps").value.trim(),
      people: document.getElementById("profile-people").value.trim(),
      style,
    };
    saveProfile(profile);

    const msg = document.getElementById("profile-saved-msg");
    msg.textContent = "Saved";
    msg.classList.add("show");
    setTimeout(() => msg.classList.remove("show"), 2200);
  });

  // ── Insights ───────────────────────────────────────────────────────────
  function detectMood(text) {
    const t = text.toLowerCase();
    const heavy = ["anxious","anxiety","stressed","overwhelmed","depressed","sad","crying","scared","hopeless","worthless","panic","hurt","pain","lonely","alone","suicide","die","death","grief","loss","exhausted","hate myself","can't do","giving up"];
    const positive = ["happy","good","great","better","excited","grateful","thankful","joy","calm","peaceful","hopeful","proud","relieved","content","wonderful"];
    const hCount = heavy.filter(w => t.includes(w)).length;
    const pCount = positive.filter(w => t.includes(w)).length;
    if (hCount >= 3) return { cls: "heavy", label: "Heavy" };
    if (hCount >= 1 && pCount === 0) return { cls: "mixed", label: "Mixed" };
    if (pCount >= 2) return { cls: "positive", label: "Positive" };
    return { cls: "neutral", label: "Neutral" };
  }

  function renderInsights() {
    const el = document.getElementById("insights-list");
    const allChats = loadChats().filter(c => c.messages && c.messages.length > 0);

    if (!allChats.length) {
      el.innerHTML = `<p class="empty-insights">Your insights will appear after your first conversation with Anchor.</p>`;
      return;
    }

    const sorted = [...allChats].sort((a, b) => b.updatedAt - a.updatedAt);
    el.innerHTML = sorted.map(chat => {
      const date = new Date(chat.updatedAt).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
      const turns = Math.floor(chat.messages.length / 2);
      const userText = chat.messages.filter(m => m.role === "user").map(m => m.content).join(" ");
      const mood = detectMood(userText);
      const firstMsg = chat.messages.find(m => m.role === "user")?.content || "";
      const preview = firstMsg.length > 130 ? firstMsg.slice(0, 130) + "…" : firstMsg;
      return `
        <div class="insight-card">
          <div class="insight-meta">
            <span class="insight-date">${date}</span>
            <span class="insight-turns">${turns} exchange${turns !== 1 ? "s" : ""}</span>
            <span class="mood-badge mood-${mood.cls}">${mood.label}</span>
          </div>
          <div class="insight-title">${escapeHtml(chat.title)}</div>
          <div class="insight-preview">${escapeHtml(preview)}</div>
        </div>
      `;
    }).join("");
  }

  // ── Contact modal ──────────────────────────────────────────────────────
  const contactModal = document.getElementById("contact-modal");
  document.getElementById("contact-btn").addEventListener("click", () => contactModal.classList.add("open"));
  document.getElementById("contact-close").addEventListener("click", () => contactModal.classList.remove("open"));
  contactModal.addEventListener("click", e => { if (e.target === contactModal) contactModal.classList.remove("open"); });

  // ── Feedback pulse ─────────────────────────────────────────────────────
  const feedbackBtn = document.getElementById("feedback-btn");
  setTimeout(() => {
    feedbackBtn.classList.add("pulsing");
    feedbackBtn.addEventListener("animationend", () => feedbackBtn.classList.remove("pulsing"), { once: true });
  }, 30000);

  const feedbackToast = document.getElementById("feedback-toast");
  let messagesSent = 0;
  let toastShown = false;

  // ── Init ───────────────────────────────────────────────────────────────
  createNewChat();
  loadModels();
})();
