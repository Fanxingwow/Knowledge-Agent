(() => {
  "use strict";

  const authStorage = {
    accessToken: "enterprise.console.access_token",
    refreshToken: "enterprise.console.refresh_token",
    idToken: "enterprise.console.id_token",
    expiresAt: "enterprise.console.expires_at",
    loginState: "enterprise.console.login_state",
    loginNonce: "enterprise.console.login_nonce",
    codeVerifier: "enterprise.console.code_verifier",
  };

  const state = {
    token: sessionStorage.getItem(authStorage.accessToken) || "",
    refreshToken: sessionStorage.getItem(authStorage.refreshToken) || "",
    idToken: sessionStorage.getItem(authStorage.idToken) || "",
    tokenExpiresAt: Number(sessionStorage.getItem(authStorage.expiresAt) || 0),
    tokenClaims: null,
    authConfig: null,
    refreshPromise: null,
    currentView: "chat",
    currentSessionId: null,
    pendingDeleteSessionId: null,
    sessions: [],
    sending: false,
    uploadTasks: [],
  };

  const elements = {
    sidebar: document.getElementById("sidebar"),
    sidebarBackdrop: document.getElementById("sidebarBackdrop"),
    sidebarClose: document.getElementById("sidebarClose"),
    mobileMenuButton: document.getElementById("mobileMenuButton"),
    newChatButton: document.getElementById("newChatButton"),
    refreshSessionsButton: document.getElementById("refreshSessionsButton"),
    sessionList: document.getElementById("sessionList"),
    authButton: document.getElementById("authButton"),
    authStatus: document.getElementById("authStatus"),
    authDialog: document.getElementById("authDialog"),
    authDialogCloseButton: document.getElementById("authDialogCloseButton"),
    authDialogDescription: document.getElementById("authDialogDescription"),
    authIdentity: document.getElementById("authIdentity"),
    authIdentityAvatar: document.getElementById("authIdentityAvatar"),
    authIdentityName: document.getElementById("authIdentityName"),
    authIdentityMeta: document.getElementById("authIdentityMeta"),
    loginButton: document.getElementById("loginButton"),
    logoutButton: document.getElementById("logoutButton"),
    deleteSessionDialog: document.getElementById("deleteSessionDialog"),
    deleteSessionName: document.getElementById("deleteSessionName"),
    cancelDeleteSessionButton: document.getElementById("cancelDeleteSessionButton"),
    confirmDeleteSessionButton: document.getElementById("confirmDeleteSessionButton"),
    viewTitle: document.getElementById("viewTitle"),
    viewSubtitle: document.getElementById("viewSubtitle"),
    chatView: document.getElementById("chatView"),
    knowledgeView: document.getElementById("knowledgeView"),
    welcomePanel: document.getElementById("welcomePanel"),
    messageList: document.getElementById("messageList"),
    chatScroll: document.getElementById("chatScroll"),
    messageInput: document.getElementById("messageInput"),
    sendButton: document.getElementById("sendButton"),
    openUploadButton: document.getElementById("openUploadButton"),
    serviceBadge: document.getElementById("serviceBadge"),
    knowledgeBadge: document.getElementById("knowledgeBadge"),
    importBadge: document.getElementById("importBadge"),
    refreshHealthButton: document.getElementById("refreshHealthButton"),
    dropZone: document.getElementById("dropZone"),
    fileInput: document.getElementById("fileInput"),
    selectFilesButton: document.getElementById("selectFilesButton"),
    uploadQueue: document.getElementById("uploadQueue"),
    searchForm: document.getElementById("searchForm"),
    searchInput: document.getElementById("searchInput"),
    searchResults: document.getElementById("searchResults"),
    toastRegion: document.getElementById("toastRegion"),
  };

  class ApiError extends Error {
    constructor(message, status, code = "REQUEST_FAILED") {
      super(message);
      this.status = status;
      this.code = code;
    }
  }

  function consoleRedirectUri() {
    return `${window.location.origin}/console`;
  }

  function oidcEndpoint(path) {
    const issuer = state.authConfig?.issuer?.replace(/\/$/, "");
    if (!issuer) throw new ApiError("企业登录尚未配置", 503, "OIDC_NOT_CONFIGURED");
    return `${issuer}/protocol/openid-connect/${path}`;
  }

  function randomBase64Url(byteLength = 32) {
    const bytes = crypto.getRandomValues(new Uint8Array(byteLength));
    let binary = "";
    bytes.forEach((value) => { binary += String.fromCharCode(value); });
    return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  }

  async function sha256Base64Url(value) {
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
    let binary = "";
    new Uint8Array(digest).forEach((byte) => { binary += String.fromCharCode(byte); });
    return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  }

  function decodeJwtClaims(token) {
    if (!token || token.split(".").length !== 3) return null;
    try {
      const encoded = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
      const padded = encoded.padEnd(Math.ceil(encoded.length / 4) * 4, "=");
      const bytes = Uint8Array.from(atob(padded), (character) => character.charCodeAt(0));
      return JSON.parse(new TextDecoder().decode(bytes));
    } catch (_error) {
      return null;
    }
  }

  function clearLoginTransaction() {
    sessionStorage.removeItem(authStorage.loginState);
    sessionStorage.removeItem(authStorage.loginNonce);
    sessionStorage.removeItem(authStorage.codeVerifier);
  }

  function clearOidcTokens() {
    state.token = "";
    state.refreshToken = "";
    state.idToken = "";
    state.tokenExpiresAt = 0;
    state.tokenClaims = null;
    [
      authStorage.accessToken,
      authStorage.refreshToken,
      authStorage.idToken,
      authStorage.expiresAt,
    ].forEach((key) => sessionStorage.removeItem(key));
  }

  function storeOidcTokens(payload) {
    if (!payload?.access_token) {
      throw new ApiError("登录服务没有返回完整凭证", 502, "OIDC_TOKEN_INVALID");
    }
    const expiresIn = Number(payload.expires_in || 0);
    if (!Number.isFinite(expiresIn) || expiresIn <= 0) {
      throw new ApiError("登录凭证有效期无效", 502, "OIDC_TOKEN_INVALID");
    }
    state.token = payload.access_token;
    state.refreshToken = payload.refresh_token || state.refreshToken;
    state.idToken = payload.id_token || state.idToken;
    state.tokenExpiresAt = Date.now() + expiresIn * 1000;
    state.tokenClaims = decodeJwtClaims(state.idToken) || decodeJwtClaims(state.token);
    sessionStorage.setItem(authStorage.accessToken, state.token);
    sessionStorage.setItem(authStorage.idToken, state.idToken);
    sessionStorage.setItem(authStorage.expiresAt, String(state.tokenExpiresAt));
    if (state.refreshToken) {
      sessionStorage.setItem(authStorage.refreshToken, state.refreshToken);
    }
  }

  async function requestOidcTokens(parameters) {
    const response = await fetch(oidcEndpoint("token"), {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams(parameters),
    });
    let payload = null;
    try {
      payload = await response.json();
    } catch (_error) {
      payload = null;
    }
    if (!response.ok) {
      throw new ApiError("企业登录失败，请重试", response.status, "OIDC_TOKEN_FAILED");
    }
    return payload;
  }

  function authorizationHeaders() {
    return state.token ? { Authorization: `Bearer ${state.token}` } : {};
  }

  function requestHeaders(extra = {}) {
    return { ...authorizationHeaders(), ...extra };
  }

  async function refreshAccessToken() {
    if (state.refreshPromise) return state.refreshPromise;
    if (!state.refreshToken || state.authConfig?.mode !== "oidc") return false;
    state.refreshPromise = (async () => {
      try {
        const payload = await requestOidcTokens({
          grant_type: "refresh_token",
          client_id: state.authConfig.client_id,
          refresh_token: state.refreshToken,
        });
        storeOidcTokens(payload);
        updateAuthStatus();
        return true;
      } catch (_error) {
        clearOidcTokens();
        updateAuthStatus();
        return false;
      } finally {
        state.refreshPromise = null;
      }
    })();
    return state.refreshPromise;
  }

  async function ensureFreshAccessToken() {
    if (state.authConfig?.mode !== "oidc") return;
    if (state.token && state.tokenExpiresAt > Date.now() + 30_000) return;
    if (await refreshAccessToken()) return;
    showAuthDialog();
    throw new ApiError("请先登录企业账户", 401, "AUTHENTICATION_REQUIRED");
  }

  async function authenticatedFetch(path, options = {}) {
    await ensureFreshAccessToken();
    return fetch(path, {
      ...options,
      headers: requestHeaders(options.headers || {}),
    });
  }

  async function parseResponse(response) {
    const raw = await response.text();
    let payload = null;
    if (raw) {
      try {
        payload = JSON.parse(raw);
      } catch (_error) {
        payload = null;
      }
    }
    if (!response.ok) {
      const message = payload?.message || `请求失败（HTTP ${response.status}）`;
      const error = new ApiError(message, response.status, payload?.code);
      if (response.status === 401) {
        if (state.authConfig?.mode === "oidc") {
          clearOidcTokens();
          updateAuthStatus();
        }
        showAuthDialog();
      }
      throw error;
    }
    return payload;
  }

  async function apiJson(path, options = {}) {
    const response = await authenticatedFetch(path, options);
    return parseResponse(response);
  }

  async function loadAuthConfig() {
    const response = await fetch("/console/auth-config", {
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    if (!response.ok) throw new ApiError("无法读取登录配置", response.status);
    const config = await response.json();
    if (!["disabled", "oidc"].includes(config.mode)) {
      throw new ApiError("登录配置无效", 503, "OIDC_CONFIG_INVALID");
    }
    if (config.mode === "oidc" && (!config.issuer || !config.client_id)) {
      throw new ApiError("企业登录尚未完成配置", 503, "OIDC_CONFIG_INVALID");
    }
    state.authConfig = config;
    state.tokenClaims = decodeJwtClaims(state.idToken) || decodeJwtClaims(state.token);
  }

  async function handleOidcCallback() {
    const current = new URL(window.location.href);
    const code = current.searchParams.get("code");
    const error = current.searchParams.get("error");
    if (!code && !error) return;
    const cleanUrl = consoleRedirectUri();
    if (state.authConfig?.mode !== "oidc") {
      window.history.replaceState({}, document.title, cleanUrl);
      throw new ApiError("企业登录尚未配置", 503, "OIDC_NOT_CONFIGURED");
    }
    if (error) {
      clearLoginTransaction();
      window.history.replaceState({}, document.title, cleanUrl);
      throw new ApiError("登录已取消或失败", 401, "OIDC_AUTHORIZATION_FAILED");
    }
    const returnedState = current.searchParams.get("state");
    const expectedState = sessionStorage.getItem(authStorage.loginState);
    const verifier = sessionStorage.getItem(authStorage.codeVerifier);
    const expectedNonce = sessionStorage.getItem(authStorage.loginNonce);
    if (!returnedState || returnedState !== expectedState || !verifier || !expectedNonce) {
      clearLoginTransaction();
      window.history.replaceState({}, document.title, cleanUrl);
      throw new ApiError("登录回调校验失败", 401, "OIDC_STATE_INVALID");
    }
    try {
      const payload = await requestOidcTokens({
        grant_type: "authorization_code",
        client_id: state.authConfig.client_id,
        code,
        redirect_uri: cleanUrl,
        code_verifier: verifier,
      });
      const idClaims = decodeJwtClaims(payload?.id_token);
      if (!idClaims || idClaims.nonce !== expectedNonce) {
        throw new ApiError("登录响应校验失败", 401, "OIDC_NONCE_INVALID");
      }
      storeOidcTokens(payload);
    } finally {
      clearLoginTransaction();
      window.history.replaceState({}, document.title, cleanUrl);
    }
  }

  async function beginLogin() {
    if (state.authConfig?.mode !== "oidc") {
      showToast("当前为开发模式，无需登录");
      return;
    }
    const verifier = randomBase64Url(64);
    const loginState = randomBase64Url(32);
    const nonce = randomBase64Url(32);
    sessionStorage.setItem(authStorage.codeVerifier, verifier);
    sessionStorage.setItem(authStorage.loginState, loginState);
    sessionStorage.setItem(authStorage.loginNonce, nonce);
    const authorizationUrl = new URL(oidcEndpoint("auth"));
    authorizationUrl.search = new URLSearchParams({
      response_type: "code",
      client_id: state.authConfig.client_id,
      redirect_uri: consoleRedirectUri(),
      scope: state.authConfig.scopes || "openid profile email",
      state: loginState,
      nonce,
      code_challenge: await sha256Base64Url(verifier),
      code_challenge_method: "S256",
    }).toString();
    window.location.assign(authorizationUrl.toString());
  }

  function logout() {
    const idToken = state.idToken;
    const oidcEnabled = state.authConfig?.mode === "oidc";
    clearOidcTokens();
    clearLoginTransaction();
    updateAuthStatus();
    if (!oidcEnabled || !idToken) {
      window.location.assign(consoleRedirectUri());
      return;
    }
    const logoutUrl = new URL(oidcEndpoint("logout"));
    logoutUrl.search = new URLSearchParams({
      client_id: state.authConfig.client_id,
      id_token_hint: idToken,
      post_logout_redirect_uri: consoleRedirectUri(),
    }).toString();
    window.location.assign(logoutUrl.toString());
  }

  function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast${type === "error" ? " is-error" : ""}`;
    toast.textContent = message;
    elements.toastRegion.appendChild(toast);
    window.setTimeout(() => toast.remove(), 4200);
  }

  function setBadge(element, stateName, label) {
    element.className = `status-badge is-${stateName}`;
    element.lastChild.textContent = label;
  }

  async function refreshHealth() {
    setBadge(elements.serviceBadge, "loading", "服务检测中");
    try {
      const response = await fetch("/health/ready", { headers: requestHeaders() });
      const payload = await response.json();
      const checks = payload.checks || {};
      setBadge(
        elements.serviceBadge,
        response.ok ? "ok" : "error",
        response.ok ? "服务正常" : "服务未就绪",
      );
      setBadge(
        elements.knowledgeBadge,
        checks.knowledge_search === "ok" ? "ok" : "error",
        checks.knowledge_search === "ok" ? "知识检索正常" : "知识检索不可用",
      );
      const importReady = [
        "import_registry",
        "import_processor",
        "import_workspace",
        "import_worker",
      ].every((key) => checks[key] === "ok");
      setBadge(
        elements.importBadge,
        importReady ? "ok" : "error",
        importReady ? "文档导入正常" : "文档导入不可用",
      );
    } catch (_error) {
      setBadge(elements.serviceBadge, "error", "服务连接失败");
      setBadge(elements.knowledgeBadge, "error", "知识检索未知");
      setBadge(elements.importBadge, "error", "文档导入未知");
    }
  }

  function switchView(view) {
    state.currentView = view;
    const chat = view === "chat";
    elements.chatView.classList.toggle("is-active", chat);
    elements.chatView.hidden = !chat;
    elements.knowledgeView.classList.toggle("is-active", !chat);
    elements.knowledgeView.hidden = chat;
    elements.viewTitle.textContent = chat ? "企业智能知识库问答" : "知识库";
    elements.viewSubtitle.textContent = chat
      ? "基于企业知识库的可信回答"
      : "文档导入、发布与检索验证";
    document.querySelectorAll(".nav-item").forEach((item) => {
      const active = item.dataset.view === view;
      item.classList.toggle("is-active", active);
      if (active) {
        item.setAttribute("aria-current", "page");
      } else {
        item.removeAttribute("aria-current");
      }
    });
    closeSidebar();
    if (!chat) {
      refreshHealth();
    }
  }

  function openSidebar() {
    elements.sidebar.classList.add("is-open");
    elements.sidebarBackdrop.hidden = false;
  }

  function closeSidebar() {
    elements.sidebar.classList.remove("is-open");
    elements.sidebarBackdrop.hidden = true;
  }

  function formatTime(value) {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "";
    return new Intl.DateTimeFormat("zh-CN", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    }).format(date);
  }

  function renderSessions() {
    elements.sessionList.replaceChildren();
    if (!state.sessions.length) {
      const empty = document.createElement("div");
      empty.className = "sidebar-placeholder";
      empty.textContent = "还没有会话，点击上方新建对话";
      elements.sessionList.appendChild(empty);
      return;
    }
    state.sessions.forEach((session) => {
      const item = document.createElement("div");
      item.className = "session-item";
      item.classList.toggle("is-active", session.session_id === state.currentSessionId);
      const button = document.createElement("button");
      button.type = "button";
      button.className = "session-select";
      const title = document.createElement("strong");
      title.textContent = session.title || "新会话";
      const time = document.createElement("small");
      time.textContent = formatTime(session.updated_at);
      button.append(title, time);
      button.addEventListener("click", () => selectSession(session.session_id));
      const deleteButton = document.createElement("button");
      deleteButton.type = "button";
      deleteButton.className = "session-delete";
      deleteButton.textContent = "删除";
      deleteButton.title = `删除会话：${session.title || "新会话"}`;
      deleteButton.setAttribute("aria-label", deleteButton.title);
      deleteButton.addEventListener("click", () => requestSessionDelete(session));
      item.append(button, deleteButton);
      elements.sessionList.appendChild(item);
    });
  }

  function requestSessionDelete(session) {
    if (state.sending) {
      showToast("请等待当前回答完成后再删除会话", "error");
      return;
    }
    state.pendingDeleteSessionId = session.session_id;
    elements.deleteSessionName.textContent = session.title || "新会话";
    if (!elements.deleteSessionDialog.open) elements.deleteSessionDialog.showModal();
  }

  function closeDeleteSessionDialog() {
    state.pendingDeleteSessionId = null;
    if (elements.deleteSessionDialog.open) elements.deleteSessionDialog.close();
  }

  async function deletePendingSession() {
    const sessionId = state.pendingDeleteSessionId;
    if (!sessionId) return;
    elements.confirmDeleteSessionButton.disabled = true;
    elements.confirmDeleteSessionButton.textContent = "正在删除…";
    try {
      await apiJson(`/api/v1/chat/sessions/${encodeURIComponent(sessionId)}`, {
        method: "DELETE",
      });
      state.sessions = state.sessions.filter((session) => session.session_id !== sessionId);
      if (state.currentSessionId === sessionId) {
        state.currentSessionId = null;
        clearMessages();
      }
      closeDeleteSessionDialog();
      renderSessions();
      showToast("会话已删除");
    } catch (error) {
      if (error.status !== 401) showToast(error.message || "删除会话失败", "error");
    } finally {
      elements.confirmDeleteSessionButton.disabled = false;
      elements.confirmDeleteSessionButton.textContent = "删除会话";
    }
  }

  async function loadSessions({ quiet = false } = {}) {
    try {
      const payload = await apiJson("/api/v1/chat/sessions?limit=50");
      state.sessions = payload?.items || [];
      renderSessions();
    } catch (error) {
      state.sessions = [];
      renderSessions();
      if (!quiet && error.status !== 401) showToast(error.message, "error");
    }
  }

  async function createSession() {
    const session = await apiJson("/api/v1/chat/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    state.currentSessionId = session.session_id;
    state.sessions.unshift(session);
    renderSessions();
    clearMessages();
    return session;
  }

  function clearMessages() {
    elements.messageList.replaceChildren();
    elements.welcomePanel.classList.remove("is-hidden");
  }

  function citationTitle(citation, index) {
    return citation.title || citation.file_title || citation.item_name || `参考资料 ${index}`;
  }

  function renderCitations(container, citations) {
    const valid = Array.isArray(citations) ? citations : [];
    if (!valid.length) return;
    const list = document.createElement("div");
    list.className = "citation-list";
    valid.forEach((citation, offset) => {
      const card = document.createElement("div");
      card.className = "citation-card";
      const title = document.createElement("strong");
      const index = citation.index || offset + 1;
      title.textContent = `[${index}] ${citationTitle(citation, index)}`;
      const excerpt = document.createElement("p");
      excerpt.textContent = citation.content || citation.citation_id || "引用来源";
      card.append(title, excerpt);
      list.appendChild(card);
    });
    container.appendChild(list);
  }

  function createMessageElement(role, content = "", options = {}) {
    const row = document.createElement("article");
    row.className = `message-row${role === "user" ? " is-user" : " is-assistant"}`;
    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? "我" : "AI";
    const body = document.createElement("div");
    body.className = "message-body";
    const meta = document.createElement("div");
    meta.className = "message-meta";
    const name = document.createElement("strong");
    name.textContent = role === "user" ? "你" : "企业智能知识库问答";
    const time = document.createElement("span");
    time.textContent = options.createdAt ? formatTime(options.createdAt) : "刚刚";
    meta.append(name, time);
    const text = document.createElement("div");
    text.className = "message-content";
    text.textContent = content;
    body.append(meta, text);
    row.append(avatar, body);
    elements.messageList.appendChild(row);
    elements.welcomePanel.classList.add("is-hidden");
    if (options.citations) renderCitations(body, options.citations);
    return { row, body, text };
  }

  function showTyping(message) {
    message.text.replaceChildren();
    const indicator = document.createElement("span");
    indicator.className = "typing-indicator";
    indicator.setAttribute("aria-label", "Agent 正在生成回答");
    indicator.append(document.createElement("span"), document.createElement("span"), document.createElement("span"));
    message.text.appendChild(indicator);
  }

  function scrollChatToBottom() {
    window.requestAnimationFrame(() => {
      window.scrollTo({ top: document.documentElement.scrollHeight, behavior: "smooth" });
    });
  }

  async function selectSession(sessionId) {
    if (state.sending) return;
    state.currentSessionId = sessionId;
    renderSessions();
    switchView("chat");
    elements.messageList.replaceChildren();
    try {
      const payload = await apiJson(`/api/v1/chat/sessions/${encodeURIComponent(sessionId)}/history?limit=100`);
      const messages = payload?.items || [];
      if (!messages.length) {
        elements.welcomePanel.classList.remove("is-hidden");
        return;
      }
      elements.welcomePanel.classList.add("is-hidden");
      messages.forEach((message) => {
        if (!["user", "assistant"].includes(message.role)) return;
        const attributes = message.attributes || {};
        const citations = (attributes.citation_ids || "")
          .split(",")
          .filter(Boolean)
          .map((citationId, index) => ({ citation_id: citationId, index: index + 1 }));
        createMessageElement(message.role, message.content, {
          createdAt: message.created_at,
          citations,
        });
      });
      scrollChatToBottom();
    } catch (error) {
      showToast(error.message, "error");
    }
  }

  function resizeComposer() {
    const visualLines = elements.messageInput.value
      .split("\n")
      .reduce((count, line) => count + Math.max(1, Math.ceil(line.length / 72)), 0);
    elements.messageInput.rows = Math.min(7, Math.max(1, visualLines));
  }

  function parseSseBlock(block) {
    const event = { id: null, type: "message", data: "" };
    const dataLines = [];
    block.split(/\r?\n/).forEach((line) => {
      if (line.startsWith("id:")) event.id = line.slice(3).trim();
      if (line.startsWith("event:")) event.type = line.slice(6).trim();
      if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    });
    event.data = dataLines.join("\n");
    return event;
  }

  async function consumeSse(response, onEvent) {
    if (!response.body) throw new ApiError("浏览器不支持流式响应", 500);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const blocks = buffer.split(/\r?\n\r?\n/);
      buffer = blocks.pop() || "";
      for (const block of blocks) {
        if (block.trim()) await onEvent(parseSseBlock(block));
      }
      if (done) break;
    }
    if (buffer.trim()) await onEvent(parseSseBlock(buffer));
  }

  async function sendMessage(prefilled = null) {
    const content = (prefilled ?? elements.messageInput.value).trim();
    if (!content || state.sending) return;
    state.sending = true;
    elements.sendButton.disabled = true;
    elements.messageInput.disabled = true;
    let assistant = null;
    try {
      if (!state.currentSessionId) await createSession();
      createMessageElement("user", content);
      assistant = createMessageElement("assistant");
      showTyping(assistant);
      elements.messageInput.value = "";
      resizeComposer();
      scrollChatToBottom();

      const messageId = crypto.randomUUID();
      const response = await authenticatedFetch(
        `/api/v1/chat/sessions/${encodeURIComponent(state.currentSessionId)}/responses`,
        {
          method: "POST",
          headers: {
            Accept: "text/event-stream",
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ message_id: messageId, content, top_k: 5 }),
        },
      );
      if (!response.ok) await parseResponse(response);
      let answer = "";
      let citations = [];
      let completed = false;
      await consumeSse(response, async (event) => {
        let payload = {};
        try {
          payload = event.data ? JSON.parse(event.data) : {};
        } catch (_error) {
          return;
        }
        if (event.type === "retrieval.completed") {
          citations = payload.citations || [];
        } else if (event.type === "message.delta") {
          if (!answer) assistant.text.replaceChildren();
          answer += payload.delta || "";
          assistant.text.textContent = answer;
          scrollChatToBottom();
        } else if (event.type === "message.completed") {
          completed = true;
          answer = payload.message?.content || answer;
          assistant.text.textContent = answer;
          citations = payload.citations?.length ? payload.citations : citations;
          renderCitations(assistant.body, citations);
          scrollChatToBottom();
        } else if (event.type === "run.failed") {
          throw new ApiError(payload.message || "Agent 响应失败", 503, payload.code);
        }
      });
      if (!completed) throw new ApiError("Agent 流式响应意外结束", 503);
      await loadSessions({ quiet: true });
    } catch (error) {
      if (assistant) {
        assistant.text.textContent = error.message || "Agent 响应失败，请稍后重试。";
      }
      if (error.status !== 401) showToast(error.message || "发送失败", "error");
    } finally {
      state.sending = false;
      elements.sendButton.disabled = false;
      elements.messageInput.disabled = false;
      elements.messageInput.focus();
    }
  }

  function acceptedFile(file) {
    const name = file.name.toLowerCase();
    return name.endsWith(".pdf") || name.endsWith(".md") || name.endsWith(".markdown");
  }

  function taskProgress(status) {
    return { uploading: 10, queued: 25, processing: 68, completed: 100, failed: 100 }[status] || 10;
  }

  function statusLabel(status) {
    return {
      uploading: "上传中",
      queued: "等待处理",
      processing: "解析与索引中",
      completed: "已发布",
      failed: "处理失败",
    }[status] || status;
  }

  function renderUploadQueue() {
    elements.uploadQueue.replaceChildren();
    if (!state.uploadTasks.length) {
      const empty = document.createElement("div");
      empty.className = "empty-state compact";
      empty.textContent = "暂时没有导入任务";
      elements.uploadQueue.appendChild(empty);
      return;
    }
    state.uploadTasks.forEach((task) => {
      const item = document.createElement("div");
      item.className = "upload-task";
      const mark = document.createElement("div");
      mark.className = "file-mark";
      mark.textContent = task.name.toLowerCase().endsWith(".pdf") ? "PDF" : "MD";
      const copy = document.createElement("div");
      copy.className = "task-copy";
      const name = document.createElement("strong");
      name.textContent = task.name;
      const detail = document.createElement("small");
      detail.textContent = task.error || task.step || "企业知识导入链路";
      const progress = document.createElement("progress");
      progress.max = 100;
      progress.value = taskProgress(task.status);
      copy.append(name, detail, progress);
      const status = document.createElement("span");
      status.className = `task-status is-${task.status}`;
      status.textContent = statusLabel(task.status);
      item.append(mark, copy, status);
      elements.uploadQueue.appendChild(item);
    });
  }

  async function pollImportTask(task) {
    const deadline = Date.now() + 15 * 60 * 1000;
    while (Date.now() < deadline && !["completed", "failed"].includes(task.status)) {
      await new Promise((resolve) => window.setTimeout(resolve, 2500));
      try {
        const payload = await apiJson(`/api/v1/imports/${encodeURIComponent(task.taskId)}`);
        task.status = payload.status;
        task.step = payload.progress?.current_step || "正在处理";
        task.error = payload.error_code || "";
        renderUploadQueue();
      } catch (error) {
        task.status = "failed";
        task.error = error.message;
        renderUploadQueue();
      }
    }
    if (task.status === "completed") {
      showToast(`“${task.name}”已发布到知识库`);
      refreshHealth();
    }
  }

  async function uploadFiles(files) {
    const selected = Array.from(files).filter((file) => file.size > 0 && acceptedFile(file));
    if (!selected.length) {
      showToast("请选择 PDF 或 Markdown 文件", "error");
      return;
    }
    switchView("knowledge");
    const localTasks = selected.map((file) => ({
      localId: crypto.randomUUID(),
      name: file.name,
      status: "uploading",
      step: "正在上传到对象存储",
      error: "",
      taskId: null,
    }));
    state.uploadTasks.unshift(...localTasks);
    renderUploadQueue();
    const formData = new FormData();
    selected.forEach((file) => formData.append("files", file, file.name));
    try {
      const response = await authenticatedFetch("/api/v1/imports", {
        method: "POST",
        headers: { "Idempotency-Key": `console-${crypto.randomUUID()}` },
        body: formData,
      });
      const payload = await parseResponse(response);
      const acceptedTasks = payload?.tasks || [];
      localTasks.forEach((task, index) => {
        const accepted = acceptedTasks[index];
        if (!accepted) {
          task.status = "failed";
          task.error = "服务未返回对应导入任务";
          return;
        }
        task.taskId = accepted.task_id;
        task.status = accepted.status;
        task.step = accepted.progress?.current_step || "等待导入 Worker";
      });
      renderUploadQueue();
      localTasks.filter((task) => task.taskId).forEach((task) => pollImportTask(task));
    } catch (error) {
      localTasks.forEach((task) => {
        task.status = "failed";
        task.error = error.message || "上传失败";
      });
      renderUploadQueue();
      if (error.status !== 401) showToast(error.message || "上传失败", "error");
    } finally {
      elements.fileInput.value = "";
    }
  }

  function renderSearchResults(hits) {
    elements.searchResults.replaceChildren();
    if (!hits.length) {
      const empty = document.createElement("div");
      empty.className = "empty-state";
      const icon = document.createElement("span");
      icon.textContent = "⌕";
      const title = document.createElement("strong");
      title.textContent = "没有检索到结果";
      const copy = document.createElement("p");
      copy.textContent = "请确认文档已完成发布，或换一个更具体的关键词。";
      empty.append(icon, title, copy);
      elements.searchResults.appendChild(empty);
      return;
    }
    hits.forEach((hit, index) => {
      const card = document.createElement("article");
      card.className = "search-hit";
      const heading = document.createElement("div");
      heading.className = "hit-heading";
      const title = document.createElement("strong");
      title.textContent = `[${index + 1}] ${hit.metadata?.title || hit.metadata?.file_title || "知识片段"}`;
      const score = document.createElement("span");
      score.textContent = `相关度 ${Number(hit.score || 0).toFixed(3)}`;
      heading.append(title, score);
      const content = document.createElement("p");
      content.textContent = hit.content || "";
      card.append(heading, content);
      elements.searchResults.appendChild(card);
    });
  }

  async function searchKnowledge(event) {
    event.preventDefault();
    const query = elements.searchInput.value.trim();
    if (!query) return;
    elements.searchResults.replaceChildren();
    const loading = document.createElement("div");
    loading.className = "empty-state compact";
    loading.textContent = "正在进行 BGE-M3 混合检索…";
    elements.searchResults.appendChild(loading);
    try {
      const payload = await apiJson("/api/v1/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 5 }),
      });
      renderSearchResults(payload?.hits || []);
    } catch (error) {
      renderSearchResults([]);
      if (error.status !== 401) showToast(error.message, "error");
    }
  }

  function showAuthDialog() {
    updateAuthStatus();
    if (!elements.authDialog.open) elements.authDialog.showModal();
  }

  function updateAuthStatus() {
    const oidcEnabled = state.authConfig?.mode === "oidc";
    const claims = state.tokenClaims || decodeJwtClaims(state.idToken) || decodeJwtClaims(state.token);
    const signedIn = oidcEnabled && Boolean(state.token && claims?.sub);
    if (!oidcEnabled) {
      elements.authStatus.textContent = "开发模式 / 无需登录";
      elements.authDialogDescription.textContent = "当前环境未启用企业登录。生产环境应切换到 Keycloak OIDC。";
      elements.authIdentity.hidden = true;
      elements.loginButton.textContent = "关闭";
      elements.loginButton.hidden = false;
      elements.logoutButton.hidden = true;
      return;
    }
    if (!signedIn) {
      elements.authStatus.textContent = "未登录";
      elements.authDialogDescription.textContent = "使用企业账户登录后，你的会话和问题将与其他用户独立保存。";
      elements.authIdentity.hidden = true;
      elements.loginButton.textContent = "登录企业账户";
      elements.loginButton.hidden = false;
      elements.logoutButton.hidden = true;
      return;
    }
    const displayName = claims.name || claims.preferred_username || claims.email || "企业用户";
    const meta = claims.email || `用户标识 ${String(claims.sub).slice(0, 12)}`;
    elements.authStatus.textContent = displayName;
    elements.authDialogDescription.textContent = "当前账户的会话记录、问题和回答仅对该账户可见。";
    elements.authIdentityName.textContent = displayName;
    elements.authIdentityMeta.textContent = meta;
    elements.authIdentityAvatar.textContent = String(displayName).trim().slice(0, 1) || "企";
    elements.authIdentity.hidden = false;
    elements.loginButton.hidden = true;
    elements.logoutButton.hidden = false;
  }

  function bindEvents() {
    document.querySelectorAll(".nav-item").forEach((button) => {
      button.addEventListener("click", () => switchView(button.dataset.view));
    });
    document.querySelectorAll(".suggestion-card").forEach((button) => {
      button.addEventListener("click", () => sendMessage(button.dataset.prompt));
    });
    elements.mobileMenuButton.addEventListener("click", openSidebar);
    elements.sidebarClose.addEventListener("click", closeSidebar);
    elements.sidebarBackdrop.addEventListener("click", closeSidebar);
    elements.newChatButton.addEventListener("click", async () => {
      try {
        await createSession();
        switchView("chat");
        elements.messageInput.focus();
      } catch (error) {
        if (error.status !== 401) showToast(error.message, "error");
      }
    });
    elements.refreshSessionsButton.addEventListener("click", () => loadSessions());
    elements.cancelDeleteSessionButton.addEventListener("click", closeDeleteSessionDialog);
    elements.confirmDeleteSessionButton.addEventListener("click", deletePendingSession);
    elements.deleteSessionDialog.addEventListener("cancel", (event) => {
      event.preventDefault();
      closeDeleteSessionDialog();
    });
    elements.sendButton.addEventListener("click", () => sendMessage());
    elements.messageInput.addEventListener("input", resizeComposer);
    elements.messageInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        sendMessage();
      }
    });
    elements.openUploadButton.addEventListener("click", () => switchView("knowledge"));
    elements.refreshHealthButton.addEventListener("click", refreshHealth);
    elements.selectFilesButton.addEventListener("click", (event) => {
      event.stopPropagation();
      elements.fileInput.click();
    });
    elements.dropZone.addEventListener("click", () => elements.fileInput.click());
    elements.dropZone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        elements.fileInput.click();
      }
    });
    ["dragenter", "dragover"].forEach((name) => {
      elements.dropZone.addEventListener(name, (event) => {
        event.preventDefault();
        elements.dropZone.classList.add("is-dragging");
      });
    });
    ["dragleave", "drop"].forEach((name) => {
      elements.dropZone.addEventListener(name, (event) => {
        event.preventDefault();
        elements.dropZone.classList.remove("is-dragging");
      });
    });
    elements.dropZone.addEventListener("drop", (event) => uploadFiles(event.dataTransfer.files));
    elements.fileInput.addEventListener("change", () => uploadFiles(elements.fileInput.files));
    elements.searchForm.addEventListener("submit", searchKnowledge);
    elements.authButton.addEventListener("click", showAuthDialog);
    elements.authDialogCloseButton.addEventListener("click", () => elements.authDialog.close());
    elements.loginButton.addEventListener("click", async () => {
      if (state.authConfig?.mode === "oidc") {
        await beginLogin();
      } else {
        elements.authDialog.close();
      }
    });
    elements.logoutButton.addEventListener("click", logout);
  }

  async function initialize() {
    bindEvents();
    resizeComposer();
    try {
      await loadAuthConfig();
      await handleOidcCallback();
      if (state.authConfig.mode === "oidc" && state.token) {
        await ensureFreshAccessToken();
      }
      updateAuthStatus();
      if (state.authConfig.mode === "oidc" && !state.token) {
        showAuthDialog();
      }
      await Promise.all([refreshHealth(), loadSessions({ quiet: true })]);
    } catch (error) {
      clearOidcTokens();
      updateAuthStatus();
      showAuthDialog();
      showToast(error.message || "企业登录初始化失败", "error");
      await refreshHealth();
    }
  }

  initialize();
})();
