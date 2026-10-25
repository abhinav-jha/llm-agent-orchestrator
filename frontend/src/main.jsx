import React, { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_ROOT = "/api/v1";
const TOKEN_KEY = "agent_studio_access_token";

async function apiRequest(path, { token, ...options } = {}) {
  const headers = new Headers(options.headers || {});
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (options.body && !(options.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(`${API_ROOT}${path}`, { ...options, headers });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      message = typeof body.detail === "string" ? body.detail : message;
    } catch {
      // Keep the HTTP status message when a response is not JSON.
    }
    throw new Error(message);
  }
  return response;
}

function AuthScreen({ onAuthenticated }) {
  const [mode, setMode] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event) {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      if (mode === "register") {
        await apiRequest("/auth/register", {
          method: "POST",
          body: JSON.stringify({ email, password }),
        });
      }
      const response = await apiRequest("/auth/login", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });
      const payload = await response.json();
      localStorage.setItem(TOKEN_KEY, payload.access_token);
      onAuthenticated(payload.access_token, email);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="auth-page">
      <section className="auth-card">
        <div className="brand-mark">A</div>
        <p className="eyebrow">LLM AGENT ORCHESTRATOR</p>
        <h1>{mode === "login" ? "Welcome back." : "Create your workspace."}</h1>
        <p className="muted">
          {mode === "login"
            ? "Sign in to continue your conversations."
            : "Create an account to start chatting with your agent."}
        </p>
        <form onSubmit={submit} className="auth-form">
          <label>
            Email
            <input
              autoComplete="email"
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              required
            />
          </label>
          <label>
            Password
            <input
              autoComplete={mode === "login" ? "current-password" : "new-password"}
              type="password"
              minLength={mode === "register" ? 8 : undefined}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              required
            />
          </label>
          {error && <div className="notice error">{error}</div>}
          <button className="primary-button full-width" disabled={busy}>
            {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
          </button>
        </form>
        <p className="switch-auth">
          {mode === "login" ? "New to Agent Studio?" : "Already have an account?"}{" "}
          <button
            className="text-button"
            onClick={() => {
              setError("");
              setMode(mode === "login" ? "register" : "login");
            }}
          >
            {mode === "login" ? "Create account" : "Sign in"}
          </button>
        </p>
      </section>
    </main>
  );
}

function App() {
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY) || "");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [email, setEmail] = useState("");
  const [conversations, setConversations] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [toolStatus, setToolStatus] = useState("");
  const bottomRef = useRef(null);
  const promptRef = useRef(null);

  const loadConversations = useCallback(async (activeToken = token) => {
    const response = await apiRequest("/conversations", { token: activeToken });
    const items = await response.json();
    setConversations(items);
    return items;
  }, [token]);

  const openConversation = useCallback(async (id, activeToken = token) => {
    if (!id) {
      setConversationId(null);
      setMessages([]);
      return;
    }
    setError("");
    try {
      const response = await apiRequest(`/conversations/${id}`, { token: activeToken });
      const detail = await response.json();
      setConversationId(detail.id);
      setMessages(detail.messages);
    } catch (requestError) {
      setError(requestError.message);
    }
  }, [token]);

  useEffect(() => {
    if (!token) return;
    let mounted = true;
    loadConversations(token)
      .then((items) => {
        if (mounted && items.length) return openConversation(items[0].id, token);
      })
      .catch((requestError) => {
        if (mounted) {
          if (requestError.message.includes("(401)")) signOut();
          else setError(requestError.message);
        }
      });
    return () => {
      mounted = false;
    };
  }, [token, loadConversations, openConversation]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, toolStatus]);

  function signOut() {
    localStorage.removeItem(TOKEN_KEY);
    setToken("");
    setEmail("");
    setConversations([]);
    setConversationId(null);
    setMessages([]);
  }

  function startNewConversation() {
    setConversationId(null);
    setMessages([]);
    setError("");
    setNotice("");
    setSidebarOpen(false);
    promptRef.current?.focus();
  }

  async function sendMessage(event) {
    event.preventDefault();
    const text = prompt.trim();
    if (!text || busy) return;

    setPrompt("");
    setError("");
    setNotice("");
    setToolStatus("");
    setBusy(true);
    setMessages((current) => [
      ...current,
      { id: `local-user-${Date.now()}`, role: "user", content: text },
      { id: `local-assistant-${Date.now()}`, role: "assistant", content: "" },
    ]);

    try {
      const response = await apiRequest("/chat/stream", {
        method: "POST",
        token,
        body: JSON.stringify({ prompt: text, conversation_id: conversationId }),
      });
      const returnedConversation = response.headers.get("X-Conversation-ID");
      if (returnedConversation) setConversationId(returnedConversation);
      if (!response.body) throw new Error("Streaming is not available in this browser.");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let streamError = "";
      let completed = false;

      function processFrame(frame) {
        const data = frame
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trimStart())
          .join("\n");
        if (!data) return;
        let event;
        try {
          event = JSON.parse(data);
        } catch {
          return;
        }
        if (event.type === "token") {
          setMessages((current) => {
            const next = [...current];
            const assistant = next[next.length - 1];
            if (assistant?.role === "assistant") {
              next[next.length - 1] = {
                ...assistant,
                content: (assistant.content || "") + event.content,
              };
            }
            return next;
          });
        } else if (event.type === "tool_start") {
          setToolStatus(`Using ${event.name.replaceAll("_", " ")}…`);
        } else if (event.type === "tool_complete") {
          setToolStatus("");
        } else if (event.type === "error") {
          streamError = event.message || "Agent request failed.";
        } else if (event.type === "done") {
          completed = true;
          setToolStatus("");
          setMessages((current) => {
            const next = [...current];
            const assistant = next[next.length - 1];
            if (assistant?.role === "assistant" && !assistant.content) {
              next[next.length - 1] = { ...assistant, content: event.content || "" };
            }
            return next;
          });
        }
      }

      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const frames = buffer.split(/\r?\n\r?\n/);
        buffer = frames.pop() || "";
        frames.forEach(processFrame);
        if (done) break;
      }
      if (buffer.trim()) processFrame(buffer);
      if (streamError) throw new Error(streamError);
      if (!completed) throw new Error("The response stream ended before completion.");
      const items = await loadConversations();
      if (returnedConversation) {
        const item = items.find((conversation) => conversation.id === returnedConversation);
        if (item) setConversationId(item.id);
      }
    } catch (requestError) {
      setError(requestError.message);
      setMessages((current) => {
        const next = [...current];
        const assistant = next[next.length - 1];
        if (assistant?.role === "assistant" && !assistant.content) {
          next.pop();
        }
        return next;
      });
    } finally {
      setBusy(false);
      setToolStatus("");
      promptRef.current?.focus();
    }
  }

  if (!token) {
    return (
      <AuthScreen
        onAuthenticated={(newToken, userEmail) => {
          setEmail(userEmail);
          setToken(newToken);
        }}
      />
    );
  }

  return (
    <main className={`app-shell ${sidebarOpen ? "sidebar-open" : ""}`}>
      {sidebarOpen && (
        <button
          className="sidebar-scrim"
          aria-label="Close conversations menu"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      <aside className="sidebar">
        <div className="sidebar-top">
          <div className="brand-lockup">
            <div className="brand-mark small">A</div>
            <div>
              <strong>Agent Studio</strong>
              <span>Gemini Flash</span>
            </div>
          </div>
          <button className="new-chat-button" onClick={startNewConversation}>
            <span>＋</span> New conversation
          </button>
          <p className="sidebar-label">YOUR CONVERSATIONS</p>
          <nav className="conversation-list" aria-label="Conversations">
            {conversations.map((conversation) => (
              <button
                key={conversation.id}
                className={`conversation-item ${conversationId === conversation.id ? "selected" : ""}`}
                onClick={() => {
                  setSidebarOpen(false);
                  openConversation(conversation.id);
                }}
              >
                <span className="conversation-icon">◷</span>
                <span className="conversation-title">{conversation.title}</span>
              </button>
            ))}
            {!conversations.length && (
              <p className="empty-conversations">Your conversations will appear here.</p>
            )}
          </nav>
        </div>
        <div className="sidebar-user">
          <div className="avatar">{(email[0] || "U").toUpperCase()}</div>
          <div className="user-email">{email || "Signed in"}</div>
          <button className="signout-button" onClick={signOut} aria-label="Sign out" title="Sign out">
            ↗
          </button>
        </div>
      </aside>

      <section className="chat-panel">
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="mobile-menu-button"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open conversations menu"
            >
              ☰
            </button>
            <div className="connection-state"><span /> Agent ready</div>
          </div>
          <div className="topbar-model">Gemini 2.5 Flash <span>·</span> Tool-enabled</div>
        </header>

        <section className="conversation-view" aria-live="polite">
          {messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-spark">✳</div>
              <p className="eyebrow">YOUR AI WORKSPACE</p>
              <h1>What can I help<br />you think through?</h1>
              <p className="welcome-copy">
                Ask a question, explore an idea, or pick up where you left off.
              </p>
              <div className="suggestions">
                <button onClick={() => setPrompt("Help me plan a productive week")}>
                  <span>✦</span><strong>Plan something</strong><small>Break a goal into clear steps</small>
                </button>
                <button onClick={() => setPrompt("Explain a complex topic in simple terms")}>
                  <span>⌘</span><strong>Learn something</strong><small>Make a challenging idea click</small>
                </button>
                <button onClick={() => setPrompt("Help me brainstorm some fresh ideas")}>
                  <span>◌</span><strong>Explore an idea</strong><small>Get unstuck and think creatively</small>
                </button>
              </div>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message) => (
                <article className={`message ${message.role}`} key={message.id}>
                  {message.role === "assistant" && <div className="assistant-avatar">✳</div>}
                  <div className="message-body">
                    <div className="message-role">{message.role === "user" ? "You" : "Assistant"}</div>
                    <div className="message-content">
                      {message.content || (busy && message.role === "assistant" ? <span className="typing-dots">Thinking<span>.</span><span>.</span><span>.</span></span> : "")}
                    </div>
                  </div>
                </article>
              ))}
              {toolStatus && <div className="tool-indicator"><span className="pulse-dot" />{toolStatus}</div>}
              <div ref={bottomRef} />
            </div>
          )}
        </section>

        <div className="composer-area">
          {error && <div className="notice error composer-notice">{error}</div>}
          {notice && <div className="notice composer-notice">{notice}</div>}
          <form className="composer" onSubmit={sendMessage}>
            <textarea
              ref={promptRef}
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  event.currentTarget.form.requestSubmit();
                }
              }}
              placeholder="Message your agent…"
              rows={1}
              maxLength={20000}
              disabled={busy}
            />
            <div className="composer-footer">
              <span>Enter to send · Shift + Enter for a new line</span>
              <button className="send-button" disabled={busy || !prompt.trim()} aria-label="Send message">
                {busy ? <span className="spinner" /> : "↑"}
              </button>
            </div>
          </form>
          <p className="disclaimer">Your assistant can make mistakes. Verify important information.</p>
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
