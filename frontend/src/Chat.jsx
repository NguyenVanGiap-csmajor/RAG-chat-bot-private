import { useEffect, useRef, useState } from "react";
import { getHealth, sendMessage } from "./api";

const SUGGESTIONS = [
  "What are the main causes of environmental pollution discussed across the papers?",
  "Summarize the impacts of air, water, and soil pollution on ecosystems and human health.",
  "What sustainable solutions or mitigation strategies are recommended in the documents?",
  "How do policy, public engagement, and green technology help reduce environmental pollution?",
  "What does the microplastic paper say about identifying microplastics in aquatic systems?",
  "Compare the broad environmental pollution papers with the microplastic-focused study.",
];

export default function Chat() {
  const [messages, setMessages] = useState([
    {
      id: crypto.randomUUID(),
      role: "assistant",
      text: "Ask about pollution drivers, mitigation strategies, environmental policy, or microplastic detection. Use a suggested question to jump straight into the papers.",
      sources: [],
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [backendStatus, setBackendStatus] = useState("loading");
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  useEffect(() => {
    if (backendStatus !== "loading") {
      return undefined;
    }

    let isMounted = true;
    let timeoutId;

    const pollHealth = async () => {
      try {
        const health = await getHealth();
        if (!isMounted) {
          return;
        }

        const nextStatus = health.status ?? "loading";
        setBackendStatus(nextStatus);

        if (nextStatus === "error") {
          setError(health.error || "Backend warm-up failed.");
          return;
        }

        if (nextStatus !== "loading" || health.ready) {
          return;
        }
      } catch {
        if (!isMounted) {
          return;
        }
        setBackendStatus("loading");
      }

      timeoutId = window.setTimeout(pollHealth, 2000);
    };

    pollHealth();

    return () => {
      isMounted = false;
      window.clearTimeout(timeoutId);
    };
  }, [backendStatus]);

  const isBackendReady = backendStatus === "ready";
  const isComposerDisabled = isLoading || !isBackendReady;

  const handleSend = async (prefilledQuestion) => {
    const question = (prefilledQuestion ?? input).trim();
    if (!question || isComposerDisabled) {
      return;
    }

    setError("");
    setIsLoading(true);
    setInput("");

    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      text: question,
      sources: [],
    };

    setMessages((current) => [...current, userMessage]);

    try {
      const reply = await sendMessage(question);
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: reply.answer,
          sources: reply.sources ?? [],
        },
      ]);
    } catch (requestError) {
      if (requestError?.code === "ECONNABORTED") {
        setError("The backend is still processing your question. Please wait a little longer and try again.");
      } else {
        setError(
          requestError?.response?.data?.detail ||
            "Cannot connect to the backend. Check FastAPI and Ollama, then try again."
        );
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    await handleSend();
  };

  return (
    <main className="chat-shell">
      <section className="hero-panel">
        <div className="hero-content">
          <p className="eyebrow">Private Knowledge Base</p>
          <h1>Explore environmental research faster</h1>
        </div>

        <div className="suggestion-panel">
          <div className="suggestion-heading">
            <p className="chat-kicker">Quick Start</p>
            <h2>Suggested questions</h2>
          </div>

          <div className="suggestion-list">
            {SUGGESTIONS.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                className="suggestion-chip"
                onClick={() => handleSend(suggestion)}
                disabled={isComposerDisabled}
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className="chat-panel">
        <div className="chat-header">
          <div>
            <p className="chat-kicker">Live chat</p>
            <h2>Chat with your documents</h2>
          </div>
          <span className={`status-pill ${isLoading ? "busy" : isBackendReady ? "ready" : "warming"}`}>
            {isLoading ? "Thinking..." : isBackendReady ? "Ready" : "Preparing..."}
          </span>
        </div>

        {!isBackendReady && (
          <p className="info-banner">
            The backend is loading PDFs, chunking content, creating embeddings,
            and warming up the model. Chat will unlock automatically when
            everything is ready.
          </p>
        )}

        <div className="message-list">
          {messages.map((message) => (
            <article
              key={message.id}
              className={`message-card ${message.role === "user" ? "user" : "assistant"}`}
            >
              <div className="message-meta">
                <span>{message.role === "user" ? "You" : "Assistant"}</span>
              </div>
              <p>{message.text}</p>
              {message.sources?.length > 0 && (
                <div className="source-list">
                  {message.sources.map((source) => (
                    <span key={source} className="source-chip">
                      {source}
                    </span>
                  ))}
                </div>
              )}
            </article>
          ))}

          {isLoading && (
            <article className="message-card assistant loading-card">
              <div className="message-meta">
                <span>Assistant</span>
              </div>
              <div className="typing-dots" aria-label="Loading answer">
                <span />
                <span />
                <span />
              </div>
            </article>
          )}

          <div ref={messagesEndRef} />
        </div>

        {error && <p className="error-banner">{error}</p>}

        <form className="composer" onSubmit={handleSubmit}>
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder={
              isBackendReady
                ? "Ask a question about the PDFs in papers/..."
                : "Backend is preparing documents and models..."
            }
            rows={3}
            disabled={isComposerDisabled}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                handleSubmit(event);
              }
            }}
          />
          <div className="composer-footer">
            <p>{isBackendReady ? "Shift + Enter for a new line" : "Preparing backend..."}</p>
            <button type="submit" disabled={isComposerDisabled || !input.trim()}>
              Send message
            </button>
          </div>
        </form>
      </section>
    </main>
  );
}
