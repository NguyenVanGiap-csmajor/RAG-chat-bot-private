import { useEffect, useRef, useState } from "react";
import { sendMessage } from "./api";

const SUGGESTIONS = [
  "What are the main climate mitigation policy instruments discussed in the papers?",
  "How does urban biodiversity support ecosystem services?",
  "Summarize the key challenges mentioned across the documents.",
];

export default function Chat() {
  const [messages, setMessages] = useState([
    {
      id: crypto.randomUUID(),
      role: "assistant",
      text: "Ask about the PDF files in your knowledge base. I will answer only from the uploaded documents.",
      sources: [],
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleSend = async (prefilledQuestion) => {
    const question = (prefilledQuestion ?? input).trim();
    if (!question || isLoading) {
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
      setError(
        requestError?.response?.data?.detail ||
          "Cannot connect to the backend. Check FastAPI and Ollama, then try again."
      );
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
        <p className="eyebrow">Private Knowledge Base</p>
        <h1>RAG chatbot for your PDF papers</h1>

        <div className="suggestion-list">
          {SUGGESTIONS.map((suggestion) => (
            <button
              key={suggestion}
              type="button"
              className="suggestion-chip"
              onClick={() => handleSend(suggestion)}
              disabled={isLoading}
            >
              {suggestion}
            </button>
          ))}
        </div>
      </section>

      <section className="chat-panel">
        <div className="chat-header">
          <div>
            <p className="chat-kicker">Live chat</p>
            <h2>Chat with your documents</h2>
          </div>
          <span className={`status-pill ${isLoading ? "busy" : "ready"}`}>
            {isLoading ? "Thinking..." : "Ready"}
          </span>
        </div>

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
            placeholder="Ask a question about the PDFs in papers/..."
            rows={3}
            disabled={isLoading}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                handleSubmit(event);
              }
            }}
          />
          <div className="composer-footer">
            <p>Shift + Enter for a new line</p>
            <button type="submit" disabled={isLoading || !input.trim()}>
              Send message
            </button>
          </div>
        </form>
      </section>
    </main>
  );
}
