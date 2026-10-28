import { useEffect, useRef, useState } from 'react';
import { api } from '../api/client.js';

export default function Chat({ caseId, onNeedCase }) {
  const [messages, setMessages] = useState([]);
  const [text, setText] = useState('');
  const [pending, setPending] = useState(false);
  const endRef = useRef(null);

  useEffect(() => { setMessages([]); }, [caseId]);
  useEffect(() => { endRef.current?.scrollIntoView({ block: 'nearest' }); }, [messages]);

  async function send(e) {
    e.preventDefault();
    const q = text.trim();
    if (!q) return;
    if (!caseId) return onNeedCase();
    setText('');
    setPending(true);
    setMessages((m) => [...m, { role: 'user', text: q }]);
    try {
      const r = await api.chat(caseId, q);
      setMessages((m) => [...m, {
        role: 'assistant', text: r.answer || '(empty answer)',
        citations: r.citations || [], unverified: r.unverified_citations || [],
      }]);
    } catch (err) {
      setMessages((m) => [...m, { role: 'error', text: err.message }]);
    } finally {
      setPending(false);
    }
  }

  return (
    <section className="panel">
      <h2>Ask the evidence</h2>
      <div className="chatlog" aria-live="polite">
        {messages.length === 0 && <p className="hint">Ask about people, objects or events in this case.</p>}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <p>{m.text}</p>
            {m.citations?.length > 0 && (
              <p className="cites">Citations: {m.citations.map((c) => <code key={c}>{c}</code>)}</p>
            )}
            {m.unverified?.length > 0 && (
              <p className="note warn">Unverified citations: {m.unverified.join(', ')}</p>
            )}
          </div>
        ))}
        {pending && <p className="hint">Thinking…</p>}
        <div ref={endRef} />
      </div>
      <form className="chatform" onSubmit={send}>
        <input value={text} onChange={(e) => setText(e.target.value)} placeholder="Who was present at 00:12?"
          aria-label="Question" disabled={pending} />
        <button type="submit" className="btn primary" disabled={pending}>Ask</button>
      </form>
    </section>
  );
}
