import React, { useEffect, useRef, useState } from 'react';
import { Card, Cites, ErrorBox, Spinner } from './common.jsx';
import { api } from '../api.js';

export default function ChatPanel({ caseId, disabled }) {
  const [msgs, setMsgs] = useState([]);
  const [q, setQ] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const end = useRef(null);
  useEffect(() => { setMsgs([]); setError(''); }, [caseId]);
  useEffect(() => { end.current?.scrollIntoView?.({ block: 'nearest' }); }, [msgs, busy]);
  const send = async (e) => {
    e.preventDefault();
    const question = q.trim();
    if (!question || busy || !caseId) return;
    setBusy(true); setError(''); setQ('');
    setMsgs((m) => [...m, { role: 'user', text: question }]);
    try {
      const r = await api.chat(caseId, question);
      setMsgs((m) => [...m, { role: 'assistant', text: r.answer, cites: r.citations, unverified: r.unverified_citations }]);
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  };
  return (
    <Card title="Evidence chat">
      <div className="banner">Answers are AI interpretations of this case's model outputs. Check every citation.</div>
      <div className="chat" aria-live="polite">
        {msgs.length === 0 && <p className="empty">Ask about this case's evidence.</p>}
        {msgs.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <p>{m.text}</p>
            {m.role === 'assistant' && <><Cites ids={m.cites} />{m.unverified?.length > 0 && <p className="small warn-text">Unverified references (not found in this case): {m.unverified.join(', ')}</p>}</>}
          </div>
        ))}
        {busy && <div className="msg assistant"><Spinner label="Thinking…" /></div>}
        <div ref={end} />
      </div>
      <ErrorBox error={error} />
      <form className="chat-form" onSubmit={send}>
        <input value={q} onChange={(e) => setQ(e.target.value)} maxLength={2000} placeholder={caseId ? 'Ask a question…' : 'Select a case first'} disabled={disabled || busy || !caseId} aria-label="Question" />
        <button className="btn primary" disabled={disabled || busy || !caseId || !q.trim()}>Send</button>
      </form>
    </Card>
  );
}
