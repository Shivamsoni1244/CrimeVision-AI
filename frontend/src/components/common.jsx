import React from 'react';

export const fmtTime = (t) => (t == null ? 'untimed' : `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, '0')}`);
export const pct = (v) => `${Math.round(v * 100)}%`;

export function Spinner({ label }) {
  return <span className="spinner-wrap" role="status"><span className="spinner" aria-hidden="true" />{label}</span>;
}
export function ErrorBox({ error, onRetry }) {
  if (!error) return null;
  return (
    <div className="alert error" role="alert">
      <span>{error}</span>{onRetry && <button className="btn small" onClick={onRetry}>Retry</button>}
    </div>
  );
}
export function Empty({ children }) { return <p className="empty">{children}</p>; }
export function Card({ title, children, actions, className = '' }) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && <header className="card-head"><h2>{title}</h2><div className="card-actions">{actions}</div></header>}
      {children}
    </section>
  );
}
export function Cites({ ids }) {
  if (!ids?.length) return null;
  return <span className="cites">{ids.map((c) => <code key={c}>{c}</code>)}</span>;
}
const hue = (s) => [...s].reduce((a, c) => (a * 31 + c.charCodeAt(0)) % 360, 7);
export const labelColor = (label) => `hsl(${hue(label)} 75% 55%)`;

// Says what the firearm check actually did. "not configured" must never read as "no firearm present".
export function FirearmNote({ info }) {
  if (!info) return null;
  if (info.status === 'not_configured') return <div className="alert warn">Firearm detection is not enabled on the server. RT-DETR cannot detect firearms, so this result says nothing about whether one is present.</div>;
  if (info.status === 'error') return <div className="alert error">Firearm detection failed: {info.error}. Firearm presence is unknown.</div>;
  if (info.status === 'ok') return <p className="muted small">Firearm model <code>{info.model}</code> ran: {info.found} firearm-class detection(s). These are predictions and can be wrong, and a real firearm can be missed.</p>;
  return null;
}
