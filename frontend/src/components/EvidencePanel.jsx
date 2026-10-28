import { useEffect, useState } from 'react';
import { api } from '../api/client.js';

export default function EvidencePanel({ caseId, evidence, onNeedCase }) {
  const [integrity, setIntegrity] = useState(null);
  const [audit, setAudit] = useState(null);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');

  useEffect(() => { setIntegrity(null); setAudit(null); setError(''); }, [caseId]);

  async function run(name, fn, set) {
    if (!caseId) return onNeedCase();
    setBusy(name);
    setError('');
    try { set(await fn(caseId)); } catch (e) { setError(e.message); } finally { setBusy(''); }
  }

  const status = (id) => integrity?.find((r) => r.evidence_id === id);

  return (
    <section className="panel">
      <h2>Evidence &amp; integrity</h2>
      {evidence.length === 0 ? (
        <p className="hint">No evidence stored for this case.</p>
      ) : (
        <ul className="evlist">
          {evidence.map((e) => {
            const s = status(e.id);
            return (
              <li key={e.id}>
                <span className="badge">{e.kind}</span> <b>{e.filename}</b>
                <div className="mute small">
                  <code>{e.id}</code> · {(e.size / 1024).toFixed(0)} KB · sha256 <code>{e.sha256.slice(0, 12)}…</code>
                </div>
                {s && <div className={s.ok ? 'ok small' : 'weapon small'}>{s.ok ? 'Integrity verified' : `Integrity FAILED: ${s.reason}`}</div>}
              </li>
            );
          })}
        </ul>
      )}
      <div className="btnrow">
        <button type="button" className="btn" disabled={busy === 'integrity'}
          onClick={() => run('integrity', api.integrity, setIntegrity)}>
          {busy === 'integrity' ? 'Verifying…' : 'Verify integrity'}
        </button>
        <button type="button" className="btn" disabled={busy === 'audit'}
          onClick={() => run('audit', api.audit, setAudit)}>
          {busy === 'audit' ? 'Loading…' : 'Audit history'}
        </button>
      </div>
      {error && <p className="note error" role="alert">{error}</p>}
      {audit && (
        <ul className="audit">
          {audit.length === 0 && <li className="hint">No audit entries.</li>}
          {audit.map((a, i) => (
            <li key={i}><span className="mute small">{a.ts}</span> <b>{a.action}</b> <span className="small">{a.detail}</span></li>
          ))}
        </ul>
      )}
    </section>
  );
}
