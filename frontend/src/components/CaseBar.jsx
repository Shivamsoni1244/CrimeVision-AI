import React, { useState } from 'react';
import { Spinner } from './common.jsx';

export default function CaseBar({ cases, caseId, onSelect, onCreate, onRefresh, busy, onReport, reporting }) {
  const [name, setName] = useState('');
  const submit = (e) => { e.preventDefault(); onCreate(name.trim()).then((ok) => ok && setName('')); };
  return (
    <div className="casebar">
      <label className="field">Case
        <select value={caseId} onChange={(e) => onSelect(e.target.value)} disabled={busy}>
          <option value="">{cases.length ? '— select a case —' : '— no cases yet —'}</option>
          {cases.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
      </label>
      <form className="newcase" onSubmit={submit}>
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="New case ID (optional)" maxLength={40} pattern="[A-Za-z0-9_-]*" title="Letters, digits, - and _" aria-label="New case ID" />
        <button className="btn" disabled={busy}>Create</button>
      </form>
      <button className="btn" onClick={onRefresh} disabled={busy}>Refresh</button>
      <button className="btn" onClick={onReport} disabled={!caseId || reporting}>{reporting ? <Spinner label="Building PDF…" /> : 'PDF report'}</button>
    </div>
  );
}
