import React from 'react';
import { Card, Empty, fmtTime } from './common.jsx';

export default function AudioResult({ au, onReanalyze, busy }) {
  const segs = au.segments || [];
  return (
    <Card title={au.file || au.evidence_id} className="result" actions={<button className="btn small" disabled={busy} onClick={() => onReanalyze(au.evidence_id)}>Re-run</button>}>
      <p className="muted small">Evidence {au.evidence_id} · language: {au.language || 'unknown'} · Whisper {au.model}</p>
      <div className="banner">Machine transcript. Unverified; not a confirmed witness statement.</div>
      {segs.length ? (
        <ol className="segments">{segs.map((s, i) => <li key={i}><span className="time">{fmtTime(s.start)}–{fmtTime(s.end)}</span> {s.text}</li>)}</ol>
      ) : <Empty>{au.note || 'No speech segments returned.'}</Empty>}
      {au.translation_en && <details><summary>English translation (machine)</summary><p>{au.translation_en.text}</p></details>}
    </Card>
  );
}
