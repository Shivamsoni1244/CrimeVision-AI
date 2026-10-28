import React, { useMemo, useRef, useState } from 'react';
import BoxStage from './BoxStage.jsx';
import { Card, Empty, ErrorBox, FirearmNote, fmtTime } from './common.jsx';
import { api } from '../api.js';

export default function VideoResult({ v, events, caseId, onReanalyze, busy }) {
  const frames = v.frames || [];
  const [sel, setSel] = useState(0);
  const [playErr, setPlayErr] = useState(false);
  const player = useRef(null);
  const evs = useMemo(() => [...events].sort((a, b) => (a.t ?? 0) - (b.t ?? 0)), [events]);
  const frame = frames[sel];
  const jump = (e) => {
    let i = frames.findIndex((f) => f.frame_index === e.frame_index);
    if (i < 0 && e.t != null) i = frames.reduce((best, f, k) => (Math.abs(f.t - e.t) < Math.abs(frames[best].t - e.t) ? k : best), 0);
    if (i >= 0) setSel(i);
    if (player.current && e.t != null) { try { player.current.currentTime = e.t; } catch { /* not seekable */ } }
  };
  const s = v.summary || {};
  return (
    <Card title={v.file} className="result" actions={<button className="btn small" disabled={busy} onClick={() => onReanalyze(v.evidence_id)}>Re-run analysis</button>}>
      <p className="muted small">Evidence {v.evidence_id} · {v.video?.width}×{v.video?.height} · {v.video?.fps} fps · {s.samples} sampled frames every {s.sample_interval_s}s</p>
      {s.truncated && <div className="alert warn">Analysis stopped at the sample limit; footage after {fmtTime(s.analysed_until_s)} was not analysed.</div>}
      {v.warnings?.map((w) => <div className="alert warn" key={w}>{w}</div>)}
      <ErrorBox error={v.errors && Object.keys(v.errors).length ? `Some model steps failed: ${Object.entries(v.errors).map(([k, m]) => `${k} (${m})`).join('; ')}` : ''} />
      <FirearmNote info={v.firearm?.error ? { status: 'error', error: v.firearm.error } : v.firearm?.ran ? { status: 'ok', model: v.firearm_detector?.model, found: `${v.firearm.frames_with_firearm_class} sampled frame(s) with` } : v.firearm_detector} />
      <div className="cols">
        <div>
          <h4>Original video</h4>
          {playErr ? <p className="alert warn">This browser cannot play the original file (format/codec). Use the sampled frames instead.</p>
            : <video ref={player} className="player" controls preload="metadata" src={api.evidenceFileUrl(caseId, v.evidence_id)} onError={() => setPlayErr(true)} />}
        </div>
        <div>
          <h4>Sampled frame {frame ? `· ${fmtTime(frame.t)} (frame ${frame.frame_index})` : ''}</h4>
          {frame ? (
            <>
              <BoxStage src={api.frameUrl(caseId, v.evidence_id, frame.frame_ref)} alt={`Frame at ${frame.t}s`}
                        boxes={(frame.detections || []).filter((d) => d.relevant !== false)} />
              {frame.caption && <p className="small"><b>Caption:</b> {frame.caption}</p>}
              {frame.scene?.[0] && <p className="small muted">Top CLIP prompt: {frame.scene[0].label} ({Math.round(frame.scene[0].score * 100)}%, relative)</p>}
              {frame.error && <p className="alert error">{frame.error}</p>}
              <input type="range" min="0" max={Math.max(frames.length - 1, 0)} value={sel} onChange={(e) => setSel(+e.target.value)} aria-label="Frame" />
            </>
          ) : <Empty>No frames were stored.</Empty>}
        </div>
      </div>
      {s.text && <><h4>Summary <span className="tagline">generated from measurements, not an LLM</span></h4><p>{s.text}</p></>}
      <h4>Timeline ({evs.length} events)</h4>
      {evs.length ? (
        <ol className="timeline">
          {evs.map((e) => (
            <li key={e.event_id}>
              <button className="ev" onClick={() => jump(e)}>
                <span className="time">{fmtTime(e.t)}</span>
                <span className={`badge ${e.type}`}>{e.type.replace(/_/g, ' ')}</span>
                <span className="ev-text">{e.text}</span>
                <span className="muted small">{e.basis}{e.confidence != null ? ` · ${Math.round(e.confidence * 100)}%` : ''}{e.track_id ? ` · ${e.track_id}` : ''} · {e.event_id}</span>
              </button>
            </li>
          ))}
        </ol>
      ) : <Empty>No events were recorded for this video.</Empty>}
      {v.tracks?.length > 0 && (
        <details><summary>{v.tracks.length} tracked object(s)</summary>
          <table className="tbl"><thead><tr><th>Track</th><th>Class</th><th>Seen</th><th>Samples</th></tr></thead>
            <tbody>{v.tracks.map((t) => <tr key={t.track_id}><td>{t.track_id}</td><td>{t.label}</td><td>{fmtTime(t.first_t)}–{fmtTime(t.last_t)}</td><td>{t.observations}</td></tr>)}</tbody></table>
          <p className="muted small">Tracks link boxes between sampled frames. They are not identities of people.</p>
        </details>
      )}
      {v.limitations?.length > 0 && <details><summary>Limitations</summary><ul>{v.limitations.map((l) => <li key={l}>{l}</li>)}</ul></details>}
    </Card>
  );
}
